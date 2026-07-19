import requests
import pandas as pd
from datetime import date, timedelta
from urllib.parse import quote
from pathlib import Path

# =============================================================================
# USDA AMS MPR Datamart: National Dairy Products Sales Report
#
# Downloads weekly real-world dairy sales data and checks whether each product
# series is a continuous weekly time series.
#
# Important update:
#   - Checks whether the product's sales target column contains missing values.
#   - Keeps the longest consecutive weekly block with non-missing sales values.
#   - Saves raw cleaned downloads separately from model-ready consecutive data.
#
# Goal:
#   Use as many consecutive timesteps as possible without missing target values.
# =============================================================================

BASE = "https://mpr.datamart.ams.usda.gov/services/v1.1/reports/2993"

SECTIONS = {
    "cheddar_40lb_blocks": "Final 40 Pound Block Cheddar Cheese Prices and Sales",
    "dry_whey": "Final Dry Whey Prices and Sales",
    "nonfat_dry_milk": "Final Nonfat Dry Milk Prices and Sales",
}

# Expected target sales columns after column-name cleaning.
# The fallback function below also tries to infer the sales column if one of
# these names differs slightly in the USDA payload.
TARGET_SALES_COLUMNS = {
    "cheddar_40lb_blocks": "cheese_40_sales",
    "dry_whey": "dry_whey_sales",
    "nonfat_dry_milk": "nonfat_dry_milk_sales",
}

EXPECTED_PRICE_COLUMNS = {
    "cheddar_40lb_blocks": "cheese_40_price",
    "dry_whey": "dry_whey_price",
    "nonfat_dry_milk": "nonfat_dry_milk_price",
}

START = date(2001, 4, 1)
END = date.today()

OUTPUT_DIR = Path("data")
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw_downloads"
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_OUTPUT_DIR.mkdir(exist_ok=True)


def fmt(d):
    return d.strftime("%m/%d/%Y")


def extract_rows(payload):
    """
    Recursively extracts dictionaries that contain a week/date field.
    This is robust to nested USDA API payload structures.
    """
    rows = []

    def walk(x):
        if isinstance(x, dict):
            lower_keys = {str(k).lower() for k in x.keys()}

            if any("week" in k and "date" in k for k in lower_keys):
                rows.append(x)

            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(payload)
    return rows


def clean_column_names(df):
    """
    Cleans column names and removes duplicated columns created by normalization.

    Example:
        'Week Ending Date' and 'week-ending-date' may both become
        'week_ending_date'.
    """
    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )

    if df.columns.duplicated().any():
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        print("\nDuplicate columns after cleaning:")
        print(duplicate_cols)

        # Keep first occurrence of each duplicated column name.
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def find_and_standardize_week_date_column(df, product):
    """
    Finds the weekly date column and renames it to 'week_ending_date'.
    """
    date_candidates = [
        col for col in df.columns
        if "week" in col and "date" in col
    ]

    if not date_candidates:
        raise ValueError(
            f"{product}: Could not find a weekly date column. "
            f"Available columns are: {list(df.columns)}"
        )

    # Prefer exact standardized name if available.
    if "week_ending_date" in date_candidates:
        date_col = "week_ending_date"
    else:
        date_col = date_candidates[0]

    if date_col != "week_ending_date":
        df = df.rename(columns={date_col: "week_ending_date"})

    return df


def parse_numeric_series(series):
    """
    Converts numeric USDA fields that may contain commas into floats.

    Examples:
        "6,118,090" -> 6118090.0
        ""          -> NaN
        None        -> NaN
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )

    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "null": pd.NA,
        }
    )

    return pd.to_numeric(cleaned, errors="coerce")


def find_sales_column(df, product):
    """
    Finds the target sales column for a product.

    First tries the expected mapping. If that fails, falls back to detecting
    columns that contain 'sales'.
    """
    expected = TARGET_SALES_COLUMNS.get(product)

    if expected in df.columns:
        return expected

    sales_candidates = [
        col for col in df.columns
        if "sales" in col.lower()
    ]

    # Prefer columns ending exactly with "_sales".
    exact_like = [
        col for col in sales_candidates
        if col.lower().endswith("_sales")
    ]

    if len(exact_like) == 1:
        print(
            f"{product}: expected sales column '{expected}' not found. "
            f"Using inferred column '{exact_like[0]}'."
        )
        return exact_like[0]

    if len(sales_candidates) == 1:
        print(
            f"{product}: expected sales column '{expected}' not found. "
            f"Using inferred column '{sales_candidates[0]}'."
        )
        return sales_candidates[0]

    raise ValueError(
        f"{product}: Could not uniquely identify sales column. "
        f"Expected '{expected}'. Sales candidates: {sales_candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def find_price_column(df, product):
    """
    Finds a price column for diagnostics only.
    """
    expected = EXPECTED_PRICE_COLUMNS.get(product)

    if expected in df.columns:
        return expected

    price_candidates = [
        col for col in df.columns
        if "price" in col.lower()
    ]

    if len(price_candidates) == 1:
        return price_candidates[0]

    return None


def choose_one_row_per_week(df, target_col):
    """
    If duplicate week rows exist, keep one row per week.

    Preference:
      1. Row with non-missing target value.
      2. Latest published_date if available.
      3. Latest created_date if available.
      4. First remaining row.

    This prevents duplicate weeks from breaking the time series while preserving
    the best available target value.
    """
    df = df.copy()

    if not df.duplicated(subset=["week_ending_date"]).any():
        return df

    print("\nDuplicate weekly dates detected. Selecting best row per week.")

    df["_target_missing_for_selection"] = df[target_col].isna()

    if "published_date" in df.columns:
        df["_published_dt_for_selection"] = pd.to_datetime(
            df["published_date"],
            errors="coerce",
        )
    else:
        df["_published_dt_for_selection"] = pd.NaT

    if "created_date" in df.columns:
        df["_created_dt_for_selection"] = pd.to_datetime(
            df["created_date"],
            errors="coerce",
        )
    else:
        df["_created_dt_for_selection"] = pd.NaT

    df = df.sort_values(
        by=[
            "week_ending_date",
            "_target_missing_for_selection",
            "_published_dt_for_selection",
            "_created_dt_for_selection",
        ],
        ascending=[True, True, False, False],
        na_position="last",
    )

    before = len(df)
    df = df.drop_duplicates(subset=["week_ending_date"], keep="first")
    after = len(df)

    print(f"Dropped duplicate weekly rows: {before - after}")

    df = df.drop(
        columns=[
            "_target_missing_for_selection",
            "_published_dt_for_selection",
            "_created_dt_for_selection",
        ],
        errors="ignore",
    )

    return df


def check_weekly_continuity(df, product, date_col="week_ending_date"):
    """
    Checks whether the provided product series is a continuous weekly time series.
    Returns a summary dict and prints human-readable diagnostics.
    """
    if date_col not in df.columns:
        raise ValueError(f"{product}: missing required date column '{date_col}'")

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])

    duplicate_count = temp.duplicated(subset=[date_col]).sum()

    unique_dates = (
        temp[date_col]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if unique_dates.empty:
        print("\n" + "=" * 80)
        print(f"Continuity check for {product}")
        print("=" * 80)
        print("No valid weekly dates found.")

        return {
            "sample_size_rows": len(df),
            "sample_size_unique_weeks": 0,
            "start_date": None,
            "end_date": None,
            "expected_weeks": 0,
            "missing_weeks": 0,
            "duplicate_week_rows": int(duplicate_count),
            "is_continuous_weekly": False,
            "min_gap_days": None,
            "max_gap_days": None,
        }

    start_date = unique_dates.iloc[0]
    end_date = unique_dates.iloc[-1]

    expected_dates = pd.date_range(start=start_date, end=end_date, freq="7D")
    missing_dates = expected_dates.difference(unique_dates)

    diffs = unique_dates.diff().dropna().dt.days
    all_gaps_are_7_days = bool((diffs == 7).all()) if len(diffs) > 0 else True

    is_continuous = (
        len(missing_dates) == 0
        and all_gaps_are_7_days
        and len(unique_dates) == len(expected_dates)
    )

    print("\n" + "=" * 80)
    print(f"Continuity check for {product}")
    print("=" * 80)
    print(f"Start date:              {start_date.date()}")
    print(f"End date:                {end_date.date()}")
    print(f"Rows in CSV:             {len(df)}")
    print(f"Unique weekly dates:     {len(unique_dates)}")
    print(f"Expected weekly dates:   {len(expected_dates)}")
    print(f"Duplicate week rows:     {duplicate_count}")
    print(f"Missing weekly dates:    {len(missing_dates)}")
    print(f"Continuous weekly:       {is_continuous}")

    if len(diffs) > 0:
        print(f"Minimum gap in days:     {int(diffs.min())}")
        print(f"Maximum gap in days:     {int(diffs.max())}")

    if len(missing_dates) > 0:
        print("\nFirst missing weeks:")
        for d in missing_dates[:20]:
            print(f"  - {d.date()}")

    if not all_gaps_are_7_days:
        bad_gaps = pd.DataFrame({
            "previous_week": unique_dates.shift(1),
            "current_week": unique_dates,
            "gap_days": unique_dates.diff().dt.days,
        }).dropna()

        bad_gaps = bad_gaps[bad_gaps["gap_days"] != 7]

        print("\nNon-weekly gaps:")
        print(bad_gaps.head(20).to_string(index=False))

    return {
        "sample_size_rows": int(len(df)),
        "sample_size_unique_weeks": int(len(unique_dates)),
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "expected_weeks": int(len(expected_dates)),
        "missing_weeks": int(len(missing_dates)),
        "duplicate_week_rows": int(duplicate_count),
        "is_continuous_weekly": bool(is_continuous),
        "min_gap_days": int(diffs.min()) if len(diffs) > 0 else None,
        "max_gap_days": int(diffs.max()) if len(diffs) > 0 else None,
    }


def print_missing_target_diagnostics(df, product, target_col, price_col=None):
    """
    Prints rows with missing target sales values.
    """
    missing = df[df[target_col].isna()].copy()

    print("\n" + "=" * 80)
    print(f"Target-value check for {product}")
    print("=" * 80)
    print(f"Target sales column:          {target_col}")
    print(f"Rows with missing target:     {len(missing)}")
    print(f"Rows with non-missing target: {df[target_col].notna().sum()}")

    if missing.empty:
        return

    diagnostic_cols = ["week_ending_date"]

    if "created_date" in df.columns:
        diagnostic_cols.append("created_date")

    if "published_date" in df.columns:
        diagnostic_cols.append("published_date")

    if price_col is not None and price_col in df.columns:
        diagnostic_cols.append(price_col)

    diagnostic_cols.append(target_col)

    print("\nRows with missing target sales:")
    print(missing[diagnostic_cols].to_string(index=False))


def longest_consecutive_complete_block(df, target_col):
    """
    Keeps the longest consecutive weekly block where target_col is non-missing.

    A row is valid for the model-ready time series only if:
      - week_ending_date is valid
      - target_col is non-missing
      - weekly gaps inside the selected block are exactly 7 days

    If several blocks have the same length, the most recent block is selected.
    """
    valid = df.copy()
    valid = valid.dropna(subset=["week_ending_date"])
    valid = valid[valid[target_col].notna()].copy()
    valid = valid.sort_values("week_ending_date").reset_index(drop=True)

    if valid.empty:
        raise ValueError(
            f"No usable rows remain after dropping missing target column '{target_col}'."
        )

    gaps = valid["week_ending_date"].diff().dt.days

    # Start a new block whenever the gap is not exactly 7 days.
    valid["_new_block"] = gaps.ne(7)
    valid.loc[0, "_new_block"] = True
    valid["_block_id"] = valid["_new_block"].cumsum()

    block_summary = (
        valid.groupby("_block_id")
        .agg(
            rows=("week_ending_date", "size"),
            start_date=("week_ending_date", "min"),
            end_date=("week_ending_date", "max"),
        )
        .reset_index()
    )

    # Choose largest block; if tie, choose the most recent block.
    best_block = (
        block_summary
        .sort_values(["rows", "end_date"], ascending=[False, False])
        .iloc[0]
    )

    best_block_id = best_block["_block_id"]

    model_df = (
        valid[valid["_block_id"] == best_block_id]
        .drop(columns=["_new_block", "_block_id"], errors="ignore")
        .reset_index(drop=True)
    )

    block_summary = block_summary.sort_values(
        ["rows", "end_date"],
        ascending=[False, False],
    )

    return model_df, block_summary


def prepare_product_dataframe(product_df, product):
    """
    Cleans one product dataframe, checks target completeness, and returns:
      - raw_clean_df: date-cleaned, numeric target parsed, one row per week
      - model_df: longest consecutive weekly block without missing target values
      - metadata dict for summary
    """
    product_df = clean_column_names(product_df)
    product_df = find_and_standardize_week_date_column(product_df, product)

    product_df["week_ending_date"] = pd.to_datetime(
        product_df["week_ending_date"],
        errors="coerce",
    )

    # Remove rows without valid weekly date.
    product_df = product_df.dropna(subset=["week_ending_date"])

    # Sort by week ending date.
    product_df = product_df.sort_values("week_ending_date").reset_index(drop=True)

    # Remove exact duplicate rows first.
    before_exact = len(product_df)
    product_df = product_df.drop_duplicates().reset_index(drop=True)
    exact_duplicates_removed = before_exact - len(product_df)

    target_col = find_sales_column(product_df, product)
    price_col = find_price_column(product_df, product)

    # Parse target sales column as numeric.
    product_df[target_col] = parse_numeric_series(product_df[target_col])

    # Parse price column as numeric for clean diagnostics if available.
    if price_col is not None and price_col in product_df.columns:
        product_df[price_col] = parse_numeric_series(product_df[price_col])

    # If duplicate weeks exist, select the best row per week.
    before_week_dedup = len(product_df)
    product_df = choose_one_row_per_week(product_df, target_col)
    product_df = product_df.sort_values("week_ending_date").reset_index(drop=True)
    weekly_duplicates_removed = before_week_dedup - len(product_df)

    raw_clean_df = product_df.copy()

    missing_target_count = int(raw_clean_df[target_col].isna().sum())
    usable_target_count = int(raw_clean_df[target_col].notna().sum())

    print_missing_target_diagnostics(
        raw_clean_df,
        product=product,
        target_col=target_col,
        price_col=price_col,
    )

    model_df, block_summary = longest_consecutive_complete_block(
        raw_clean_df,
        target_col=target_col,
    )

    print("\n" + "=" * 80)
    print(f"Longest consecutive complete target block for {product}")
    print("=" * 80)
    print(block_summary.head(10).to_string(index=False))

    dropped_for_model_block = len(raw_clean_df) - len(model_df)

    metadata = {
        "target_sales_column": target_col,
        "price_column": price_col,
        "exact_duplicate_rows_removed": int(exact_duplicates_removed),
        "weekly_duplicate_rows_removed": int(weekly_duplicates_removed),
        "raw_rows_after_date_cleaning": int(len(raw_clean_df)),
        "raw_unique_week_count": int(raw_clean_df["week_ending_date"].nunique()),
        "raw_missing_target_values": int(missing_target_count),
        "raw_usable_target_rows": int(usable_target_count),
        "model_rows_longest_complete_block": int(len(model_df)),
        "model_start_date": model_df["week_ending_date"].min().date(),
        "model_end_date": model_df["week_ending_date"].max().date(),
        "dropped_rows_to_get_longest_complete_block": int(dropped_for_model_block),
    }

    return raw_clean_df, model_df, metadata


summary_rows = []

for product, section in SECTIONS.items():
    product_frames = []
    current = START

    print("\n" + "#" * 80)
    print(f"Downloading {product}")
    print("#" * 80)

    while current <= END:
        window_end = min(current + timedelta(days=179), END)

        section_encoded = quote(section, safe="")
        url = (
            f"{BASE}/{section_encoded}"
            f"?q=week_ending_date={fmt(current)}:{fmt(window_end)}"
            f"&sort=week_ending_date"
        )

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed for {product}: {fmt(current)} to {fmt(window_end)}")
            print(e)
            current = window_end + timedelta(days=1)
            continue

        payload = r.json()
        rows = extract_rows(payload)

        if rows:
            df = pd.DataFrame(rows)
            df["product_series"] = product
            df["source_section"] = section
            product_frames.append(df)

        print(f"{product}: {fmt(current)} to {fmt(window_end)} -> {len(rows)} rows")

        current = window_end + timedelta(days=1)

    if product_frames:
        product_df = pd.concat(product_frames, ignore_index=True)

        try:
            raw_clean_df, model_df, metadata = prepare_product_dataframe(
                product_df,
                product,
            )

            raw_output_path = RAW_OUTPUT_DIR / f"{product}_raw_cleaned.csv"
            model_output_path = OUTPUT_DIR / f"{product}.csv"

            # Save raw cleaned data before trimming to the longest complete block.
            raw_clean_df.to_csv(raw_output_path, index=False)

            # Save model-ready data: longest consecutive weekly block without
            # missing sales target values.
            model_df.to_csv(model_output_path, index=False)

            print(f"\nSaved raw cleaned {product}:   {raw_output_path}")
            print(f"Saved model-ready {product}:   {model_output_path}")

            model_continuity = check_weekly_continuity(model_df, product)

            summary = {
                "product": product,
                **metadata,
                "sample_size_rows": model_continuity["sample_size_rows"],
                "sample_size_unique_weeks": model_continuity["sample_size_unique_weeks"],
                "start_date": model_continuity["start_date"],
                "end_date": model_continuity["end_date"],
                "expected_weeks": model_continuity["expected_weeks"],
                "missing_weeks": model_continuity["missing_weeks"],
                "duplicate_week_rows": model_continuity["duplicate_week_rows"],
                "is_continuous_weekly": model_continuity["is_continuous_weekly"],
                "min_gap_days": model_continuity["min_gap_days"],
                "max_gap_days": model_continuity["max_gap_days"],
            }

            summary_rows.append(summary)

        except Exception as e:
            print(f"\nFailed to prepare {product}: {e}")
            summary_rows.append({
                "product": product,
                "target_sales_column": None,
                "price_column": None,
                "exact_duplicate_rows_removed": None,
                "weekly_duplicate_rows_removed": None,
                "raw_rows_after_date_cleaning": None,
                "raw_unique_week_count": None,
                "raw_missing_target_values": None,
                "raw_usable_target_rows": None,
                "model_rows_longest_complete_block": 0,
                "model_start_date": None,
                "model_end_date": None,
                "dropped_rows_to_get_longest_complete_block": None,
                "sample_size_rows": 0,
                "sample_size_unique_weeks": 0,
                "start_date": None,
                "end_date": None,
                "expected_weeks": 0,
                "missing_weeks": None,
                "duplicate_week_rows": None,
                "is_continuous_weekly": False,
                "min_gap_days": None,
                "max_gap_days": None,
                "error": str(e),
            })

    else:
        print(f"No data returned for {product}")
        summary_rows.append({
            "product": product,
            "target_sales_column": None,
            "price_column": None,
            "exact_duplicate_rows_removed": None,
            "weekly_duplicate_rows_removed": None,
            "raw_rows_after_date_cleaning": 0,
            "raw_unique_week_count": 0,
            "raw_missing_target_values": None,
            "raw_usable_target_rows": 0,
            "model_rows_longest_complete_block": 0,
            "model_start_date": None,
            "model_end_date": None,
            "dropped_rows_to_get_longest_complete_block": None,
            "sample_size_rows": 0,
            "sample_size_unique_weeks": 0,
            "start_date": None,
            "end_date": None,
            "expected_weeks": 0,
            "missing_weeks": None,
            "duplicate_week_rows": None,
            "is_continuous_weekly": False,
            "min_gap_days": None,
            "max_gap_days": None,
        })


summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "dataset_sample_size_and_continuity_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "#" * 80)
print("FINAL DATASET SAMPLE SIZE, TARGET COMPLETENESS, AND CONTINUITY SUMMARY")
print("#" * 80)
print(summary_df.to_string(index=False))
print(f"\nSaved summary: {summary_path}")
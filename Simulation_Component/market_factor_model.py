"""Factor-model demand generator with lead-lag propagation.

This module provides a synthetic demand generator for supply-chain
simulation studies of *collaborative* forecasting, where several firms
at the same tier each forecast their own demand and may benefit from
sharing information. It generalizes a simple single-signal market model
into a configurable multi-stream model in which the amount and type of
structure shared across firms can be controlled explicitly.

Motivation
----------
Whether collaboration helps a forecaster depends on whether other
firms' demand series carry information about a firm's own future demand.
Two questions govern this:

  (1) How much structure do the series share?      -> controlled by `lam`
  (2) Is the shared structure also recoverable from a firm's own
      history, or does it live only in the peers' series?  -> by `tau`

This generator exposes both as parameters, so an experiment can sweep
them and trace how the value of collaboration varies with the data-
generating process, rather than reporting a single point result.

Model
-----
Let there be R retailers and P products, giving S = R * P demand
streams (one per retailer-product pair).

    F(t)                A shared latent market factor common to all
                        products: a trend + seasonality + shared-noise
                        curve. It represents the market-wide component
                        of demand.

    F_p(t) = F(t - p*tau)
                        The factor as product p experiences it, delayed
                        by p * tau periods. This models lead-lag
                        propagation: a market movement reaches product 0
                        first and later products with a delay. When
                        tau > 0, an early product's *current* value is
                        informative about a later product's *future*
                        value -- information that is present in peers'
                        series but not in a firm's own history. When
                        tau = 0 all products see the factor
                        simultaneously, so sharing can only average out
                        noise (no exclusive cross-series information).

    I_s(t)              A stream-specific (idiosyncratic) component:
                        each stream's own seasonal phase and frequency,
                        its own noise, and optional demand shocks. This
                        is the part unique to a stream and unavailable
                        to peers.

    Each stream's demand is a variance-normalized mixture of the two:

        demand_s(t) = base_s + scale_s * ( lam * F_p(t)
                                           + sqrt(1 - lam^2) * I_s(t) )

Parameters of interest
----------------------
    lam in [0, 1]   Factor loading -- the fraction of each stream's
                    fluctuation variance coming from the shared factor.
                        lam = 1 -> streams are shared-factor only (all
                                   streams are scaled copies of one
                                   signal; with tau = 0 this is the
                                   classic single-signal market).
                        lam = 0 -> streams are idiosyncratic only
                                   (firms share nothing).
                    The sqrt(1 - lam^2) weighting keeps each stream's
                    total fluctuation variance approximately constant as
                    lam varies, so changing the amount of sharing does
                    not simultaneously change the overall signal
                    magnitude (the two effects would otherwise be
                    confounded).

    tau >= 0        Lead-lag delay between consecutive products. Set
                    tau >= the shortest forecasting lead time so that
                    the cross-series information is still usable at the
                    moment a forecast is made.

Design choices worth noting
---------------------------
  * Streams are generated *individually* from the factor and their own
    components; they are not obtained by proportionally splitting one
    total. The base levels are chosen so that the *expected* total
    matches a target demand level, but the realized total fluctuates.
    An exact pathwise sum constraint is deliberately avoided: forcing
    the streams to sum to a fixed total each period would make their
    idiosyncratic parts cancel and thus perfectly (negatively)
    correlated, silently re-coupling series that are meant to be
    independent.

  * With lam = 1, tau = 0, and P = 1, the generator reduces to a single
    shared signal split across retailers -- i.e. the simple single-
    signal market model. `verify_backward_compat()` checks this
    reduction against a reference implementation and can be used as a
    correctness test before running experiments.

Interface
---------
Exposes the same methods a simulation expects from a market object
(`get_demand`, `split_demand_on_time`, `split_demand`, `act`,
`split_demand_history`). `split_demand_on_time(t)` returns one demand
value per stream, ordered as s = retailer_index * P + product_index.
Mapping streams onto specific supply-chain agents (e.g. assigning each
product to a manufacturer) is handled by the simulation's connectivity
configuration, not by this class.
"""

import math

import numpy as np

# Import the Market abstract base class when this file is part of the
# simulation package; fall back to `object` so the module can also be
# imported on its own (e.g. for unit testing the generator in isolation).
try:
    from .market import Market  # package-relative
except ImportError:  # pragma: no cover
    try:
        from market import Market
    except ImportError:
        Market = object


class MarketFactorModel(Market):
    """Shared-factor + idiosyncratic demand for R retailers x P products.

    Parameters
    ----------
    sim_time, primary_demand, trend_mag, seasonality_mag,
    seasonality_freq, random_walk_mean, random_walk_var :
        Parameters of the shared market factor F(t): total number of
        time steps, base demand level, linear trend, seasonal amplitude
        and frequency, and the mean/variance of the shared noise. These
        also set the overall demand scale.
    retailer_num : int
        Number of retailers R.
    product_num : int
        Number of products P. Total streams S = R * P.
    market_demand_split : list[float], shape (R*P,)
        Base-level share of each stream; must sum to 1. Base level of a
        stream is share * primary_demand, so the streams' base levels
        sum to the target demand level.
    lam : float in [0, 1]
        Factor loading (shared-variance fraction). lam = 1 makes streams
        pure shared-factor copies; lam = 0 makes them purely
        idiosyncratic.
    tau : int >= 0
        Lead-lag delay: product p sees the factor delayed by p * tau
        steps. tau = 0 gives a contemporaneous shared factor (sharing
        can only reduce noise). Choose tau >= the shortest forecasting
        lead time so the cross-series signal is usable at forecast time.
    idio_freqs, idio_phases : optional list[float], shape (R*P,)
        Seasonal frequency and phase of each stream's idiosyncratic
        component. Defaults spread frequencies over 0.5x-1.5x of the
        base frequency and phases evenly over [0, 2*pi) so that no two
        streams have identical idiosyncratic seasonality.
    idio_noise_scale : float
        Standard deviation of the idiosyncratic white-noise term before
        standardization; sets the seasonal-vs-noise balance within I_s.
    shock_prob, shock_mag : float
        Optional demand-shock process: in each period, with probability
        shock_prob, add a shock drawn from N(0, shock_mag^2) to that
        stream's idiosyncratic component. Set shock_prob = 0 to disable.
    seed : optional int
        Seed for this generator's idiosyncratic RNG. The shared factor
        F(t) is drawn from NumPy's global RNG (matching the reference
        single-signal model), which is what allows the lam = 1
        reduction to be checked against that reference under a common
        global seed.
    """

    def __init__(self, sim_time: int, primary_demand: float, trend_mag: float,
                 seasonality_mag: float, seasonality_freq: int,
                 random_walk_mean: float, random_walk_var: float,
                 retailer_num: int, market_demand_split,
                 product_num: int = 1,
                 lam: float = 1.0,
                 tau: int = 0,
                 idio_freqs=None,
                 idio_phases=None,
                 idio_noise_scale: float = 1.0,
                 shock_prob: float = 0.0,
                 shock_mag: float = 0.0,
                 seed=None) -> None:

        assert 0.0 <= lam <= 1.0, "lam must be in [0, 1]"
        assert tau >= 0 and int(tau) == tau, "tau must be a non-negative int"

        self.sim_time = sim_time
        self.primary_demand = primary_demand
        self.trend_mag = trend_mag
        self.seasonality_mag = seasonality_mag
        self.seasonality_freq = seasonality_freq
        self.random_walk_mean = random_walk_mean
        self.random_walk_var = random_walk_var

        self.retailer_num = retailer_num
        self.product_num = product_num
        self.num_streams = retailer_num * product_num

        self.market_demand_split = np.asarray(market_demand_split, dtype=float)
        assert self.market_demand_split.shape == (self.num_streams,), (
            f"market_demand_split must have one entry per stream "
            f"(R*P = {self.num_streams}), got {self.market_demand_split.shape}")
        assert np.isclose(self.market_demand_split.sum(), 1.0), (
            "market_demand_split must sum to 1 (base levels partition the total)")

        self.lam = float(lam)
        self.tau = int(tau)
        self.idio_noise_scale = float(idio_noise_scale)
        self.shock_prob = float(shock_prob)
        self.shock_mag = float(shock_mag)

        # Idiosyncratic seasonal structure. Defaults give every stream a
        # distinct seasonal frequency and phase so that idiosyncratic
        # components do not coincide across streams.
        if idio_freqs is None:
            mults = np.linspace(0.5, 1.5, self.num_streams)
            idio_freqs = [seasonality_freq * m for m in mults]
        if idio_phases is None:
            idio_phases = [2 * math.pi * s / self.num_streams
                           for s in range(self.num_streams)]
        self.idio_freqs = np.asarray(idio_freqs, dtype=float)
        self.idio_phases = np.asarray(idio_phases, dtype=float)

        # Separate RNG for idiosyncratic components (see note on `seed`).
        self._idio_rng = np.random.default_rng(seed)

        # Populated by demand_funtion():
        self.demand = None            # total demand per time step (sum of streams)
        self.stream_demand = None     # per-stream demand, shape (sim_time, S)
        self.factor = None            # standardized shared factor F(t)
        self.factor_raw = None        # shared-factor curve before standardization

        self.demand_funtion()

        self.split_demand_history = []

    # ------------------------------------------------------------------
    # Demand generation
    # ------------------------------------------------------------------

    def _old_demand_curve(self) -> np.ndarray:
        """Shared market-factor curve: trend + seasonality + shared noise.

        This is the single-signal demand curve that the multi-stream
        model builds on. It draws from the global NumPy RNG so that,
        under a fixed global seed, it matches the reference single-signal
        model exactly (used by `verify_backward_compat`).
        """
        time = np.arange(self.sim_time)
        return np.round(
            self.primary_demand
            + self.trend_mag * time
            + self.seasonality_mag * np.sin(((2 * math.pi) / self.seasonality_freq) * time)
            + self.random_walk_mean * np.random.normal(loc=0, scale=1, size=self.sim_time),
            0)

    @staticmethod
    def _standardize(x: np.ndarray) -> np.ndarray:
        """Center to zero mean and scale to unit variance (no-op if constant)."""
        sd = x.std()
        if sd == 0:
            return x - x.mean()
        return (x - x.mean()) / sd

    def _build_idiosyncratic(self) -> np.ndarray:
        """One standardized idiosyncratic series per stream.

        Each series combines a stream-specific seasonal wave (its own
        frequency and phase), white noise, and optional demand shocks,
        then is standardized to unit variance so that `lam` alone
        controls the shared/idiosyncratic balance.
        """
        time = np.arange(self.sim_time)
        idio = np.empty((self.sim_time, self.num_streams))
        for s in range(self.num_streams):
            seasonal = np.sin((2 * math.pi / self.idio_freqs[s]) * time
                              + self.idio_phases[s])
            noise = self._idio_rng.normal(0.0, self.idio_noise_scale,
                                          size=self.sim_time)
            comp = seasonal + noise
            if self.shock_prob > 0.0 and self.shock_mag > 0.0:
                mask = self._idio_rng.random(self.sim_time) < self.shock_prob
                shocks = self._idio_rng.normal(0.0, self.shock_mag,
                                               size=self.sim_time)
                comp = comp + mask * shocks
            idio[:, s] = self._standardize(comp)
        return idio

    def demand_funtion(self):
        """Construct the shared factor, per-stream demand, and total demand.

        The shared factor is split into a deterministic level+trend path
        and a standardized fluctuation (seasonality + noise). Each stream
        mixes the (possibly delayed) factor fluctuation with its own
        standardized idiosyncratic fluctuation using weights lam and
        sqrt(1 - lam^2), then rescales by the stream's share of the
        fluctuation magnitude. This keeps each stream's fluctuation
        variance roughly constant across lam, so that lam changes the
        *composition* of the signal without changing its magnitude.
        """
        # 1) Shared factor, separated into a deterministic level path and
        #    a standardized fluctuation.
        self.factor_raw = self._old_demand_curve()
        time = np.arange(self.sim_time)
        level_path = self.primary_demand + self.trend_mag * time  # deterministic level + trend
        fluct = self.factor_raw - level_path                      # seasonality + noise
        sigma_f = fluct.std()
        self.factor = self._standardize(fluct)                    # unit-variance F(t)

        # 2) Idiosyncratic components (unit variance each).
        idio = self._build_idiosyncratic()

        # 3) Per-product delayed views of the factor: F_p(t) = F(t - p*tau).
        #    The first p*tau steps are padded with the initial value; this
        #    only affects the simulation warm-up window.
        factor_views = np.empty((self.sim_time, self.product_num))
        for p in range(self.product_num):
            d = p * self.tau
            if d == 0:
                factor_views[:, p] = self.factor
            else:
                factor_views[d:, p] = self.factor[:-d]
                factor_views[:d, p] = self.factor[0]

        # 4) Mix shared and idiosyncratic parts per stream.
        #    Stream s corresponds to (retailer r, product p): s = r*P + p.
        lam = self.lam
        w_idio = math.sqrt(max(0.0, 1.0 - lam ** 2))
        self.stream_demand = np.empty((self.sim_time, self.num_streams))
        for r in range(self.retailer_num):
            for p in range(self.product_num):
                s = r * self.product_num + p
                base_s = self.market_demand_split[s] * level_path
                scale_s = self.market_demand_split[s] * sigma_f
                mix = lam * factor_views[:, p] + w_idio * idio[:, s]
                self.stream_demand[:, s] = np.round(base_s + scale_s * mix, 0)

        # Total demand fluctuates around the target level; the base levels
        # sum to the target, but the realized per-period total is free to
        # vary (no exact pathwise sum constraint, by design).
        self.demand = self.stream_demand.sum(axis=1)

    # ------------------------------------------------------------------
    # Simulation-facing interface (same methods as the base market model)
    # ------------------------------------------------------------------

    def get_demand(self, t):
        return self.demand[t]

    def split_demand_on_time(self, t):
        """Return the per-stream demand at time t.

        Streams are generated individually rather than split from a
        single total, so this simply returns the precomputed per-stream
        values (ordered s = retailer * P + product).
        """
        demand_list = self.stream_demand[t, :].tolist()
        self.split_demand_history.append(demand_list)
        return demand_list

    def split_demand(self, demand_t):
        """Proportionally split a supplied total into streams.

        Retained for interface compatibility with the data-driven market
        path, which passes an externally supplied total demand.
        """
        demand_list = demand_t * self.market_demand_split
        d = 0
        for i in range(len(demand_list) - 1):
            demand_list[i] = np.round(demand_list[i])
            d = np.round(demand_list[i]) + d
        demand_list[-1] = demand_t - d
        self.split_demand_history.append(demand_list)
        return [demand_list]

    def act(self, demand_t):
        demand = self.split_demand(demand_t=demand_t)
        return [d for d in demand]

    def _calcrho(self, rho, rho1, rho2):
        part_1 = 1 - (rho1 * rho2)
        part_2 = np.sqrt((1 - math.pow(rho1, 2)) * (1 - math.pow(rho2, 2)))
        return rho * (part_1 / part_2)

    def _init_splitting_meachanism(self):
        # Streams are generated individually, so there is no proportional
        # splitting mechanism to initialize.
        pass


# ----------------------------------------------------------------------
# Reduction check: with lam=1, tau=0, P=1 the model must reproduce the
# reference single-signal market. Run before experiments as a sanity test.
# ----------------------------------------------------------------------

def verify_backward_compat(seed: int = 42, sim_time: int = 500,
                           primary_demand: float = 500, trend_mag: float = 0.2,
                           seasonality_mag: float = 40, seasonality_freq: int = 52,
                           random_walk_mean: float = 20, random_walk_var: float = 1,
                           retailer_num: int = 2,
                           split=(0.75, 0.25)) -> bool:
    """Check that the factor model reduces to the single-signal market.

    Under a fixed global seed it verifies two things:

      1. The shared-factor curve equals the reference single-signal
         demand series exactly (identical RNG consumption).
      2. With lam = 1, tau = 0, P = 1, the summed per-stream demand
         tracks the reference series: the fluctuations are perfectly
         correlated and any difference is bounded by integer rounding
         (exact equality is not expected because the factor curve is
         rounded once and each stream is rounded again after scaling).

    Returns True if both checks pass. Intended as a pre-experiment
    correctness test.
    """
    from market import MarketArtifical  # reference single-signal model

    np.random.seed(seed)
    old = MarketArtifical(sim_time, primary_demand, trend_mag,
                          seasonality_mag, seasonality_freq,
                          random_walk_mean, random_walk_var,
                          retailer_num, list(split))

    np.random.seed(seed)
    # The reference model's demand_funtion draws three normal arrays (two
    # are from earlier formulas that are overwritten, one is the active
    # curve). Only the last draw defines its output. This generator draws
    # a single array, so we replay the two unused draws first to align the
    # RNG state before constructing the factor model.
    # NOTE: if those unused lines are ever removed from the reference
    # model, remove these two replay draws too, or this check will fail.
    _ = np.random.normal(random_walk_mean, random_walk_var, size=sim_time)
    _ = np.random.normal(loc=0, scale=1, size=sim_time)
    new = MarketFactorModel(sim_time, primary_demand, trend_mag,
                            seasonality_mag, seasonality_freq,
                            random_walk_mean, random_walk_var,
                            retailer_num, list(split),
                            product_num=1, lam=1.0, tau=0, seed=0)

    ok_factor = np.array_equal(old.demand, new.factor_raw)

    # Fluctuations should be perfectly correlated; level error is bounded
    # by cumulative rounding (a few units), not zero.
    corr = np.corrcoef(new.demand, old.demand)[0, 1]
    max_diff = np.max(np.abs(new.demand - old.demand))
    ok_streams = corr > 0.9999 and max_diff <= 2.0 * retailer_num + 2.0

    print(f"factor path identical:        {ok_factor}")
    print(f"stream sum tracks reference:  {ok_streams} "
          f"(corr = {corr:.6f}, max |diff| = {max_diff:.2f})")
    return ok_factor and ok_streams


if __name__ == "__main__":
    verify_backward_compat()

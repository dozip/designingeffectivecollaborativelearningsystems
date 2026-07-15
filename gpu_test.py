#!/usr/bin/env python3

import os
import time
import signal
import multiprocessing as mp


NUM_GPUS = 8

# Rechenlast pro Prozess. Bei Speichermangel kleiner setzen.
BATCH_SIZE = 2048
INPUT_SIZE = 1024
HIDDEN_SIZE = 4096
OUTPUT_SIZE = 1024


def gpu_worker(physical_gpu_id: int) -> None:
    """
    Ein Prozess pro physischer GPU.

    Durch CUDA_VISIBLE_DEVICES sieht dieser Prozess nur eine GPU.
    Diese erscheint innerhalb des Prozesses daher immer als cuda:0.
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    # Erst nach CUDA_VISIBLE_DEVICES importieren.
    import torch
    import torch.nn as nn

    device = torch.device("cuda:0")

    # CUDA explizit initialisieren und die tatsächliche Fehlermeldung ausgeben.
    try:
        torch.cuda.init()

        visible_device_count = torch.cuda.device_count()

        if visible_device_count != 1:
            raise RuntimeError(
                f"Erwartet wurde genau eine sichtbare GPU, "
                f"gefunden wurden jedoch {visible_device_count}."
            )

        torch.cuda.set_device(device)

        # Kleine Test-Allokation, um sicherzustellen, dass die GPU tatsächlich
        # für CUDA-Berechnungen verwendet werden kann.
        test_tensor = torch.zeros(
            1,
            device=device,
            dtype=torch.float32,
        )
        torch.cuda.synchronize(device)
        del test_tensor

        gpu_name = torch.cuda.get_device_name(device)
        gpu_properties = torch.cuda.get_device_properties(device)

        print(
            f"PID {os.getpid()}: CUDA erfolgreich initialisiert. "
            f"Physische GPU {physical_gpu_id} wird als {device} verwendet. "
            f"GPU: {gpu_name}, "
            f"Compute Capability: "
            f"{gpu_properties.major}.{gpu_properties.minor}",
            flush=True,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Prozess {os.getpid()}: CUDA-Initialisierung für "
            f"physische GPU {physical_gpu_id} fehlgeschlagen.\n"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n"
            f"PyTorch-Version={torch.__version__}\n"
            f"Fehler: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        ).to(
            device=device,
            dtype=torch.float16,
        )

        model.eval()

        x = torch.randn(
            BATCH_SIZE,
            INPUT_SIZE,
            device=device,
            dtype=torch.float16,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Prozess {os.getpid()}: Modell oder Eingabedaten konnten auf "
            f"GPU {physical_gpu_id} nicht angelegt werden.\n"
            f"Fehler: {type(exc).__name__}: {exc}"
        ) from exc

    print(
        f"PID {os.getpid()} nutzt physische GPU {physical_gpu_id} "
        f"als {device}.",
        flush=True,
    )

    iterations = 0
    last_report = time.time()

    try:
        with torch.inference_mode():
            while True:
                x = model(x)

                # Die Ausgabe hat wieder OUTPUT_SIZE=INPUT_SIZE und kann
                # daher direkt erneut durch das Netz geschickt werden.
                iterations += 1

                # Gelegentlich synchronisieren und Status ausgeben.
                if iterations % 100 == 0:
                    torch.cuda.synchronize(device)

                    now = time.time()

                    if now - last_report >= 5:
                        allocated_gib = (
                            torch.cuda.memory_allocated(device) / 1024**3
                        )
                        reserved_gib = (
                            torch.cuda.memory_reserved(device) / 1024**3
                        )

                        print(
                            f"GPU {physical_gpu_id} | "
                            f"PID {os.getpid()} | "
                            f"Iterationen: {iterations} | "
                            f"Speicher belegt: {allocated_gib:.2f} GiB | "
                            f"reserviert: {reserved_gib:.2f} GiB",
                            flush=True,
                        )

                        last_report = now

    except Exception as exc:
        raise RuntimeError(
            f"Prozess {os.getpid()}: Berechnung auf physischer "
            f"GPU {physical_gpu_id} fehlgeschlagen.\n"
            f"Fehler: {type(exc).__name__}: {exc}"
        ) from exc


def main() -> None:
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []

    for gpu_id in range(NUM_GPUS):
        process = ctx.Process(
            target=gpu_worker,
            args=(gpu_id,),
            name=f"gpu-worker-{gpu_id}",
        )
        process.start()
        processes.append(process)

    print(f"{len(processes)} GPU-Prozesse gestartet.", flush=True)

    try:
        for process in processes:
            process.join()

        failed_processes = [
            process
            for process in processes
            if process.exitcode not in (None, 0)
        ]

        if failed_processes:
            print("\nFehlgeschlagene GPU-Prozesse:", flush=True)

            for process in failed_processes:
                print(
                    f"- {process.name}: Exit-Code {process.exitcode}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nBeende alle GPU-Prozesse ...", flush=True)

        for process in processes:
            if process.is_alive():
                process.terminate()

        for process in processes:
            process.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
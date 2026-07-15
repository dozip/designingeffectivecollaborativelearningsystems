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

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Prozess {os.getpid()}: CUDA ist für GPU {physical_gpu_id} nicht verfügbar."
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.GELU(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    ).to(device=device, dtype=torch.float16)

    model.eval()

    x = torch.randn(
        BATCH_SIZE,
        INPUT_SIZE,
        device=device,
        dtype=torch.float16,
    )

    print(
        f"PID {os.getpid()} nutzt physische GPU {physical_gpu_id} "
        f"als {device}.",
        flush=True,
    )

    iterations = 0
    last_report = time.time()

    with torch.inference_mode():
        while True:
            x = model(x)

            # Die Ausgabe hat wieder OUTPUT_SIZE=INPUT_SIZE und kann daher direkt
            # erneut durch das Netz geschickt werden.
            iterations += 1

            # Gelegentlich synchronisieren und Status ausgeben.
            if iterations % 100 == 0:
                torch.cuda.synchronize(device)

                now = time.time()
                if now - last_report >= 5:
                    print(
                        f"GPU {physical_gpu_id} | PID {os.getpid()} | "
                        f"Iterationen: {iterations}",
                        flush=True,
                    )
                    last_report = now


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

    except KeyboardInterrupt:
        print("\nBeende alle GPU-Prozesse ...", flush=True)

        for process in processes:
            if process.is_alive():
                process.terminate()

        for process in processes:
            process.join(timeout=5)


if __name__ == "__main__":
    # Verhindert, dass Kinder Ctrl+C gleichzeitig behandeln.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        main()
    except KeyboardInterrupt:
        pass
from numba import cuda

if cuda.is_available():
    device = cuda.select_device(0)

    name = device.name
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    capability = device.compute_capability
    sm_count = device.MULTIPROCESSOR_COUNT
    _, total_memory = cuda.current_context().get_memory_info()

    cores_per_sm = {
        (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,
        (5, 0): 128, (5, 2): 128, (5, 3): 128,
        (6, 0): 64, (6, 1): 128, (6, 2): 128,
        (7, 0): 64, (7, 2): 64, (7, 5): 64,
        (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
        (9, 0): 128,
        (10, 0): 128, (10, 1): 128, (10, 3): 128, (10, 7): 128,
        (11, 0): 128, (12, 0): 128, (12, 1): 128,
    }

    print(f"Device name: {name}")
    print(f"Multiprocessor count: {sm_count}")

    if capability in cores_per_sm:
        print(f"CUDA core count: {sm_count * cores_per_sm[capability]}")
    else:
        print("CUDA core count: unknown")

    print(f"Memory: {total_memory / 1024**3:.2f} GiB")
else:
    raise SystemExit("No CUDA GPU found.")

from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from PIL import Image


def grayscale_cpu(src, dst):
    for i in range(src.shape[0]):
        dst[i] = (int(src[i, 0]) + int(src[i, 1]) + int(src[i, 2])) / 3


@cuda.jit
def grayscale_gpu(src, dst):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < src.shape[0]:
        dst[i] = (int(src[i, 0]) + int(src[i, 1]) + int(src[i, 2])) / 3


def main():
    folder = Path(__file__).resolve().parent
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "input.jpg"
    image = mpimg.imread(image_path)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Use an RGB image.")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image[:, :, :3])
    height, width, _ = image.shape
    src = image.reshape(height * width, 3)

    if not cuda.is_available():
        raise SystemExit("No CUDA GPU found.")

    cpu_result = np.empty(height * width, dtype=np.uint8)
    cpu_times = []
    repeats = 5
    for _ in range(repeats):
        start = time.time()
        grayscale_cpu(src, cpu_result)
        cpu_times.append(time.time() - start)
    cpu_time = np.mean(cpu_times)
    Image.fromarray(cpu_result.reshape(height, width)).save(folder / "gray_cpu.png")

    d_src = cuda.to_device(src)
    d_dst = cuda.device_array(height * width, dtype=np.uint8)
    gpu_result = np.empty_like(cpu_result)
    block_sizes = [32, 64, 128, 256, 512, 1024]
    gpu_times = []
    gpu_std = []

    print(f"Image: {image_path.name} ({width} x {height})")
    print(f"CPU mean ({repeats} runs): {cpu_time * 1000:.3f} ms")
    print("GPU timing includes both transfers and kernel execution.")
    print("Block size    GPU mean (ms)    Std (ms)    Speedup")
    for block_size in block_sizes:
        grid_size = (src.shape[0] + block_size - 1) // block_size

        # Warm up before timing, so compilation is not measured.
        grayscale_gpu[grid_size, block_size](d_src, d_dst)
        cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.time()
            d_src.copy_to_device(src)
            grayscale_gpu[grid_size, block_size](d_src, d_dst)
            d_dst.copy_to_host(gpu_result)
            cuda.synchronize()
            times.append(time.time() - start)
            np.testing.assert_array_equal(cpu_result, gpu_result)

        gpu_time = np.mean(times)
        gpu_times.append(gpu_time * 1000)
        gpu_std.append(np.std(times) * 1000)
        print(f"{block_size:10d}    {gpu_time * 1000:13.3f}"
              f"    {np.std(times) * 1000:8.3f}    {cpu_time / gpu_time:7.2f}x")

    print("CPU and GPU results match exactly for all block sizes.")
    Image.fromarray(gpu_result.reshape(height, width)).save(folder / "gray_gpu.png")
    plt.errorbar(block_sizes, gpu_times, yerr=gpu_std, marker="o", capsize=4)
    plt.xscale("log", base=2)
    plt.xticks(block_sizes, block_sizes)
    plt.xlabel("Block size (threads)")
    plt.ylabel("GPU time including transfers (ms)")
    plt.title("RGB to grayscale")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "block_size_time.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

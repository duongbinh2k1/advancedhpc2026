import time

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image


@cuda.jit
def grayscale(src, dst):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < src.shape[0]:
        dst[i] = (int(src[i, 0]) + int(src[i, 1]) + int(src[i, 2])) / 3


image = mpimg.imread("./input.jpg")
height, width, _ = image.shape
pixel_count = height * width
src = image.reshape(pixel_count, 3)
cpu_result = np.empty(pixel_count, dtype=np.uint8)

start = time.time()
for i in range(pixel_count):
    cpu_result[i] = (int(src[i, 0]) + int(src[i, 1]) + int(src[i, 2])) / 3
cpu_time = time.time() - start

print("CPU time:", round(cpu_time * 1000, 3), "ms")
Image.fromarray(cpu_result.reshape(height, width)).save("./gray_cpu.png")

d_src = cuda.to_device(src)
d_dst = cuda.device_array(pixel_count, dtype=np.uint8)

# Run once to compile the kernel before timing.
grayscale[(pixel_count + 255) // 256, 256](d_src, d_dst)
cuda.synchronize()

block_sizes = [32, 64, 128, 256, 512, 1024]
gpu_times = []

for block_size in block_sizes:
    grid_size = (pixel_count + block_size - 1) // block_size

    start = time.time()
    d_src.copy_to_device(src)
    grayscale[grid_size, block_size](d_src, d_dst)
    gpu_result = d_dst.copy_to_host()
    cuda.synchronize()
    gpu_time = time.time() - start
    gpu_times.append(gpu_time * 1000)

    print("Block size:", block_size)
    print("GPU time:", round(gpu_time * 1000, 3), "ms")
    print("Speedup:", round(cpu_time / gpu_time, 2))

Image.fromarray(gpu_result.reshape(height, width)).save("./gray_gpu.png")

plt.plot(block_sizes, gpu_times, "o-")
plt.xscale("log", base=2)
plt.xticks(block_sizes, block_sizes)
plt.xlabel("Block size")
plt.ylabel("GPU time including transfers (ms)")
plt.tight_layout()
plt.savefig("./block_size_time.png")
plt.close()

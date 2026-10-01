import time

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, int32
from PIL import Image


@cuda.jit
def blur_global(src, dst, weights):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        total = 0
        for row in range(7):
            yy = min(max(y + row - 3, 0), src.shape[0] - 1)
            for col in range(7):
                xx = min(max(x + col - 3, 0), src.shape[1] - 1)
                total += int(src[yy, xx]) * weights[row, col]
        dst[y, x] = total // 1003


@cuda.jit
def blur_shared(src, dst, weights):
    shared_weights = cuda.shared.array((7, 7), dtype=int32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if tx < 7 and ty < 7:
        shared_weights[ty, tx] = weights[ty, tx]
    cuda.syncthreads()

    x = tx + cuda.blockIdx.x * cuda.blockDim.x
    y = ty + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        total = 0
        for row in range(7):
            yy = min(max(y + row - 3, 0), src.shape[0] - 1)
            for col in range(7):
                xx = min(max(x + col - 3, 0), src.shape[1] - 1)
                total += int(src[yy, xx]) * shared_weights[row, col]
        dst[y, x] = total // 1003


image = np.array(Image.open("./input.png"))
height, width = image.shape
weights = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
], dtype=np.int32)

cpu_result = np.empty_like(image)
start = time.time()
for y in range(height):
    for x in range(width):
        total = 0
        for row in range(7):
            yy = min(max(y + row - 3, 0), height - 1)
            for col in range(7):
                xx = min(max(x + col - 3, 0), width - 1)
                total += int(image[yy, xx]) * int(weights[row, col])
        cpu_result[y, x] = total // 1003
cpu_time = time.time() - start
print("CPU time:", round(cpu_time * 1000, 3), "ms")
Image.fromarray(cpu_result).save("./blur_cpu.png")

d_image = cuda.to_device(image)
d_weights = cuda.to_device(weights)
d_result = cuda.device_array_like(image)
block_sizes = [(8, 8), (16, 16), (32, 32)]
global_speedups = []
shared_speedups = []

block_size = (16, 16)
grid_size = ((width + block_size[0] - 1) // block_size[0],
             (height + block_size[1] - 1) // block_size[1])
# Run once before timing to compile the kernel.
blur_global[grid_size, block_size](d_image, d_result, d_weights)
cuda.synchronize()

print("GPU global memory")
for block_size in block_sizes:
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])
    start = time.time()
    d_image.copy_to_device(image)
    blur_global[grid_size, block_size](d_image, d_result, d_weights)
    global_result = d_result.copy_to_host()
    cuda.synchronize()
    gpu_time = time.time() - start
    global_speedups.append(cpu_time / gpu_time)
    print("Block size:", block_size)
    print("GPU time:", round(gpu_time * 1000, 3), "ms")
    print("Speedup:", round(cpu_time / gpu_time, 2))
Image.fromarray(global_result).save("./blur_global.png")

blur_shared[grid_size, block_size](d_image, d_result, d_weights)
cuda.synchronize()

print("GPU shared memory")
for block_size in block_sizes:
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])
    start = time.time()
    d_image.copy_to_device(image)
    blur_shared[grid_size, block_size](d_image, d_result, d_weights)
    shared_result = d_result.copy_to_host()
    cuda.synchronize()
    gpu_time = time.time() - start
    shared_speedups.append(cpu_time / gpu_time)
    print("Block size:", block_size)
    print("GPU time:", round(gpu_time * 1000, 3), "ms")
    print("Speedup:", round(cpu_time / gpu_time, 2))
Image.fromarray(shared_result).save("./blur_shared.png")

labels = ["8x8", "16x16", "32x32"]
plt.plot(labels, global_speedups, "o-", label="Global filter")
plt.plot(labels, shared_speedups, "o-", label="Shared filter")
plt.xlabel("Block size")
plt.ylabel("Speedup over CPU")
plt.legend()
plt.tight_layout()
plt.savefig("./block_size_speedup.png")
plt.close()

import time

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image


@cuda.jit
def grayscale_1d(src, dst):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < src.shape[0]:
        dst[i] = (int(src[i, 0]) + int(src[i, 1]) + int(src[i, 2])) / 3


@cuda.jit
def grayscale_2d(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = (int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) / 3


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
block_size = 256
grid_size = (pixel_count + block_size - 1) // block_size

# Run once before timing to compile the kernel.
grayscale_1d[grid_size, block_size](d_src, d_dst)
cuda.synchronize()

start = time.time()
d_src.copy_to_device(src)
grayscale_1d[grid_size, block_size](d_src, d_dst)
gpu_1d = d_dst.copy_to_host()
cuda.synchronize()
time_1d = time.time() - start
speedup_1d = cpu_time / time_1d
print("1D block size:", block_size)
print("1D time:", round(time_1d * 1000, 3), "ms")
print("1D speedup:", round(speedup_1d, 2))

d_image = cuda.to_device(image)
d_gray = cuda.device_array((height, width), dtype=np.uint8)
block_size = (16, 16)
grid_size = ((width + block_size[0] - 1) // block_size[0],
             (height + block_size[1] - 1) // block_size[1])
grayscale_2d[grid_size, block_size](d_image, d_gray)
cuda.synchronize()

block_sizes = [(8, 8), (16, 16), (32, 32)]
speedups = []
for block_size in block_sizes:
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])

    start = time.time()
    d_image.copy_to_device(image)
    grayscale_2d[grid_size, block_size](d_image, d_gray)
    gpu_2d = d_gray.copy_to_host()
    cuda.synchronize()
    time_2d = time.time() - start
    speedup = cpu_time / time_2d
    speedups.append(speedup)

    print("2D block size:", block_size)
    print("2D time:", round(time_2d * 1000, 3), "ms")
    print("2D speedup:", round(speedup, 2))

Image.fromarray(gpu_2d).save("./gray_gpu.png")
plt.plot(["8x8", "16x16", "32x32"], speedups, "o-", label="2D")
plt.axhline(speedup_1d, linestyle="--", label="1D: 256 threads/block")
plt.xlabel("2D block size")
plt.ylabel("Speedup over CPU")
plt.legend()
plt.tight_layout()
plt.savefig("./block_size_speedup.png")
plt.close()

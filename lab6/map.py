import time

import numpy as np
from numba import cuda
from PIL import Image


@cuda.jit
def binarize(src, dst, threshold):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = 1 if src[y, x] >= threshold else 0


@cuda.jit
def brightness(src, dst, change):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = min(255, max(0, int(src[y, x]) + change))


@cuda.jit
def blend(first, second, dst, weight):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < first.shape[1] and y < first.shape[0]:
        dst[y, x] = weight * first[y, x] + (1 - weight) * second[y, x]


first = np.array(Image.open("./input1.png"))
second = np.array(Image.open("./input2.png"))
height, width = first.shape
threshold = 128
change = 40
weight = 0.5

d_first = cuda.to_device(first)
d_second = cuda.to_device(second)
d_binary = cuda.device_array_like(first)
d_bright = cuda.device_array_like(first)
d_dark = cuda.device_array_like(first)
d_blend = cuda.device_array_like(first)
block_size = (16, 16)
grid_size = ((width + block_size[0] - 1) // block_size[0],
             (height + block_size[1] - 1) // block_size[1])
# Compile the three kernels before timing.
binarize[grid_size, block_size](d_first, d_binary, threshold)
brightness[grid_size, block_size](d_first, d_bright, change)
blend[grid_size, block_size](d_first, d_second, d_blend, weight)
cuda.synchronize()

for block_size in [(8, 8), (16, 16), (32, 32)]:
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])
    start = time.time()
    d_first.copy_to_device(first)
    d_second.copy_to_device(second)
    binarize[grid_size, block_size](d_first, d_binary, threshold)
    brightness[grid_size, block_size](d_first, d_bright, change)
    brightness[grid_size, block_size](d_first, d_dark, -change)
    blend[grid_size, block_size](d_first, d_second, d_blend, weight)
    binary = d_binary.copy_to_host()
    bright = d_bright.copy_to_host()
    dark = d_dark.copy_to_host()
    blended = d_blend.copy_to_host()
    cuda.synchronize()
    elapsed = time.time() - start
    print("Block size:", block_size)
    print("Total time:", round(elapsed * 1000, 3), "ms")

Image.fromarray(binary * 255).save("./binary.png")
Image.fromarray(bright).save("./bright.png")
Image.fromarray(dark).save("./dark.png")
Image.fromarray(blended).save("./blend.png")

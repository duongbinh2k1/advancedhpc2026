import time

import numpy as np
from numba import cuda, int32
from PIL import Image


@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = (int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) // 3


@cuda.jit
def reduce_range(src_min, src_max, dst_min, dst_max):
    low = cuda.shared.array(1024, dtype=int32)
    high = cuda.shared.array(1024, dtype=int32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * cuda.blockDim.x + tx
    x = tx + cuda.blockIdx.x * cuda.blockDim.x
    y = ty + cuda.blockIdx.y * cuda.blockDim.y
    low[tid] = 255
    high[tid] = 0
    if x < src_min.shape[1] and y < src_min.shape[0]:
        low[tid] = src_min[y, x]
        high[tid] = src_max[y, x]
    cuda.syncthreads()
    stride = cuda.blockDim.x * cuda.blockDim.y // 2
    while stride > 0:
        if tid < stride:
            low[tid] = min(low[tid], low[tid + stride])
            high[tid] = max(high[tid], high[tid + stride])
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        dst_min[cuda.blockIdx.y, cuda.blockIdx.x] = low[0]
        dst_max[cuda.blockIdx.y, cuda.blockIdx.x] = high[0]


@cuda.jit
def stretch(src, dst, minimum, maximum):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        if maximum == minimum:
            dst[y, x] = 0
        else:
            dst[y, x] = (int(src[y, x]) - minimum) * 255 // (maximum - minimum)


image = np.array(Image.open("./input.png"))
height, width, _ = image.shape
d_image = cuda.to_device(image)
d_gray = cuda.device_array((height, width), dtype=np.int32)
d_result = cuda.device_array((height, width), dtype=np.uint8)
block_size = (16, 16)
grid_size = ((width + block_size[0] - 1) // block_size[0],
             (height + block_size[1] - 1) // block_size[1])
d_low = cuda.device_array((grid_size[1], grid_size[0]), dtype=np.int32)
d_high = cuda.device_array_like(d_low)
# Run each kernel once before timing to compile it.
grayscale[grid_size, block_size](d_image, d_gray)
reduce_range[grid_size, block_size](d_gray, d_gray, d_low, d_high)
stretch[grid_size, block_size](d_gray, d_result, 0, 255)
cuda.synchronize()

for block_size in [(8, 8), (16, 16), (32, 32)]:
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])
    start = time.time()
    d_image.copy_to_device(image)
    grayscale[grid_size, block_size](d_image, d_gray)
    d_low = d_gray
    d_high = d_gray
    while d_low.size > 1:
        rows, cols = d_low.shape
        reduce_grid = ((cols + block_size[0] - 1) // block_size[0],
                       (rows + block_size[1] - 1) // block_size[1])
        next_low = cuda.device_array((reduce_grid[1], reduce_grid[0]), dtype=np.int32)
        next_high = cuda.device_array_like(next_low)
        reduce_range[reduce_grid, block_size](d_low, d_high, next_low, next_high)
        d_low = next_low
        d_high = next_high
    minimum = int(d_low.copy_to_host()[0, 0])
    maximum = int(d_high.copy_to_host()[0, 0])
    stretch[grid_size, block_size](d_gray, d_result, minimum, maximum)
    result = d_result.copy_to_host()
    cuda.synchronize()
    elapsed = time.time() - start
    print("Block size:", block_size)
    print("Min:", minimum, "Max:", maximum)
    print("Total time:", round(elapsed * 1000, 3), "ms")

Image.fromarray(d_gray.copy_to_host().astype(np.uint8)).save("./gray.png")
Image.fromarray(result).save("./stretched.png")

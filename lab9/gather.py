import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image


@cuda.jit
def row_histograms(src, local_hist):
    row = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if row < src.shape[0]:
        for x in range(src.shape[1]):
            level = src[row, x]
            local_hist[row, level] += 1


@cuda.jit
def sum_histograms(local_hist, histogram):
    level = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if level < 256:
        total = 0
        for row in range(local_hist.shape[0]):
            total += local_hist[row, level]
        histogram[level] = total


@cuda.jit
def equalize(src, dst, lookup):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = lookup[src[y, x]]


image = np.array(Image.open("./input.png"))
height, width = image.shape
pixel_count = height * width
d_image = cuda.to_device(image)
d_local = cuda.to_device(np.zeros((height, 256), dtype=np.int32))
d_histogram = cuda.device_array(256, dtype=np.int32)
block_size = 128
grid_size = (height + block_size - 1) // block_size
row_histograms[grid_size, block_size](d_image, d_local)
sum_histograms[2, block_size](d_local, d_histogram)
histogram = d_histogram.copy_to_host()

lookup = np.empty(256, dtype=np.uint8)
total = 0
for level in range(256):
    total += int(histogram[level])
    lookup[level] = total * 255 // pixel_count

d_lookup = cuda.to_device(lookup)
d_result = cuda.device_array_like(d_image)
block_size = (16, 16)
grid_size = ((width + block_size[0] - 1) // block_size[0],
             (height + block_size[1] - 1) // block_size[1])
equalize[grid_size, block_size](d_image, d_result, d_lookup)
result = d_result.copy_to_host()
print("Pixels:", pixel_count)
print("Histogram total:", histogram.sum())
print("Input range:", image.min(), image.max())
print("Output range:", result.min(), result.max())
Image.fromarray(result).save("./equalized.png")

output_histogram = np.bincount(result.ravel(), minlength=256)
fig, axes = plt.subplots(2, 1, figsize=(7, 5))
axes[0].bar(range(256), histogram, width=1)
axes[0].set_title("Input histogram")
axes[1].bar(range(256), output_histogram, width=1)
axes[1].set_title("Equalized histogram")
for axis in axes:
    axis.set_xlabel("Gray level")
    axis.set_ylabel("Pixels")
plt.tight_layout()
plt.savefig("./histograms.png")
plt.close()

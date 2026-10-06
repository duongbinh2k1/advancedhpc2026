import numpy as np
from numba import cuda
from PIL import Image


@cuda.jit
def RGB2HSV(src, hue, saturation, value):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        r = src[y, x, 0] / 255.0
        g = src[y, x, 1] / 255.0
        b = src[y, x, 2] / 255.0
        largest = max(r, g, b)
        smallest = min(r, g, b)
        delta = largest - smallest
        h = 0.0
        if delta != 0:
            if largest == r:
                h = 60 * (((g - b) / delta) % 6)
            elif largest == g:
                h = 60 * ((b - r) / delta + 2)
            else:
                h = 60 * ((r - g) / delta + 4)
        s = 0.0
        if largest != 0:
            s = delta / largest
        hue[y, x] = h
        saturation[y, x] = s
        value[y, x] = largest


@cuda.jit
def HSV2RGB(hue, saturation, value, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < dst.shape[1] and y < dst.shape[0]:
        d = hue[y, x] / 60.0
        hi = int(d) % 6
        f = d - int(d)
        s = saturation[y, x]
        v = value[y, x]
        l = v * (1 - s)
        m = v * (1 - f * s)
        n = v * (1 - (1 - f) * s)
        if hi == 0:
            r, g, b = v, n, l
        elif hi == 1:
            r, g, b = m, v, l
        elif hi == 2:
            r, g, b = l, v, n
        elif hi == 3:
            r, g, b = l, m, v
        elif hi == 4:
            r, g, b = n, l, v
        else:
            r, g, b = v, l, m
        dst[y, x, 0] = min(255, max(0, int(r * 255 + 0.5)))
        dst[y, x, 1] = min(255, max(0, int(g * 255 + 0.5)))
        dst[y, x, 2] = min(255, max(0, int(b * 255 + 0.5)))


image = np.array(Image.open("./input.jpg"))
height, width, _ = image.shape
d_image = cuda.to_device(image)
d_hue = cuda.device_array((height, width), dtype=np.float32)
d_saturation = cuda.device_array_like(d_hue)
d_value = cuda.device_array_like(d_hue)
d_result = cuda.device_array_like(d_image)
block_size = (16, 16)
grid_size = ((width + block_size[0] - 1) // block_size[0],
             (height + block_size[1] - 1) // block_size[1])
RGB2HSV[grid_size, block_size](d_image, d_hue, d_saturation, d_value)
HSV2RGB[grid_size, block_size](d_hue, d_saturation, d_value, d_result)
result = d_result.copy_to_host()
difference = np.abs(image.astype(np.int16) - result.astype(np.int16))
print("Maximum channel difference:", difference.max())
print("Different pixels:", np.count_nonzero(np.any(difference != 0, axis=2)))
Image.fromarray(result).save("./restored.png")

import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage.data import astronaut
import math

def rgb_to_hsv(x):
    img = x
    array = np.array(img)
    r = array.shape[0]
    g = array.shape[1]
    b = array.shape[2]

    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx

    array.shape[0] = h
    array.shape[1] = s
    array.shape[2] = v
    x = array
    return x

img = astronaut()   # Get the image

img_as_hsv = rgb_to_hsv(img)

plt.imshow(img_as_hsv)
plt.show()
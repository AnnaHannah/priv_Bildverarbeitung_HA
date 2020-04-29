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



def hsv_to_rgb(x):
    img = x
    array = np.array(img)
    h = array.shape[0]
    s = array.shape[1]
    v = array.shape[2]

    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    array.shape[0] = r
    array.shape[1] = g
    array.shape[2] = b
    x = array
    return x

img = astronaut()   # Get the image

img_as_hsv = rgb_to_hsv(img)

plt.imshow(img_as_hsv)
plt.show()
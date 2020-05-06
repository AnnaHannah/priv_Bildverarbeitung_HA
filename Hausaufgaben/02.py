import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage.data import astronaut
from skimage.color import rgb2gray
pylab.rcParams['figure.figsize'] = (12, 12)

img = rgb2gray(astronaut() / 255.)
plt.imshow(img, cmap='gray')
plt.show()

def derive_y(image):
    """Computes the derivative of the image w.r.t the y coordinate"""
    derived_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if y + 1 < image.shape[1] and y - 1 > 0:
                derived_image[x,y] = (image[x, y + 1] - image[x, y - 1]) / 2.0
    return derived_image

def derive_x(image):
    """Computes the derivative of the image w.r.t the x coordinate"""
    derived_image = np.zeros_like(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if x + 1 < image.shape[1] and x - 1 > 0:
                derived_image[x,y] = (image[x + 1, y] - image[x - 1, y]) / 2.0
    return derived_image



dx_img = derive_x(img)
dy_img = derive_y(img)

plt.figure(figsize=(18, 12))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.imshow(dx_img, cmap='gray')
plt.subplot(133)
plt.imshow(dy_img, cmap='gray')
plt.show()




def affine_transformation(img, matrix):
    ind_hg = matrix @ np.concatenate([np.indices(img.shape).reshape(2, -1), np.ones((1, np.indices(img.shape).reshape(2, -1).shape[1]))], axis=0)
    sha = int(np.ceil(np.max(ind_hg[0, :]))), int(np.ceil(np.max(ind_hg[1, :])))

    img =bicubic_interpolation(img, np.linalg.inv(matrix) @ np.concatenate([np.indices(np.zeros(sha).shape).reshape(2, -1),
                                                                              np.ones((1, np.indices(np.zeros(sha).shape).reshape(2, -1).shape[1]))], axis=0), matrix, np.zeros(sha))
    return img


def bicubic_interpolation(img, indicies, m, r):

    dx_img = derive_x(img)
    dy_img = derive_y(img)
    dxy_img = derive_x(dy_img)

    xsize = img.shape[0]
    ysize = img.shape[1]

    #wikipedia (https://en.wikipedia.org/wiki/Bicubic_interpolation):
    inv_matrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
        [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],
        [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1],
        [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],
        [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],
        [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]
    ])

    for i in range(indicies.shape[-1]):
        p = indicies[:, i]

        x_vfloor = int(np.floor(p[0]))
        y_vfloor = int(np.floor(p[1]))
        x_vceil = int(np.ceil(p[0]))
        y_vceil = int(np.ceil(p[1]))


        x_p2 = (p[0] - x_vfloor) ** 2
        y_p2 = (p[1] - y_vfloor) ** 2
        x_p3 = (p[0] - x_vfloor) ** 3
        y_p3 = (p[1] - y_vfloor) ** 3

        if 0 < x_vfloor < xsize and \
                0 < y_vfloor < ysize and \
                0 < x_vceil < xsize and \
                0 < y_vceil < ysize:

            fvalues = np.array([
                img[x_vfloor][y_vfloor], img[x_vceil][y_vfloor], img[x_vfloor][y_vceil],
                img[x_vceil][y_vceil],

                dx_img[x_vfloor][y_vfloor], dx_img[x_vceil][y_vfloor], dx_img[x_vfloor][y_vceil],
                dx_img[x_vceil][y_vceil],

                dy_img[x_vfloor][y_vfloor], dy_img[x_vceil][y_vfloor], dy_img[x_vfloor][y_vceil],
                dy_img[x_vceil][y_vceil],

                dxy_img[x_vfloor][y_vfloor], dxy_img[x_vceil][y_vfloor], dxy_img[x_vfloor][y_vceil],
                dxy_img[x_vceil][y_vceil]
            ])
            r[int(np.rint((m @ p)[0]))][int(np.rint((m @ p)[1]))] = (inv_matrix @ fvalues)[0] + (inv_matrix @ fvalues)[4] * (p[1] - y_vfloor) + (inv_matrix @ fvalues)[8] * y_p2 + (inv_matrix @ fvalues)[12] * y_p3 \
                                                                            + ((inv_matrix @ fvalues)[1] + (inv_matrix @ fvalues)[5] * (p[1] - y_vfloor) + (inv_matrix @ fvalues)[9] * y_p2 + (inv_matrix @ fvalues)[13] * y_p3) * (p[0] - x_vfloor) \
                                                                             + ((inv_matrix @ fvalues)[2] + (inv_matrix @ fvalues)[6] * (p[1] - y_vfloor) + (inv_matrix @ fvalues)[10] * y_p2 + (inv_matrix @ fvalues)[14] * y_p3) * x_p2 \
                                                                             + ((inv_matrix @ fvalues)[3] + (inv_matrix @ fvalues)[7] * (p[1] - y_vfloor) + (inv_matrix @ fvalues)[11] * y_p2 + (inv_matrix @ fvalues)[15] * y_p3) * x_p3 \


    return r

T_scale = np.array([
    [0.75, 0, 0],
    [0, 0.75, 0],
    [0, 0, 1],
])
#
T_affine = np.array([
    [1, 0.3, 0],
    [0, 1, 0],
    [0, 0, 1],
])

img_scale = affine_transformation(img, T_scale)
img_affine = affine_transformation(img, T_affine)

plt.imshow(img_scale, cmap='gray')
plt.show()
plt.imshow(img_affine, cmap='gray')
plt.show()

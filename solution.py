import cv2
import numpy as np
import math
import os

if not os.path.exists("results"):
    os.makedirs("results")

# Zadanie na ocenę dostateczną
def renew_pictures():
    kernelOnes = np.ones((3 ,3), np.uint8)

    img1 = cv2.imread("figures/crushed.png", 0)
    img1 = cv2.erode(img1, kernelOnes, iterations = 1)
    cv2.imwrite("results/crushed.png", img1)

    img2 = cv2.imread("figures/crushed2.png", 0)
    img2 = cv2.dilate(img2, kernelOnes, iterations = 1)
    img2 = cv2.erode(img2, kernelOnes, iterations = 2)
    cv2.imwrite("results/crushed2.png", img2)

    img3 = cv2.imread("figures/crushed3.png", 0)
    img3 = cv2.dilate(img3, kernelOnes, iterations = 1)
    img3 = cv2.erode(img3, kernelOnes, iterations = 2)
    img3 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernelOnes)
    cv2.imwrite("results/crushed3.png", img3)

    img4 = cv2.imread("figures/crushed4.png", 0)
    img4 = cv2.morphologyEx(img4, cv2.MORPH_CLOSE, kernelOnes, iterations = 2)
    img4 = cv2.dilate(img4, kernelOnes, iterations = 1)
    img4 = cv2.erode(img4, kernelOnes, iterations = 2)
    img4 = cv2.morphologyEx(img4, cv2.MORPH_OPEN, kernelOnes)
    cv2.imwrite("results/crushed4.png", img4)
    pass

renew_pictures()

# Zadanie na ocenę dobrą
def own_simple_erosion(image):
    new_image = np.zeros(image.shape, dtype = image.dtype)
    height = image.shape[0]
    width = image.shape[1]

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

    for y in range(0, height):
        for x in range(0, width):            
            setZero = kernel[1][1] == 1 and image[y][x] == 0

            # loops are able to process any 3x3 kernel matrix
            # corners detection is used to avoid addind padding pixels

            # left top corner
            if x == 0 and y == 0:
                setZero = setZero or (
                    (kernel[1][2] == 1 and image[y+0][x+1] == 0) or
                    (kernel[2][2] == 1 and image[y+1][x+1] == 0) or
                    (kernel[2][1] == 1 and image[y+1][x+0] == 0))

            # right top corner
            elif x == width - 1 and y == 0:                
                setZero = setZero or (
                    (kernel[1][0] == 1 and image[y+0][x-1] == 0) or
                    (kernel[2][0] == 1 and image[y+1][x-1] == 0) or
                    (kernel[2][1] == 1 and image[y+1][x+0] == 0))

            # left bottom corner
            elif x == 0 and y == height - 1:
                setZero = setZero or (
                    (kernel[0][0] == 1 and image[y-1][x-1] == 0) or
                    (kernel[0][1] == 1 and image[y-1][x+0] == 0) or
                    (kernel[1][0] == 1 and image[y+0][x-1] == 0))

            # right bottom corner
            elif x == width -1 and y == height - 1:
                setZero = setZero or (
                    (kernel[0][1] == 1 and image[y-1][x+0] == 0) or
                    (kernel[0][2] == 1 and image[y-1][x+1] == 0) or
                    (kernel[1][2] == 1 and image[y+0][x+1] == 0))
                
            # each pixel that is not in the corner
            else:
                setZero = setZero or (
                    (kernel[0][0] == 1 and image[y-1][x-1] == 0) or
                    (kernel[0][1] == 1 and image[y-1][x+0] == 0) or
                    (kernel[0][2] == 1 and image[y-1][x+1] == 0) or
                    (kernel[1][0] == 1 and image[y+0][x-1] == 0) or
                    (kernel[1][1] == 1 and image[y+0][x+0] == 0) or
                    (kernel[1][2] == 1 and image[y+0][x+1] == 0) or
                    (kernel[2][0] == 1 and image[y+1][x-1] == 0) or
                    (kernel[2][1] == 1 and image[y+1][x+0] == 0) or
                    (kernel[2][2] == 1 and image[y+1][x+1] == 0))

            if setZero:
                new_image[y][x] = 0
            else:
                new_image[y][x] = image[y][x]
                
    return new_image

cv2.imwrite("results/crushed_own_simple_erosion.png",
            own_simple_erosion(cv2.imread("figures/crushed.png", 0)))

# Zadanie na ocenę bardzo dobrą
def own_erosion(image, kernel=None):
    new_image = np.zeros(image.shape, dtype=image.dtype)

    if kernel is None:
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])
    return new_image

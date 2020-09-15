import numpy as np
import math as m
import cv2


def normalize(im):
    """
    Returns a normalized copy of an image.

    Args:
        im (ndarray): image which will be normalized (may be uint8 or float64).

    Returns:
        nrm (ndarray): normalized image with the same type as im
    """
    if im.dtype == 'uint8':
        nrm = 255 * np.uint8(im / np.max(im))
    else:
        nrm = np.float64(im / np.max(im))
    return nrm

def luminance(im):
    """
    Returns a single-channel (grayscale) image based on its luminance.

    Args:
        im (ndarray): Image in a numpy array

    Returns:
        lum (ndarray): Normalized grayscale image based on luminance
    """

    if im.shape[2] == 3:  # input must be a color image

        r = im[::, ::, 0]  # red channel
        g = im[::, ::, 1]  # green channel
        b = im[::, ::, 2]  # blue channel
        lum0 = 0.299 * r + 0.587 * g + 0.114 * b

        lum = normalize(lum0)
        return lum

    else:  # not a color image
        print('Cannot compute luminance. Not a color image. Exiting')
        return None

def mean_im(im):
    """
    Returns a single-channel (grayscale) image based on the mean of 
    the three channels.

    Args:
        im (ndarray): Image in a numpy array

    Returns:
        mean (ndarray): Normalized grayscale image based on channel mean
    """

    if im.shape[2] == 3:  # input must be a color image

        r = im[::, ::, 0]  # red channel
        g = im[::, ::, 1]  # green channel
        b = im[::, ::, 2]  # blue channel

        mean0 = np.mean(r, g, b)
        mean = normalize(mean0)

        return mean
    else:  # not a color image
        print('Cannot compute mean image. Not a color image. Exiting')
        return None

def equalize_adap(im, dil_size, blur_size):
        """
        Further improvements to better detect borders in the image
        (dilation, blur and final normalization)

        Args:
            im (ndarray): Description
            dil_size (int): Size of the dilation kernel
            blur_size (int): Size of the blur kernel

        Returns:
            eq: Transformed image
        """
        dil = cv2.dilate(im.copy(), np.ones((dil_size, dil_size)))
        bg = cv2.medianBlur(dil, blur_size)
        diff = 255 - cv2.absdiff(im, bg)
        eq = cv2.normalize(diff, diff.copy(), alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return eq


def close_borders(im, kernel_size):
    """
    Applies a closing operation where the kernel is a NxN square

    Args:
        im (ndarray): Single-channel binary image
        kernel_size (int): Kernel size (N)

    Returns:
        closed (ndarray): Single-channel binary image with closed borders
    """
    # applies a closing operation where the kernel is a NxN square
    # kernel (structuring element)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    return closed


def order_coords(coords):
    """
    Orders coordinates from a four-point path clockwise.
    Starts from top left

    Args:
        coords (ndarray): Coordinates of the four points

    Returns:
        rect (ndarray): Ordered coordinates of the four points
    """
    rect = np.zeros((4, 2), dtype='float32')
    s = coords.sum(axis=1)
    rect[0] = coords[np.argmin(s)]
    rect[2] = coords[np.argmax(s)]
    d = np.diff(coords, axis=1)
    rect[1] = coords[np.argmin(d)]
    rect[3] = coords[np.argmax(d)]
    return rect


def compute_angle(a, b, c):
    """
    Computes the angle ABC formed by three points.

    Args:
        a, b, c (ndarray): Coordinates of the three points A, B, C respectively

    Returns:
        angle (float): Angle formed by the points a, b, c
    """
    v1 = b - a
    v2 = c - b
    if np.dot(v1, v1) != 0 and np.dot(v2, v2) != 0:
        angle = m.degrees(m.acos(np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))))
        return angle
    else:
        return None


def is_rectangle(coords):
    """
    Returns whether four points in an image can form a rectangle.
    First, the function orders the four points clockwise using the order_coords
    function. 
    The points form a rectangle if all angles are between 80 and 100 degrees
    (ideally they would be right angles, i.e. 90 degrees).

    Args:
        coords (ndarray): Coordinates of the four points

    Returns:
        True if the points form a rectangle, False otherwise
    """
    coords[:, 0, :] = order_coords(coords[:, 0, :])
    vertices = np.array(coords[:, 0, :])

    for i in range(4):
        a = compute_angle(vertices[i - 1], vertices[i], vertices[(i + 1) % 4])
        if a is None:
            return False
        elif np.abs(a - 90) > 10:
            return False
    return True


def four_point_transform(im, path, approx):
    rect = order_coords(approx)
    (tl, tr, br, bl) = rect

    # im_book = im[int(min(tl[1], tr[1])) : int(max(bl[1], br[1])),
    #              int(min(tl[0], bl[0])) : int(max(tr[0], br[0]))
    #              ]
    # cv2.imshow('Book', im_book)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(width_bottom), int(width_top))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_right), int(height_left))

    # dimensions of the "corrected" image
    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]], dtype='float32')

    persp_matrix = cv2.getPerspectiveTransform(np.float32(rect), dst)
    correct = cv2.warpPerspective(im, persp_matrix, (max_w, max_h))

    return correct
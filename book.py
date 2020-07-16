"""
BOOK DETECTION ALGORITHM

This program detects if a book is present in an image.

The program can be started from the command line,
    python book.py [path],
where [path] is the path of the image from the current working directory.

It can also be started with no arguments,
    python book.py.
In that case [path] is introduced inside the program with a prompt.

The program tries to find the book using the "none" mode. If no book is found,
an additional gaussian blur is applied and the program tries again with the
"none" mode. If no book is found either, the program will use the "light" or
"dark" mode twice, first with no additional blur and then with blur.
"""

#%%
# LIBRARIES
import numpy as np
from skimage.color import rgb2hsv
import cv2
import traceback
import os
import math as m
from sys import argv
#%%


def normalize(im):
    """
    Normalizes an image.

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
    Creates a single-channel (grayscale) image based on its luminance with
    the formula L=0.299*R+0.587*G+0.114*B,
    where R,G,B are the mean values of each channel.

    Args:
        im (ndarray): Image in a numpy array

    Returns:
        lum (ndarray): Normalized grayscale image based on luminance
    """
    # creates a single-channel image based on its luminance with the formula
    # L=0.299*R+0.587*G+0.114*B,
    # where R,G,B are the mean values of each channel.

    if im.shape[2] >= 3:
        lum0 = 0.299 * im[::, ::, 0] + 0.587 * im[::, ::, 1] + 0.114 * im[::, ::, 2]
        if im.dtype == 'uint8':  # we have been working with floats
            lum = np.uint8(255 * lum0 / np.max(lum0))
        else:
            lum = np.float64(lum0 / np.max(lum0))
        return lum
    else:  # not a color image
        print('Cannot compute luminance. Not a color image. Exiting')
        return None


def mean_im(im):
    """
    Creates a single-channel (grayscale) image which is the mean of 
    the three channels.

    Args:
        im (ndarray): Image in a numpy array

    Returns:
        mean (ndarray): Normalized grayscale image based on channel mean
    """

    if im.shape[2] >= 3:
        mean0 = (im[::, ::, 0] + im[::, ::, 1] + im[::, ::, 2]) * 0.333
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


#%%
def find_single_book(im_orig, im, scf, err_perim=0.05,
                     simbg='none', denoise=False):
    """
    Finds a book in an image.

    Args:
        im_orig (ndarray): Original image
        im (ndarray): Original image
        scf (float): Scale factor to resize the original image
        err_perim (float, optional): Error in the perimeter computation
        simbg (str, optional):
            the "simbg" argument accounts for the books which have a similar
            luminance or color as the background, thus needing other channels
            to segment the image
        denoise (bool, optional): True if additional denoising must be performed

    Returns:
        TYPE: Description
    """
    if im.ndim != 3:
        print('Not a valid image')
        return None
    if im.dtype != 'uint8':
        im = np.uint8(255 * im)
    # resizes the image
    im = cv2.resize(im_orig, None, fx=1. / scf, fy=1. / scf)
    # the "simbg" argument accounts for the books which have a similar
    # luminance or color as the background, thus needing other channels
    # to segment the image
    if simbg == 'light':  # light background, taking saturation
        im_hsv = rgb2hsv(im)[::, ::, 1]
    elif simbg == 'dark':  # dark background, taking V channel
        im_hsv = rgb2hsv(im)[::, ::, 2]
    elif simbg == 'none':  # different color as background, taking luminance
        im_hsv = luminance(im)
    else:
        print('Not a valid argument for simbg')
        return None
    im_eq = equalize_adap(np.uint8(255 * im_hsv), 7, 15)

    # Additional denoising (Gaussian blur)
    if denoise:
        im_to_edge = cv2.GaussianBlur(im_eq.copy(), (3, 3), 0)
    else:
        im_to_edge = im_eq.copy()

    # The "simbg" mode controls the Canny thresholds
    if simbg == 'light':
        im_edges = cv2.Canny(im_to_edge, 20, 70)
    elif simbg == 'dark':
        im_edges = cv2.Canny(im_to_edge, 20, 250)
    elif simbg == 'none':
        im_edges = cv2.Canny(im_to_edge, 25, 500)
    cv2.imwrite('a.jpg', im_edges)

    # image with closed borders
    im_closed = close_borders(im_edges, kernel_size=5)
    cv2.imwrite('b.jpg', im_closed)

    # image where contours are drawn on
    im_cont = im.copy()

    # finds the contours and sorts by perimeter; the book's will be the first one
    cnts, h = cv2.findContours(im_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(cnts)
    cnts_sorted = sorted(cnts, key=lambda x: cv2.arcLength(x, closed=True))
    found = []
    im_perim = 2 * (im.shape[0] + im.shape[1])
    for c in cnts_sorted:
        p = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, err_perim * p, True)

        # The contour must have 4 vertices and a minimum perimeter
        # this is done to not consider false positives such as image noise
        if len(approx) == 4 and p > 0.7 * im_perim and is_rectangle(approx):
            found.append(approx)
    if len(found) == 1:
        cv2.drawContours(im_cont, [approx], -1, (0, 255, 0), 3)
        info = dict({'scf': scf, 'cont': np.array(found[0]) * scf})
        return im_cont, info, 1
    elif len(found) > 1:
        pass
        # print('More than one possible contour found.')
        # TODO: print all contours with different colors (use mpl.cmap?)
        # TODO: label contours and let the user decide which contour is the suitable one
    elif len(found) == 0:
        # print('No books found.')
        pass
    if scf < 10:
        return find_single_book(im_orig, im, scf + 1, err_perim, simbg, denoise)
    else:
        return None, None, 0


def four_point_transform(im, im_path, approx):
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


#%%
try:
    im_name = argv[1]
except:
    im_name = input('Image path:\n> ')

im_path = os.path.join(os.getcwd(), im_name)
im = cv2.imread(im_path)

try:
    mode = 'none'
    im_cont, info, found = find_single_book(im, im, scf=1, err_perim=0.05,
                                            denoise=False, simbg='none')
    if found == 0:
        im_cont, info, found = find_single_book(im, im, scf=1, err_perim=0.05,
                                                denoise=False, simbg='none')
    if found == 0:
        if np.mean(np.uint8(luminance(im))) > 128:
            mode = 'light'
        else:
            mode = 'dark'
        im_cont, info, found = find_single_book(im, im, scf=1, err_perim=0.05,
                                                denoise=True, simbg=mode)
        if found == 0:
            im_cont, info, found = find_single_book(im, im, scf=1, err_perim=0.05,
                                                    denoise=False, simbg=mode)
        if found == 0:
            print('No book detected.')
    if found:
        print('Book detected.')
        cv2.imshow('Book detected', im_cont)
        # cv2.imwrite('./res/book_nocorr.jpg', im_cont)
        corr = cv2.resize(four_point_transform(im, im_path,
                                               info['cont'][:, 0, :]),
                          None, fx=1 / 4, fy=1 / 4)
        cv2.imshow('Book detected (perspective corrected)', corr)
        cv2.imwrite('./res/book_persp_corr.jpg', corr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
except:
    traceback.print_exc()

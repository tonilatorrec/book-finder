#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.color import rgb2hsv
import cv2
import traceback
import os
import argparse
from imedit import *

class Image():

    def __init__(self, im):
        self.orig = im
        self.edited = self.orig.copy()
        self.contour = self.orig.copy()
        self.persp_corr = self.orig.copy()
        self.scf = 1
        self.cntr = []
        self.book = Book()
        self.book_found = False

class Book():

    def __init__(self):
        pass


def find_single_book(image, scf, err_perim=0.05,
                     mode='normal', denoise=False):
    """
    Finds a book in an image.

    Args:
        im_orig (ndarray): Original image
        im (ndarray): Original image
        scf (float): Scale factor to resize the original image
        err_perim (float, optional): Error in the perimeter computation
        mode (str, optional):
            the "mode" argument accounts for the books which have a similar
            luminance or color as the background, thus needing other channels
            to segment the image
        denoise (bool, optional): True if additional denoising must be performed

    Returns:
        TYPE: Description
    """

    im = image.orig.copy()
    im = cv2.resize(im, None, fx=1. / scf, fy=1. / scf)

    if im.dtype != 'uint8':  # the program works with 8-bit images
        im = np.uint8(255 * im)

    if im.ndim == 3:  # creates a grayscale image
        if mode == 'light':  # light on light background, takes saturation
            im = rgb2hsv(im)[::, ::, 1]
        elif mode == 'dark':  # dark on dark background, takes V channel
            im = rgb2hsv(im)[::, ::, 2]
        elif mode == 'normal':  # takes luminance
            im = luminance(im)

    im = equalize_adap(np.uint8(255 * im), 7, 15)  # equalization

    if denoise:  # additional denoising (Gaussian blur)
        im = cv2.GaussianBlur(im_eq.copy(), (3, 3), 0)

    # Canny edge detection
    # the Canny thresholds depend on the mode
    if mode == 'light':
        edges = cv2.Canny(im, 20, 70)
    elif mode == 'dark':
        edges = cv2.Canny(im, 20, 250)
    elif mode == 'normal':
        edges = cv2.Canny(im, 25, 500)
    # cv2.imwrite('a.jpg', edges)

    # border closing
    closed_borders = close_borders(edges, kernel_size=5)
    # cv2.imwrite('b.jpg', closed_borders)

    # finds contours and sorts by perimeter
    # the contour with the largest perimeter is assumed to be the book's
    cntr, h = cv2.findContours(closed_borders,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cntr_sorted = sorted(cntr, key=lambda x: cv2.arcLength(x, closed=True))
    cntr_candidates = []
    im_perim = 2 * (im.shape[0] + im.shape[1])
    for c in cntr_sorted:
        p = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, err_perim * p, True)

        # The contour must have 4 vertices and a minimum perimeter
        # this is done to not consider false positives such as image noise
        if len(approx) == 4 and p > 0.7 * im_perim and is_rectangle(approx):
            cntr_candidates.append(approx)
    if len(cntr_candidates) == 1:  # only one candidate: book found
        # draw_contours(im=contour, vrt=vertices, color=(0, 255, 0))
        cv2.drawContours(image.contour, [approx], -1, (0, 255, 0), 3)
        image.scf = scf
        image.cntr = np.array(cntr_candidates[0] * scf)
        image.book_found = True
    elif len(cntr_candidates) > 1:  # more than one candidate
        pass
        # TODO: print all contours with different colors (use mpl.cmap?)
        # TODO: label contours and let the user decide which contour is the suitable one
    elif len(cntr_candidates) == 0:
        # print('No books found.')
        pass
    if scf < 10:
        return find_single_book(image, scf + 1, err_perim, mode, denoise)
    else:
        print('No books found.')
        return None, None, 0

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("-p", "--path", required=True,
                    help="Relative path of the image")

    args = ap.parse_args()

    path = os.path.join(os.getcwd(), args.path)
    image = Image(cv2.imread(path))

    try:
        mode = 'normal'
        find_single_book(image, scf=1, err_perim=0.05,
                                denoise=False, mode=mode)
        if not image.book_found:
            find_single_book(image, scf=1, err_perim=0.05,
                             denoise=True, mode=mode)
            if not image.book_found:
                if np.mean(np.uint8(luminance(im))) > 128:
                    mode = 'light'
                else:
                    mode = 'dark'
                find_single_book(image, scf=1, err_perim=0.05,
                                 denoise=True, mode=mode)
                if not image.book_found:
                    find_single_book(image, scf=1, err_perim=0.05,
                                     denoise=False, mode=mode)
        if image.book_found:
            print('Book detected.')
            cv2.imshow('Book detected', image.edited)
            image.persp_corr = cv2.resize(four_point_transform(image.orig, path,
                                                   image.cntr[:, 0, :]),
                              None, fx=1 / 4, fy=1 / 4)
            cv2.imshow('Book detected (perspective corrected)', image.persp_corr)
            cv2.imwrite('./res/book_persp_corr.jpg', image.persp_corr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except:
        traceback.print_exc()

if __name__ == '__main__':
    main()

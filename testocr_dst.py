# -*- encoding: utf-8 -*-
from PIL import Image
from pytesseract import *
import cv2 as cv
import numpy as np

def detect_text(filename, save=False):
    # filename = r'../images/CaptureFont/beer_mo1.png'
    image = Image.open(filename)
    print(image)

    image = np.array(image)
    # dst = cv.resize(image, dsize=(800,800), interpolation=cv.INTER_CUBIC)
    # print(type(dst), dst.shape)
    # grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, img_result = cv.threshold(grayscale, 200, 255, cv.THRESH_BINARY)
    # blur = cv.blur(image, (3,3), anchor=(-1,-1), borderType=cv.BORDER_REFLECT_101)
    # cv.imshow("img", dst)
    # cv.waitKey()
    # cv.imshow("img", image)
    # cv.waitKey()
    text = image_to_string(image, lang='kor+eng', config='--psm 10')
    # print("[", text, "]")
    if save:
        with open('D:/pyproject/bitopencv/images/poster_ocr/mo_ocr.txt', 'w') as f:
            f.write(text)
    return text
if __name__ == "__main__":
    text = detect_text(r'../images/CaptureFont/pre_jo.png')

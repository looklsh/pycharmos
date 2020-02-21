import os
import cv2 as cv
import numpy as np
from findfont.testocr_dst import detect_text

def load_image(file_path):
    with open(file_path, 'rb') as f:
        barr = bytearray(f.read())
    barr = np.array(barr, dtype=np.uint8)
    img = cv.imdecode(barr, cv.IMREAD_ANYCOLOR)
    return img

def convert_image(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, img_result = cv.threshold(grayscale, 127, 255, cv.THRESH_BINARY)
    return img_result

def reverse_image(image):
    reverse = cv.bitwise_not(image)
    return reverse

def remove_edge(image):
    inv_image = 255 * (image < 128).astype(np.uint8)
    coords = cv.findNonZero(inv_image)
    x, y, w, h = cv.boundingRect(coords)
    rect = image[y: y+h, x: x+w]
    return rect

def show_image(image):
    cv.imshow('compare image', image)
    cv.waitKey()
    cv.destroyAllWindows()

def capture_image():
    file_path1 = "../images/CaptureFont/life_mi.png"
    image1 = load_image(file_path1)
    image1 = convert_image(image1)
    image1 = remove_edge(image1)
    detected_text = detect_text(file_path1, save=True)
    print("OCR : [", detected_text, "]")
    return image1, detected_text

def matching_exe():
    image1, detected_text = capture_image()
    count_of_pixel = image1.shape[0] * image1.shape[1]
    print('이미지의 픽셀 수 : ', image1.shape[0] * image1.shape[1])
    dir_name = "./abc/{}".format(detected_text)
    print("검색 대상 :", dir_name)
    dct = {}
    for file in os.listdir(dir_name):
        file_path = "/".join([dir_name, file])
        print("비교할 대상 : " + "/".join([dir_name, file]))
        image2 = load_image(file_path)
        image2 = convert_image(image2)
        image2 = remove_edge(image2)
        image2 = cv.resize(image2, dsize=(image1.shape[1], image1.shape[0]))

        diff = image1 == image2
        diff_image = np.zeros_like(image1)
        diff_image[diff] = 255
        diff_pixel = count_of_pixel - np.count_nonzero(diff)
        accuracy = (count_of_pixel - (diff_pixel)) * 100 / count_of_pixel
        print('정확도는 {}%입니다'.format(np.round(accuracy, 1)))
        # dct[file] = '{}%'.format(np.round(accuracy, 1)) # 키 = 값
        dct[file] = accuracy

    sort_value = sorted(dct.items(), key=(lambda x: x[1]), reverse=True)
    # print("DICT:", dct)
    print('SORT VALUE:', sort_value[0:2])

    # filter_value = {key: value for key, value in dct.items() if value > 90}
    # print('FILTER VALUE', filter_value)

if __name__ == '__main__':
    matching_exe()




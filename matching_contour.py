import cv2 as cv
import numpy as np
# from findfont import testocr_dst
from findfont.testocr_dst import detect_text
import os


def load_image(file_path):
    '''
    이미지 불러오기 함수
    디렉터리 명에 한글이 포함되면 opencv는 해당 path를 인식하지 못함
    임의로 python에서 bytearray로 불러와서 opencv에서 이미지를 인코딩하는 방식을 취함
    '''
    with open(file_path, 'rb') as f:
        barr = bytearray(f.read())
    barr = np.array(barr, dtype=np.uint8)

    img = cv.imdecode(barr, cv.IMREAD_ANYCOLOR)
    return img

def convert_image(image):
    '''
    활용하고자 하는 이미지에서 색상은 중요하지 않으므로 완전 흑백 이미지로 변경함
    주의: threshold 분기점을 잘 조정할 것
    '''
    # 진한 글씨용
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, img_result = cv.threshold(grayscale, 127, 255, cv.THRESH_BINARY)
    return img_result



def remove_edge(image):
    '''
    글자만 이미지 내에 가득 채우기 위해 테두리를 모두 없앰
    '''
    inv_image = 255 * (image < 128).astype(np.uint8) # 글꼴 영역을 찾기 위해 반전
    coords = cv.findNonZero(inv_image) # 반전 이미지에서 0이 아닌 부분을 찾음
    x, y, w, h = cv.boundingRect(coords)

    rect = image[y: y+h, x: x+w]
    return rect

def show_image(image):
    cv.imshow('img', image)
    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    # file_path1 = './abc/가/HANSolM_가.png'
    file_path1 = '../images/CaptureFont/band3.png'
    # print("OCR:", detect_text(file_path1))
    detected_text = detect_text(file_path1, save=True)
    print("OCR:", detected_text)

    image1 = load_image(file_path1)
    image1 = convert_image(image1)
    image1 = remove_edge(image1)

    show_image(image1)

    count_of_pixel = image1.shape[0] * image1.shape[1]

    print('shape of image1:', image1.shape[0] * image1.shape[1])

    # TODO
    # 1. ocr text로 디렉터리 찾아가서
    # 2. 디렉터리 내의 모든 파일을 loop
    # 3. 파일 여러개 -> dict, list

    dir_name = "./abc/{}".format(detected_text)
    print("TODO: 검색 대상 dir:", dir_name)

    # for files in enumerate(os.walk(dir_name)):
    for files in os.listdir(dir_name):
        print("/".join([dir_name, files]))


        # file_path2 = '../images/CaptureFont/gajua_ga.png'
        file_path2 = './abc/시/BinggraeTaom_시.png'
        image2 = load_image(file_path2)
        image2 = convert_image(image2)
        image2 = remove_edge(image2)
        image2 = cv.resize(image2, dsize=(image1.shape[1], image1.shape[0]))

    show_image(image2)

    diff = image1 == image2
    # print('image1:', image1)
    # print('image2:', image2)
    # print('diff:', diff)
    diff_image = np.zeros_like(image1)
    diff_image[diff] = 255
    # diff_image[diff < 128] = 0


    show_image(diff_image)

    diff_pixel = count_of_pixel - np.count_nonzero(diff)
    print('DIFF Pixels:', diff_pixel)

    accuracy = (count_of_pixel-(count_of_pixel-np.count_nonzero(diff))) * 100 / count_of_pixel
    print('정확도는 {}%입니다'.format(np.round(accuracy, 1)))

    if accuracy > 90:
        print('일치합니다')
    else:
        print('일치하지 않습니다')
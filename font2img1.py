import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm



font_path = "../font/"
fonts = os.listdir(font_path) # font폴더에 있는 파일

_0xFF = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
start_0xFF = "AC00" # 가
end_0xFF = "D7A3" # 힣
_0xFF = _0xFF.split(" ") # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']

ko_Syllables = [w + x + y + z for w in _0xFF for x in _0xFF for y in _0xFF for z in _0xFF] # 글자 조합
ko_Syllables = np.array(ko_Syllables)

start = np.where(start_0xFF == ko_Syllables)[0][0] # 44032
end = np.where(end_0xFF == ko_Syllables)[0][0] # 55203

ko_Syllables = ko_Syllables[start:end + 1] # ['AC00' 'AC01' 'AC02' ... 'D7A1' 'D7A2' 'D7A3']
unicodeChars = chr(int(ko_Syllables[0], 16)) # 가
# chr(): 아스키코드 값을 문자로 변환해주는 함수 , chr(44032)
# ko_syllables[0] == AC00
# print(len(ko_Syllables))


def generate_font_img(font_path, font_name):
    font_full_path = font_path + font_name
    # print("FFP:", font_full_path)
    font = ImageFont.truetype(font=font_full_path, size=100)

    for unicode in tqdm(ko_Syllables):
        # 글자별 디렉터리 만들고
        unicodeChars = chr(int(unicode, 16))
        path = "./abc/" + unicodeChars
        os.makedirs(path, exist_ok=True)

        msg = path + "/" + font_name[:-4] + "_" + unicodeChars # font_name[:-4] : .ttf전까지 슬라이싱
        file_name = "{}.png".format(msg)
        # 만약, file_name이 있으면 다시 안만들어도 됨
        if os.path.isfile(file_name):
            continue

        x, y = font.getsize(unicodeChars) # getsize(): 바이트 단위의 파일크기를 반환
        # print(x)
        if os.path.isfile(file_name):
            continue
        label = Image.new('RGB', (x + 3, y + 3), color='white') # Image.new(mode,size)는 주어진 형식의 새로운 이미지 생성
        # mode: 'RGB', 'CMYK', 'L(흑백모드)'
        # size: 가로,세로크기가 정수로 주어진 튜플
        canvas = ImageDraw.Draw(label)
        canvas.text((0.0, 0.0), unicodeChars[0], font=font, fill='black') # .text(position, string, options)

        label.save(file_name)

"""
for unicode in tqdm(ko_Syllables):
    unicodeChars = chr(int(unicode, 16))
    path = "./abc/" + unicodeChars
    os.makedirs(path, exist_ok=True)
    for ttf in fonts:
        def drawfont():
            font = ImageFont.truetype(font=font_path + ttf, size=100)
            x, y = font.getsize(unicodeChars)
            label = Image.new('RGB', (x + 3, y + 3), color='white')
            canvas = ImageDraw.Draw(label)
            canvas.text((0.0, 0.0), unicodeChars[0], font=font, fill='black')
            msg = path + "/" + ttf[:-4] + "_" + unicodeChars
            file_name = '{}.png'.format(msg)
            label.save(file_name)
            return file_name
"""

def generate_all_fonts():
    for ttf in fonts:
        generate_font_img(font_path, ttf)


if __name__ == '__main__':
    # file_name의 이름을 가진  파일이 있으면 저장하지 않음
    # file_name의 이름을 가진  파일이 없으면 저장함

    # font_save = drawfont()
    # generate_font_img(font_path, "NanumMyeongjoBold.ttf")

    generate_all_fonts()



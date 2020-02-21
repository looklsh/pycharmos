import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

font_path = "../font/"
fonts = os.listdir(font_path)

_0xFF = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
start_0xFF = "AC00" # 가
end_0xFF = "D7A3" # 힣

_0xFF = _0xFF.split(" ")

ko_Syllables = [w + x + y + z for w in _0xFF for x in _0xFF for y in _0xFF for z in _0xFF]

ko_Syllables = np.array(ko_Syllables)

start = np.where(start_0xFF == ko_Syllables)[0][0]
end = np.where(end_0xFF == ko_Syllables)[0][0]

ko_Syllables = ko_Syllables[start:end + 1]

unicodeChars = chr(int(ko_Syllables[0], 16))

for unicode in tqdm(ko_Syllables):
    unicodeChars = chr(int(unicode, 16))
    path = "./abc/" + unicodeChars
    os.makedirs(path, exist_ok=True)
    for ttf in fonts:
        font = ImageFont.truetype(font=font_path + ttf, size=100)
        x, y = font.getsize(unicodeChars)
        label = Image.new('RGB', (x + 3, y + 3), color='white')
        canvas = ImageDraw.Draw(label)
        canvas.text((0.0, 0.0), unicodeChars[0], font=font, fill='black')
        msg = path + "/" + ttf[:-4] + "_" + unicodeChars
        label.save('{}.png'.format(msg))
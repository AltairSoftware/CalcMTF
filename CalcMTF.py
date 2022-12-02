from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.transform import radon, rescale
import json


# 画像の読み込み、グレースケール化、型変換
imagePath = 'slash1.png'
targetImage = Image.open(imagePath)
targetImage = targetImage.convert("L")
targetImage = np.array(targetImage)
targetImage = np.asarray(targetImage, dtype='float64')


# ソーベルフィルタでエッジ検出
targetImage = filters.sobel(targetImage)


# ラドン変換と結果描写
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Edge")
ax1.imshow(targetImage, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., np.max(targetImage.shape), endpoint=False)


# 画像の中心を原点としてラドン変換している
sinogram = radon(targetImage, theta=theta) 

dx, dy = 0.5 * 180.0 / np.max(targetImage.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(-dx, dx + 180, -dy, sinogram.shape[0]+dy), aspect='auto')

fig.tight_layout()
fig.savefig("radon.png")


# エッジラインの位置と傾斜角を算出
position = 0
angle = 0
max = 0

xPixel = len(sinogram)
yPixel = len(sinogram[0])

for i in range(xPixel):
  for j in range(yPixel):
    if max < sinogram[i][j] :
      max = sinogram[i][j]
      position = i
      angle = (j/yPixel) * 180


# エッジラインの情報を出力
outputDict = {"Line Position: ": position, "Line Angle: ": angle}
with open('./LineInfo.json', 'w') as f:
    json.dump(outputDict, f)
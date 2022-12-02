import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.transform import radon, rescale
import json


imagePath = 'slash1.png' #処理したい画像のパス

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24, 48), tight_layout=True) #図の準備


def upsampling(img): #グレースケールの画像を入れると一つの画素を4つに分割したものを出力する
  resultImage = np.zeros((len(img)*2, len(img[0])*2))
  for i in range(len(img)):
    for j in range(len(img[0])):
      resultImage = np.append(resultImage[2*i][2*j], img[i][j])
      resultImage = np.append(resultImage[2*i+1][2*j], img[i][j])
      resultImage = np.append(resultImage[2*i][2*j+1], img[i][j])
      resultImage = np.append(resultImage[2*i+1][2*j+1], img[i][j])
  return resultImage


def calcPosAng(img): #ラドン変換後の画像を入力するとエッジラインの位置と傾斜角を返す
  pos = 0
  ang = 0
  max = 0

  xPixel = len(img)
  yPixel = len(img[0])

  for i in range(xPixel):
    for j in range(yPixel):
      if max < img[i][j]:
        max = img[i][j]
        pos = i
        ang = (j/yPixel) * 180
  return [pos, ang]

def esf(img, angle): #BGR画像とエッジラインの法線がx軸正方向と成す角度を入力すると、ESF曲線の配列を返す
  gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
  npImg = np.array(gImg, dtype='float64') #型変換
  upImg = upsampling(npImg) #アップサンプリング

  holizonShift = - (np.cos(np.radians(angle)) / 2) #アップサンプリングされたピクセルを横に移動したときのシフト量
  varticalShift = np.sin(np.radians(angle)) / 2 #アップサンプリングされたピクセルを縦に移動したときのシフト量
  esfX = np.array([]) #ESF曲線のx軸
  esfY = np.array([]) #ESF曲線のy軸

  for i in range(len(upImg)): #座標とピクセル値をそれぞれの配列に挿入
    for j in range(len(upImg[0])):
      esfY = np.append(esfY, upImg[i][j])
      esfX = np.append(esfX, varticalShift*i + holizonShift*j)

  esfX_arg = np.argsort(esfX) #座標を基準にでそれぞれの配列をソート
  esfX = esfX[esfX_arg]
  esfY = esfY[esfX_arg]

  for i in range(len(esfX)): #最初の要素が原点来るようにシフト
    esfX[i]-=esfX[0]

  binwid = holizonShift / 2 #ビニングするときのビンの幅, 本来の画素の投影間隔の1/4
  index = 0
  f = 0
  b = 0
  esfXB = np.array([]) #ビニング済みESF曲線のx軸
  esfYB = np.array([]) #ビニング済みESF曲線のy軸
  while(index*binwid < esfX[-1]):
    while(((index+1)*binwid > esfX[f]) and (f < len(esfX))):
      f += 1
    esfXB = np.append(esfXB, index)
    esfYB = np.append(esfYB, np.average(esfY[b:f]))
    index += 1
    b = f

  return [esfXB, esfYB]
  

# エッジラインの検出
targetImage = cv2.imread(imagePath) #画像の読み込み
biImage = cv2.bilateralFilter(targetImage, 15, 20, 20) #バイラテラルフィルタの適用
grayImage = cv2.cvtColor(biImage, cv2.COLOR_BGR2GRAY) #グレースケール化
otsuImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU) # 大津の二値化
npImage = np.array(otsuImage, dtype='float64') #型変換
upImage = upsampling(npImage) #アップサンプリング
axs[0,0].set_title("Grayscale and Upsampling")
axs[0,0].imshow(upImage, cmap=plt.cm.Greys_r)

sobelImage = filters.sobel(npImage) # ソーベルフィルタでエッジ検出
axs[0,1].set_title("Edge")
axs[0,1].imshow(sobelImage, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., np.max(sobelImage.shape), endpoint=False) #0~180の間でnp.max(sobelImage.shape)回サンプリング
radonImage = radon(sobelImage, theta=theta) # 画像の中心を原点としてラドン変換している,positionを使うときは注意

dx, dy = 0.5 * 180.0 / np.max(targetImage.shape), 0.5 / radonImage.shape[0]
axs[0,2].set_title("Radon transform")
axs[0,2].set_xlabel("Projection angle (deg)")
axs[0,2].set_ylabel("Projection position (pixels)")
axs[0,2].imshow(radonImage, cmap=plt.cm.Greys_r, extent=(-dx, dx + 180, -dy, radonImage.shape[0]+dy), aspect='auto')

edgeLine = calcPosAng(radonImage) # エッジラインの位置と傾斜角を算出
position, angle = edgeLine[0], edgeLine[1]
outputDict = {"Line Position: ": position, "Line Angle: ": angle} # エッジラインの情報を出力
with open('./LineInfo.json', 'w') as f:
    json.dump(outputDict, f)


ESF = esf(targetImage, angle) #ESF曲線の算出
axs[1,0].set_title("ESF curve")
axs[1,0].plot(ESF[0], ESF[1], linestyle='None', marker=".")

plt.savefig("result.png", format="png", dpi=300)
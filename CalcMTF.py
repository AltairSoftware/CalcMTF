import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.transform import radon
import json
from scipy import signal


imagePath = 'slash6.png' #処理したい画像のパス
pixelWidth = 1 #1pixelが何mmに相当するか MTFの横軸をlp/pixelにするなら1を入れる
binRate = 0.1 #サンプリングピッチに対するビン幅
maxspfr = 1 #グラフ化する際の最大空間周波数

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), tight_layout=True) #図の準備


def upsampling(img): #グレースケールの画像を入れると一つの画素を4つに分割したものを出力する
  resultImage = np.zeros((len(img)*2, len(img[0])*2))
  for i in range(len(img)):
    for j in range(len(img[0])):
      resultImage[2*i][2*j] = img[i][j]
      resultImage[2*i+1][2*j] = img[i][j]
      resultImage[2*i][2*j+1] = img[i][j]
      resultImage[2*i+1][2*j+1] = img[i][j]
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

def esf(img, angle, binrate): #BGR画像，エッジラインの法線がx軸正方向と成す角度，ビニングレートを入力すると、ESF曲線の配列を返す
  gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
  npImg = np.array(gImg, dtype='float64') #型変換

  holizonShift = np.abs(np.cos(np.radians(angle))) #ピクセルを横に移動したときのシフト量
  varticalShift = np.abs(np.sin(np.radians(angle))) #ピクセルを縦に移動したときのシフト量
  if(angle < 90):
    varticalShift = -varticalShift
  esfX = np.zeros(len(npImg) * len(npImg[0])) #ESF曲線のx軸
  esfY = np.zeros(len(npImg) * len(npImg[0])) #ESF曲線のy軸

  len_npImg_y = len(npImg[0])

  for i in range(len(npImg)): #座標とピクセル値をそれぞれの配列に挿入
    for j in range(len(npImg[0])):
      esfY[len_npImg_y * i + j] = npImg[i][j]
      esfX[len_npImg_y * i + j] = varticalShift*i + holizonShift*j

  esfX_arg = np.argsort(esfX) #座標を基準にでそれぞれの配列をソート
  esfX = esfX[esfX_arg]
  esfY = esfY[esfX_arg]

  for i in range(len(esfX)): #最初の要素が原点来るようにシフト
    esfX[i]-=esfX[0]

  f = 0
  fmax = len(esfX)-1
  b = 0

  count = int(esfX[-1]) //binrate + 1
  count = np.int32(count)
  esfXB = np.zeros(count) #ビニング済みESF曲線のx軸
  esfYB = np.zeros(count) #ビニング済みESF曲線のy軸

  for i in range(count):
    while(((i+1)*binrate) > esfX[f]):
      f += 1
      if(f > fmax): break
    esfXB[i] = i
    esfYB[i] = np.average(esfY[b:f])
    b = f

  return [esfXB, esfYB]


def diff(array): #(2, n)配列とそのサンプリングピッチを入力すると微分結果を返す
  result = np.zeros(len(array[0])-1)
  for i in range(len(array[0])-1):
    result[i] = (array[1][i+1] - array[1][i]) / (array[0][i+1] - array[0][i])
  return([array[0][:-1], result])

def lowpass(array): #ノイズ除去 ローパスフィルタ
  sos = signal.butter(5, 1, 'lowpass', output='sos', fs=10)
  return signal.sosfiltfilt(sos, array)


def mtf(lsfy, sampit): #LSFのY軸配列とサンプリングピッチ（ビニングレート適用済み）を入れるとMTFを返す
  mtfy = np.fft.fft(lsfy)
  mtfy = np.abs(mtfy)
  sampf = 1 / sampit #サンプリング周波数
  mtfx = np.arange(0, sampf/2, (sampf/len(mtfy)))
  mtfy = mtfy[:len(mtfx)]

  return [mtfx, mtfy]
  
  

# エッジラインの検出
targetImage = cv2.imread(imagePath) #画像の読み込み
biImage = cv2.bilateralFilter(targetImage, 15, 20, 20) #バイラテラルフィルタの適用
grayImage = cv2.cvtColor(biImage, cv2.COLOR_BGR2GRAY) #グレースケール化
otsuImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU) # 大津の二値化
npImage = np.array(otsuImage[1], dtype='float64') #型変換
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


ESF = esf(targetImage, angle, binRate) #ESF曲線の算出
axs[1,0].set_title("ESF curve")
# axs[1,0].set_xlim(2300, 2350)
axs[1,0].plot(ESF[0], ESF[1], marker=".", color='k', markersize=3)

# ESF[1] = lowpass(ESF[1])

LSF = diff(ESF) #LSF曲線の算出
axs[1,1].set_title("LSF curve")
# axs[1,1].set_xlim(2300, 2350)
axs[1,1].plot(LSF[0], LSF[1], marker=".", color='k', markersize=3)

MTF = mtf(LSF[1], pixelWidth*binRate)
axs[1,2].set_title("MTF curve")
axs[1,2].set_xlim(0,1)
axs[1,2].plot(MTF[0], MTF[1], linestyle="None", marker=".", color='k', markersize=3)

plt.savefig("result.png", format="png", dpi=300)
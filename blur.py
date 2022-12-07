import cv2

path =  'slash6.png'
targetImage = cv2.imread(path)
bluredImage = cv2.GaussianBlur(targetImage, (201,201),0)
bluredImage = cv2.GaussianBlur(bluredImage, (201,201),0)
cv2.imwrite('blured_' + path, bluredImage)
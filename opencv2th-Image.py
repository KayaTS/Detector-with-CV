import cv2
import numpy as np
import os

img = cv2.imread('MyPic.png')
my_roi = img[100:200, 100:200]
img[300:400, 300:400] = my_roi
cv2.imwrite('MyPic.jpg', img)
print("Writing complete.")

''' ~?zellikleri yazd?rma 
img = cv2.imread('MyPic.png')
print(img.shape) #height, width ve renk kanallar?. Gri resimde len(shape) == 2, renkli resimde len(shape) == 3,
print(img.size)  #dizideki eleman say?s?, gri resimde t?m pikseller, renkli resimde (BGR) 3 kat?
print(img.dtype) #dizinin eleman veri tipi. her renk kanal? 8 bit olan resim i?in numpy.uint8.
'''

''' ~Resimdeki bir k?sm? ba?ka bir k?sma kopyalama 
img = cv2.imread('MyPic.png')
my_roi = img[0:100, 0:100]
img[300:400, 300:400] = my_roi
cv2.imwrite('MyPic.jpg', img)
'''


''' ~Camera Capture
img = np.zeros((5, 3), dtype=np.uint8)
print(img.shape)


clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)

print('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read()
while cv2.waitKey(1) == -1 and not clicked:
    if frame is not None:
        cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()

cv2.destroyWindow('MyWindow')


 ~Png to Jpg 
image = cv2.imread('MyPic.png')
cv2.imwrite('MyPic.jpg', image)
'''

''' ~Png to grey
grayImage = cv2.imread('MyPic.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('MyPicGray.png', grayImage)
'''

''' image with window
img = cv2.imread('MyPic.png')
cv2.imshow('MyPic', img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

''' ~Random image
# Make an array of 120,000 random bytes.
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)

# Convert the array to make a 400x300 grayscale image.
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('RandomGray.png', grayImage)

# Convert the array to make a 400x100 color image.
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)
'''



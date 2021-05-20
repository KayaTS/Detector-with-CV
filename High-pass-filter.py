import cv2
import numpy as np
from scipy import ndimage

# 3x3 ve 5x5 kernel tanimlama
kernel_3x3 = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])

# Resim Gryscale-Gri tonlamalı olarak yüklenir
img = cv2.imread("./media/imu.jpg", 0)

# Çok boyutlu dizilerin evrişimi(convolution) için ngimage modülü kullanılır
k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

# LPF uygulama ve bu LPF ile orjinal fotoğrafın farkı sayesinde yeni bir HPF elde etme
# LPF olarak meşhur gaus bulanıklaştırma-yumuşatma kullanılır.
blurred = cv2.GaussianBlur(img, (17,17), 0)
g_hpf = img - blurred

cv2.imshow("original", img)
cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("blurred", blurred)
cv2.imshow("g_hpf", g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()

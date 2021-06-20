import cv2
import numpy as np

img = cv2.imread("./media/imu.jpg", 0)
cv2.imwrite("./media/canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("./media/canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()

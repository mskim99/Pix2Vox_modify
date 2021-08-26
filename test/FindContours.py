import cv2
import numpy as np

img_color = cv2.imread('I:/DK_Data_Process/i_1-2_Slices_Cropped/f_038_Axial_cropped.bmp')

img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)

ret, img_binary = cv2.threshold(img_gray, 112, 255, 0)
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    cv2.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue

cv2.imshow("result", img_color)
cv2.waitKey(0)
'''
for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area)

cv2.imshow("result", img_color)
cv2.destroyAllWindows()
'''
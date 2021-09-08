import cv2
import numpy as np

index = 28

c_index_start = 231
c_index_end = 323

s_index_start = 204
s_index_end = 309

a_index_start = 986
a_index_end = 1052

c_x = 210
c_y = 772

c_w = 90
c_h = 80

s_x = 216
s_y = 761

s_w = 100
s_h = 102

a_x = 193
a_y = 219

a_w = 121
a_h = 114

c_xray = np.zeros(shape=(c_h, c_w, 3))
s_xray = np.zeros(shape=(s_h, s_w, 3))
a_xray = np.zeros(shape=(a_h, a_w, 3))

for c_index in range (c_index_start, c_index_end + 1):
    c_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(index).zfill(3) + '/coronal/f_' + str(index).zfill(2) + '_Coronal_' + str(c_index).zfill(5) + '.bmp')
    if type(c_img) is np.ndarray:
        c_cropped_img = c_img[c_y: c_y + c_h, c_x: c_x + c_w]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/23_24_spine/Coronal/f_' + str(index).zfill(3)+ '/f_' + str(index).zfill(3)  + '_' + str(c_index).zfill(5) + '_Coronal_cropped.bmp', c_cropped_img)

        c_xray += c_cropped_img

for s_index in range (s_index_start, s_index_end + 1):
    s_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(index).zfill(3) + '/sagittal/f_' + str(index).zfill(2) + '_Sagittal_' + str(s_index).zfill(5) + '.bmp')
    if type(s_img) is np.ndarray:
        s_cropped_img = s_img[s_y: s_y + s_h, s_x: s_x + s_w]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/23_24_spine/Sagittal/f_' + str(index).zfill(3) + '/f_' + str(index).zfill(3) + '_' + str(s_index).zfill(5) + '_Sagittal_cropped.bmp', s_cropped_img)

        s_xray += s_cropped_img

for a_index in range (a_index_start, a_index_end + 1):
    a_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(index).zfill(3) + '/axial/f_' + str(index).zfill(2) + '_Axial_' + str(a_index).zfill(5) + '.bmp')
    if type(a_img) is np.ndarray:
        a_cropped_img = a_img[a_y: a_y + a_h, a_x: a_x + a_w]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/23_24_spine/Axial/f_' + str(index).zfill(3) + '/f_' + str(index).zfill(3) + '_' + str(a_index).zfill(5) + '_Axial_cropped.bmp', a_cropped_img)

        a_xray += a_cropped_img

c_xray = c_xray / (c_index_end - c_index_start + 1)
s_xray = s_xray / (s_index_end - s_index_start + 1)
a_xray = a_xray / (a_index_end - a_index_start + 1)

cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray/23_24_spine/Coronal/f_' + str(index).zfill(3) + '_Coronal_cropped_xray.bmp', c_xray)
cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray/23_24_spine/Sagittal/f_' + str(index).zfill(3) + '_Sagittal_cropped_xray.bmp', s_xray)
cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray/23_24_spine/Axial/f_' + str(index).zfill(3) + '_Axial_cropped_xray.bmp', a_xray)
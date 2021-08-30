import cv2
import numpy as np

w_index = 54
m_index = 57

c_index_start = 249
c_index_end = 341

s_index_start = 213
s_index_end = 331

a_index_start = 997
a_index_end = 1026

c_x_first = 136
c_y_first = 720

s_x_first = 116
s_y_first = 692

a_x_first = 132
a_y_first = 134

c_x_second = 100
c_y_second = 139

c_w = 76
c_h = 39

s_x_second = 134
s_y_second = 158

s_w = 68
s_h = 52

a_x_second = 80
a_y_second = 117

a_w = 120
a_h = 88

resolution = 256

c_xray = np.zeros(shape=(c_h, c_w, 3))
s_xray = np.zeros(shape=(s_h, s_w, 3))
a_xray = np.zeros(shape=(a_h, a_w, 3))

for c_index in range (c_index_start, c_index_end + 1):
    c_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/m_' + str(m_index).zfill(3) + '/coronal/m_' + str(m_index).zfill(2) + '_Coronal_' + str(c_index).zfill(5) + '.bmp')
    if type(c_img) is np.ndarray:
        c_cropped_img_first = c_img[c_y_first: c_y_first + resolution, c_x_first: c_x_first + resolution]
        c_cropped_img_second = c_cropped_img_first[c_y_second: c_y_second + c_h, c_x_second: c_x_second + c_w]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Coronal/m_' + str(m_index).zfill(3)+ '/m_' + str(m_index).zfill(3)  + '_' + str(c_index).zfill(5) + '_Coronal_cropped.bmp', c_cropped_img_second)

        c_xray += c_cropped_img_second

for s_index in range (s_index_start, s_index_end + 1):
    s_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/m_' + str(m_index).zfill(3) + '/sagittal/m_' + str(m_index).zfill(2) + '_Sagittal_' + str(s_index).zfill(5) + '.bmp')
    if type(s_img) is np.ndarray:
        s_cropped_img_first = s_img[s_y_first: s_y_first + resolution, s_x_first: s_x_first + resolution]
        s_cropped_img_second = s_cropped_img_first[s_y_second: s_y_second + s_h, s_x_second: s_x_second + s_w]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Sagittal/m_' + str(m_index).zfill(3) + '/m_' + str(m_index).zfill(3) + '_' + str(s_index).zfill(5) + '_Sagittal_cropped.bmp', s_cropped_img_second)

        s_xray += s_cropped_img_second

for a_index in range (a_index_start, a_index_end + 1):
    a_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/m_' + str(m_index).zfill(3) + '/axial/m_' + str(m_index).zfill(2) + '_Axial_' + str(a_index).zfill(5) + '.bmp')
    if type(a_img) is np.ndarray:
        a_cropped_img_first = a_img[a_y_first: a_y_first + resolution, a_x_first: a_x_first + resolution]
        a_cropped_img_second = a_cropped_img_first[a_y_second: a_y_second + a_h, a_x_second: a_x_second + a_w]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Axial/m_' + str(m_index).zfill(3) + '/m_' + str(m_index).zfill(3) + '_' + str(a_index).zfill(5) + '_Axial_cropped.bmp', a_cropped_img_second)

        a_xray += a_cropped_img_second

c_xray = c_xray / (c_index_end - c_index_start + 1)
s_xray = s_xray / (s_index_end - s_index_start + 1)
a_xray = a_xray / (a_index_end - a_index_start + 1)

cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray/Coronal/m_' + str(m_index).zfill(3) + '_Coronal_cropped_xray.bmp', c_xray)
cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray/Sagittal/m_' + str(m_index).zfill(3) + '_Sagittal_cropped_xray.bmp', s_xray)
cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray/Axial/m_' + str(m_index).zfill(3) + '_Axial_cropped_xray.bmp', a_xray)
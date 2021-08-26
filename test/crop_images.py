import cv2
import numpy as np

c_index = 256
s_index = 254
a_index = 1000

c_x = 136
c_y = 690

s_x = 116
s_y = 692

a_x = 132
a_y = 134

resolution = 256

# Woman (0 ~ 19)

for i in range (1, 20):
    c_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/coronal/f_' + str(i).zfill(3) + '_Coronal_' + str(c_index).zfill(5) + '.bmp')
    if type(c_img) is np.ndarray:
        c_cropped_img = c_img[c_y: c_y + resolution, c_x: c_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Coronal/f_' + str(i).zfill(3) + '_Coronal_cropped.bmp', c_cropped_img)

    s_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/sagittal/f_' + str(i).zfill(3) + '_Sagittal_' + str(s_index).zfill(5) + '.bmp')
    if type(s_img) is np.ndarray:
        s_cropped_img = s_img[s_y: s_y + resolution, s_x: s_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Sagittal/f_' + str(i).zfill(3) + '_Sagittal_cropped.bmp', s_cropped_img)

    a_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/axial/f_' + str(i).zfill(3) + '_Axial_' + str(a_index).zfill(5) + '.bmp')
    if type(a_img) is np.ndarray:
        a_cropped_img = a_img[a_y: a_y + resolution, a_x: a_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Axial/f_' + str(i).zfill(3) + '_Axial_cropped.bmp', a_cropped_img)


# Woman (20 ~ 55)

for i in range (20, 56):
    c_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/coronal/f_' + str(i).zfill(2) + '_Coronal_' + str(c_index).zfill(5) + '.bmp')
    if type(c_img) is np.ndarray:
        c_cropped_img = c_img[c_y: c_y + resolution, c_x: c_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Coronal/f_' + str(i).zfill(3) + '_Coronal_cropped.bmp', c_cropped_img)

    s_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/sagittal/f_' + str(i).zfill(2) + '_Sagittal_' + str(s_index).zfill(5) + '.bmp')
    if type(s_img) is np.ndarray:
        s_cropped_img = s_img[s_y: s_y + resolution, s_x: s_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Sagittal/f_' + str(i).zfill(3) + '_Sagittal_cropped.bmp', s_cropped_img)

    a_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/axial/f_' + str(i).zfill(2) + '_Axial_' + str(a_index).zfill(5) + '.bmp')
    if type(a_img) is np.ndarray:
        a_cropped_img = a_img[a_y: a_y + resolution, a_x: a_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Axial/f_' + str(i).zfill(3) + '_Axial_cropped.bmp', a_cropped_img)


for i in range (1, 58):
    c_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/coronal/m_' + str(i).zfill(2) + '_Coronal_' + str(c_index).zfill(5) + '.bmp')
    if type(c_img) is np.ndarray:
        c_cropped_img = c_img[c_y: c_y + resolution, c_x: c_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Coronal/m_' + str(i).zfill(3) + '_Coronal_cropped.bmp', c_cropped_img)

    s_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/sagittal/m_' + str(i).zfill(2) + '_Sagittal_' + str(s_index).zfill(5) + '.bmp')
    if type(s_img) is np.ndarray:
        s_cropped_img = s_img[s_y: s_y + resolution, s_x: s_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Sagittal/m_' + str(i).zfill(3) + '_Sagittal_cropped.bmp', s_cropped_img)

    a_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/axial/m_' + str(i).zfill(2) + '_Axial_' + str(a_index).zfill(5) + '.bmp')
    if type(a_img) is np.ndarray:
        a_cropped_img = a_img[a_y: a_y + resolution, a_x: a_x + resolution]
        cv2.imwrite('I:/DK_Data_Process/i_1-2_Slices_Cropped/Axial/m_' + str(i).zfill(3) + '_Axial_cropped.bmp', a_cropped_img)
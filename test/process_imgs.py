import numpy as np
import cv2

final_res = 137

result_img = np.zeros([final_res,final_res,3], dtype=np.uint8)

src_img = cv2.imread('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop/Axial/f_001_Axial_2nd_cropped.jpg', cv2.IMREAD_COLOR)
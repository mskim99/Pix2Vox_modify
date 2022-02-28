import glob
import shutil

for i in range(1, 58):
    image_paths = glob.glob('J:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/23_24_spine/Axial/m_' + str(i).zfill(3) + '/*.bmp')
    image_num = len(image_paths)
    target_num1 = round(image_num * 0.25)
    target_num2 = round(image_num * 0.75)
    # print (str(target_num1) + ' ' + str(target_num2) + ' ' + str(image_num))
    if (image_num > 0):
        shutil.copy(image_paths[target_num1], 'J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/00.png')
        shutil.copy(image_paths[target_num1], 'J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/01.png')

    image_paths = glob.glob('J:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/23_24_spine/Coronal/m_' + str(i).zfill(3) + '/*.bmp')
    image_num = len(image_paths)
    target_num1 = round(image_num * 0.25)
    target_num2 = round(image_num * 0.75)
    # print (str(target_num1) + ' ' + str(target_num2) + ' ' + str(image_num))
    if (image_num > 0):
        shutil.copy(image_paths[target_num1], 'J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/03.png')
        shutil.copy(image_paths[target_num1], 'J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/04.png')

    image_paths = glob.glob('J:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/23_24_spine/Sagittal/m_' + str(i).zfill(3) + '/*.bmp')
    image_num = len(image_paths)
    target_num1 = round(image_num * 0.25)
    target_num2 = round(image_num * 0.75)
    # print (str(target_num1) + ' ' + str(target_num2) + ' ' + str(image_num))
    if (image_num > 0):
        shutil.copy(image_paths[target_num1], 'J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/05.png')
        shutil.copy(image_paths[target_num1], 'J:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_23_24/m_' + str(i).zfill(3) + '/06.png')
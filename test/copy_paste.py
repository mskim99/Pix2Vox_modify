import shutil

# Image
'''
for i in range (1, 58):
    # Woman
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Axial/f_' + str(i).zfill(3) + '_Axial_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Coronal/f_' + str(i).zfill(3) + '_Coronal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Sagittal/f_' + str(i).zfill(3) + '_Sagittal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/02.png')
    
    # Man
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Axial/m_' + str(i).zfill(3) + '_Axial_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Coronal/m_' + str(i).zfill(3) + '_Coronal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped/res_128_2nd_crop_png/Sagittal/m_' + str(i).zfill(3) + '_Sagittal_2nd_cropped.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/02.png')
'''

for i in range (49, 58):
    shutil.copy('I:/DK_Data_Process/i_1-3_Target_Mesh/24_spine/res_32/m_' + str(i).zfill(3) + '_vrt_24_res_32.binvox',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Vox32/00000024/m_' + str(i).zfill(7) + '/model.binvox')


# Rendering Metadata
'''
for i in range (1, 58):
    shutil.copy('I:/rendering_metadata.txt',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/rendering_metadata.txt')
                '''

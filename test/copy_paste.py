import shutil

# Image

for i in range (49, 58):
    '''
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

    # Woman
    '''
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Axial/f_' + str(i).zfill(3) + '_Axial_cropped_xray.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Coronal/f_' + str(i).zfill(3) + '_Coronal_cropped_xray.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Sagittal/f_' + str(i).zfill(3) + '_Sagittal_cropped_xray.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/02.png')
    '''
    # Man
    '''
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Axial/m_' + str(i).zfill(3) + '_Axial_cropped_xray.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/00.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Coronal/m_' + str(i).zfill(3) + '_Coronal_cropped_xray.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/01.png')
    shutil.copy('I:/DK_Data_Process/i_1-2_Slices_Cropped_Xray_png/23_24_spine/Sagittal/m_' + str(i).zfill(3) + '_Sagittal_cropped_xray.png',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/02.png')
    '''
'''
for i in range (49, 58):
    shutil.copy('I:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_64/m_' + str(i).zfill(3) + '_vrt_23_24_res_64.binvox',
                'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Vox64/00000024/m_' + str(i).zfill(7) + '/model.binvox')
                '''

# Rendering Metadata
for i in range (24, 26):
    shutil.copy('I:/renderings.txt', 'I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KIST_xray_23_24_spine/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/renderings.txt')


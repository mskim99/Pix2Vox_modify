import os
'''
for i in range (1, 58):
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3))
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3))

    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/coronal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/sagittal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/axial')

    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/coronal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/sagittal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/axial')
'''

for i in range (1, 58):
    os.mkdir('I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_xray/KISTI_Rendering/00000024/m_' + str(i).zfill(7))
    os.mkdir('I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_xray/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering')
    os.mkdir('I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_xray/KISTI_Rendering/00000024/f_' + str(i).zfill(7))
    os.mkdir('I:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_xray/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering')

'''
for i in range (1, 56):
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Axial/f_' + str(i).zfill(3))
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Coronal/f_' + str(i).zfill(3))
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Sagittal/f_' + str(i).zfill(3))

for i in range (1, 58):
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Axial/m_' + str(i).zfill(3))
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Coronal/m_' + str(i).zfill(3))
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices_Cropped_Volume/Sagittal/m_' + str(i).zfill(3))
'''
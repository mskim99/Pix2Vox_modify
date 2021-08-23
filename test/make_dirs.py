import os

for i in range (1, 58):
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3))
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3))

    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/coronal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/sagittal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/m_' + str(i).zfill(3) + '/axial')

    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/coronal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/sagittal')
    os.mkdir('I:/DK_Data_Process/i_1-2_Slices/f_' + str(i).zfill(3) + '/axial')
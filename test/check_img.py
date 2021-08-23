import cv2
import numpy as np

save_path = 'I:/Program/Pix2Vox-master/Pix2Vox-master/output/image/test/voxels-' + str(1).zfill(6) + '.png'
print(save_path)
img = cv2.imread(save_path)
im = np.array((img * 255), dtype=np.uint8)
cv2.imshow('result', im)

cv2.waitKey(0)
cv2.destroyAllWindows()
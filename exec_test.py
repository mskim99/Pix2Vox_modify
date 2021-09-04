import os

for i in range (10, 510, 10):
    command = "python runner.py --gpu 0 --test --weights=D:/output/logs/checkpoints/ckpt-epoch-" + str(i).zfill(4) + ".pth"
    os.system(command)
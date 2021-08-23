import numpy as np
import cv2
import os
import binvox_rw
import argparse

resolution = 128

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="file name")
FLAGS = parser.parse_args()

name = FLAGS.name
target_dir = "./"

this_name = target_dir + name + ".obj"
print(this_name)

maxx = 0.5
maxy = 0.5
maxz = 0.5
minx = -0.5
miny = -0.5
minz = -0.5

command = "binvox -bb "+str(minx)+" "+str(miny)+" "+str(minz)+" "+str(maxx)+" "+str(maxy)+" "+str(maxz)+" "+" -d " + str(resolution) + " -e "+this_name

os.system(command)


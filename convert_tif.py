import sys
import numpy as np 
from PIL import Image
import os


#use like this python convert_tif.py data/training.tif data/training 1

filename = sys.argv[1]
output_folder=sys.argv[2]+"/"
scale=1
if len(sys.argv)>3:
  scale = int(sys.argv[3])

extension=filename.split('.')[1]

name=filename.split('.')[0].split('/')[-1]

img = Image.open(filename)
width, height = img.size
size= (width/scale, height/scale)

i=0
while True:
  if (i%scale==0):
    try:
      img.seek(i)
      img.thumbnail(size, Image.ANTIALIAS)
      img.save(output_folder+"/"+name+"_"+str(i/scale)+".jpg","JPEG")
    except EOFError:
      break
  i+=1
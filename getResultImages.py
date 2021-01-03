import os 
import glob 
import shutil 

path = "cases"
folder = os.path.join(path,"results")
os.mkdir(folder)




images = glob.glob(os.path.join(path,"*/experiments/*/*/y.png"))
newImages = [im.replace('/','-') for im in images]
for i, im in enumerate(images):
    shutil.copy(im,os.path.join(folder,newImages[i]))

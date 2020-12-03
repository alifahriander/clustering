import os 
import glob 
import shutil 

path = "experiments"
folder = os.path.join(path,"results")
os.mkdir(folder)

images = glob.glob(os.path.join(path,"*/*/result.png"))
newImages = [im.replace('/','-') for im in images]
for i, im in enumerate(images):
    shutil.copy(im,os.path.join(folder,newImages[i]))

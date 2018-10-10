import os,shutil,sys
from split import SPLIT



shutil.rmtree("train")

os.makedirs("train/Annotations")
os.makedirs("train/ImageSets/Main")
os.makedirs("train/JPEGImages")

names=[]
jpegs=[]
xmls=[]
with open("list.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        name=line.strip()
        names.append(name)
        jpegs.append(name+".jpg")
        xmls.append(name+".xml")

# copy train images and xmls
f=open("train/ImageSets/Main/list.txt","w")
for i in range(0,SPLIT[0]):
    print("train ",names[i])
    img=jpegs[i]
    xml=xmls[i]
    shutil.copy("all/JPEGImages/"+img, "train/JPEGImages/")
    shutil.copy("all/Annotations/"+xml, "train/Annotations/")
    f.write(names[i]+"\n")
f.close()

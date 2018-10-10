import os,shutil,sys
from split import SPLIT


shutil.rmtree("val")

os.makedirs("val/Annotations")
os.makedirs("val/ImageSets/Main")
os.makedirs("val/JPEGImages")


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


# copy val images and xmls
f=open("val/ImageSets/Main/list.txt","w")

for i in range(SPLIT[0],SPLIT[1]):
    print("val ",names[i])
    img=jpegs[i]
    xml=xmls[i]
    shutil.copy("all/JPEGImages/"+img, "val/JPEGImages/")
    shutil.copy("all/Annotations/"+xml, "val/Annotations/")
    f.write(names[i]+"\n")
f.close()


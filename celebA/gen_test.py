import os,shutil,sys
from split import SPLIT



shutil.rmtree("test")

os.makedirs("test/Annotations")
os.makedirs("test/ImageSets/Main")
os.makedirs("test/JPEGImages")

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



# copy test images and xmls
f=open("test/ImageSets/Main/list.txt","w")
for i in range(SPLIT[1],SPLIT[2]):
    print("test ",names[i])
    img=jpegs[i]
    xml=xmls[i]
    shutil.copy("all/JPEGImages/"+img, "test/JPEGImages/")
    shutil.copy("all/Annotations/"+xml, "test/Annotations/")
    f.write(names[i]+"\n")
f.close()



















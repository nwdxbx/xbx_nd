import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

xmls_dir = '/data_b/mount/bb4/celebA/all/Annotations'
labels_dir = '/data_b/mount/bb4/celebA/all/labels'
list_txt = '/data_b/mount/bb4/celebA/all/ImageSets/Main/train.txt'
rootdir = '/data_b/mount/bb4/celebA/all'

classes = ["face"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open(os.path.join(xmls_dir,'%s.xml'%(image_id)))
    out_file = open(os.path.join(labels_dir,'%s.txt'%(image_id)),'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def showgt():
    image_ids = open(list_txt).read().strip().split()
    for image_id in image_ids:
        convert_annotation(image_id)

def generatetxt(trainratio=0.7,valratio=0.2,testratio=0.1):  
    files=os.listdir(labels_dir)  
    ftrain=open(rootdir+"/"+"train.txt","w")  
    fval=open(rootdir+"/"+"val.txt","w")  
    ftrainval=open(rootdir+"/"+"trainval.txt","w")  
    ftest=open(rootdir+"/"+"test.txt","w")  
    for i in range(len(files)):  
        filename=files[i]  
        filename='/data3/xbx/celebA/all' + "/JPEGImages/" +filename[:-3]+"jpg"+"\n"  
        if i<trainratio*len(files):  
            ftrain.write(filename)  
            ftrainval.write(filename)  
        elif i<(trainratio+valratio)*len(files):  
            fval.write(filename)  
            ftrainval.write(filename)  
        elif i<(trainratio+valratio+testratio)*len(files):  
            ftest.write(filename)  
    ftrain.close()  
    fval.close()  
    ftrainval.close()  
    ftest.close()
if __name__ == "__main__":
    # showgt()
    generatetxt()
    print ('finish...')


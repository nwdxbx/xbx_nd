import xml.etree.ElementTree as ET
from os import getcwd

classes = ["face",]
def convert_annotation(in_file, list_file):
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

list_file = open('label.txt','w')
with open('train.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        in_file = line.replace('/JPEGImages/','/Annotations/')
        in_file = in_file.replace('.jpg','.xml')
        in_file = in_file.replace('.png','.xml')
        list_file.write(line)
        convert_annotation(in_file,list_file)
        list_file.write('\n')
    
    list_file.close()

import xml.etree.ElementTree as ET
from glob import glob
import codecs

classes = ["face"]

def convert_annotation(anno_path):
    in_file = open(anno_path)
    tree=ET.parse(in_file)
    root = tree.getroot()
    label = ''
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))

        label = label + (" " + ",".join([str(a) for a in b]) + ',' + str(0))
    return label


#anno_paths = glob('/data/celebA_ori/all/Annotations/*.xml')
#list_file = codecs.open('celebA_face','w')

#anno_paths = glob('/data/face_datasets/livingphoto_8000/*.lif')
anno_paths = glob('.//Annotations/*.xml')
list_file = codecs.open('celebA_good_box','a+')

for anno_path in anno_paths:
        label = convert_annotation(anno_path)
        if label == '':
            continue
        idx = anno_path.split("/")[-1].split('.xml')[0]
        list_file.write('/data/face_datasets/celebA_ori/all/JPEGImages/'+idx+'.jpg')
        #list_file.write('/data/celebA_ori/all/JPEGImages/'+idx+'.jpg')
        list_file.write(label)
        list_file.write('\n')
        print(anno_path)
list_file.close()

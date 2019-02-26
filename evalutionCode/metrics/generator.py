import os
import numpy as np

result_dir = "./predicted/"
groundtrue_dir = "./ground-truth"

def ellipse_to_rect(ellipse):
    major_axis_radius, minor_axis_radius, angle, center_x, center_y, score = ellipse
    leftx = center_x - minor_axis_radius
    topy = center_y - major_axis_radius
    width = 2 * minor_axis_radius
    height = 2 * major_axis_radius
    rect = [leftx, topy, width, height, score]
    return rect

def load(txtfile,root_dir):
    with open(txtfile,"r") as f:
        lines = f.readlines()
        i = 0
        
        while i < len(lines):
            imagename = os.path.basename(lines[i].strip())+".txt"
            print(imagename)
            absname = os.path.join(root_dir,imagename)
            txtname = os.path.basename(lines[i].strip())
            j = 0
            while os.path.exists(absname):
                j = j+1
                tmpname = txtname + "_" + str(j) + ".txt"
                absname = os.path.join(root_dir,tmpname)
            num_object = int(lines[i + 1])
            with open(absname,'w') as wf:
                for num in range(num_object):
                    boundingbox = lines[i + 2 + num].strip()
                    boundingbox = boundingbox.split()
                    boundingbox = list(map(float, boundingbox))
                    if len(boundingbox) == 6:
                        boundingbox = ellipse_to_rect(boundingbox)
                        boundingbox = list(map(str,boundingbox))
                        result = boundingbox[0] + " " + boundingbox[1] + " " + boundingbox[2] + " " +\
                                    boundingbox[3] + " " + boundingbox[4] + "\n"
                        wf.writelines(result)
                    else:
                        wf.writelines(lines[i+2+num])
            i = i + num_object + 2



def generator(resultsfile,groundtruthfile):
    load(resultsfile,result_dir)
    load(groundtruthfile,groundtrue_dir)

if __name__ == "__main__":
    generator("../curves_for_object_detection/results.txt","../curves_for_object_detection/ellipseList.txt")
import os
import sys
import cv2
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt

sys.path.append("..")
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

import dlib
from skimage import io

bfm = MorphabelModel('Data/BFM/Out/BFM.mat')


predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
def detect_landmark(imgpath):
    img = cv2.imread(imgpath)
    dlib_img = io.imread(imgpath)
    dets = detector(img,1)
    if len(dets) == 0:
        return 1,None
    x1 = dets[0].left()
    y1 = dets[0].top()
    x2 = dets[0].right()
    y2 = dets[0].bottom()
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

    shape = predictor(dlib_img,dets[0])
    nLM = shape.num_parts
    landmarks = []
    for i in range(0,nLM):
        cv2.circle(img,(shape.part(i).x,shape.part(i).y),5,(255,0,0))
        landmarks.append([shape.part(i).x,shape.part(i).y])
    import pdb
    pdb.set_trace()
    landmarks = np.array(landmarks)
    landmarks = landmarks.astype(np.float32)
    return 0,landmarks

def dlib_landmark(imgpath):
    im = cv2.imread(imgpath)
    h,w,c = im.shape
    rects = detector(im,1)
    shape = predictor(im,rects[0])
    rects = [(rects[0].tl_corner().x, rects[0].tl_corner().y), (rects[0].br_corner().x, rects[0].br_corner().y)]
    landmarks = np.zeros((68, 2))
    for i, p in enumerate(shape.parts()):
        landmarks[i] = [p.x, p.y]

    return landmarks,h,w

def get_landmark(imgpath):
    im = cv2.imread(imgpath)
    h,w,c = im.shape
    annos = imgpath.replace(".jpg",".pts")
    landmarks = []
    with open(annos,"r") as f:
        lines = f.readlines()
        for i in range(3,71):
            line = lines[i]
            line = line.strip().split()
            landmarks.append([float(line[0]),float(line[1])])
    landmarks = np.array(landmarks)
    # x = np.mean(landmarks,axis=0)
    # landmarks = landmarks - x
    return landmarks,h,w

def a_b_angle(imgpath,landmarks,h=256,w=256):
    # x = landmarks
    x = mesh.transform.from_image(landmarks, h, w)
    X_ind = bfm.kpt_ind
    tp = bfm.get_tex_para('random')
    colors = bfm.generate_colors(tp)
    colors = np.minimum(np.maximum(colors, 0), 1)
    fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 200)
    fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
    import pdb
    pdb.set_trace()
    fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
    io.imsave('fitted_{}'.format(imgpath),fitted_image)


if __name__ == "__main__":
    with open("list.txt","r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        landmarks,h,w = get_landmark(line)
        a_b_angle(line,landmarks,h,w)

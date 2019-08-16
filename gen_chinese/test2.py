import os
import cv2
import numpy as np


def Lremap():
    image = cv2.imread("/data_b/Framework/3D_landmarks/PRNet/myinput/face3.jpg")
    h,w,_ = image.shape
    imagex = np.zeros((h,w),dtype=np.float32)
    imagey = np.zeros((h,w),dtype=np.float32)
    for j in range(h):
        for i in range(w):
            if i>0.25*w and i<0.75*w and j>0.25*h and j<0.75*h:
                imagex[j,i] = 2*i-w*0.5+0.5
                imagey[j,i] = 2*j-h*0.5+0.5
            else:
                imagex[j,i] = -1
                imagex[j,i] = -1

    mapX = np.array([imagex,imagey])
    mapX = np.transpose(mapX,(1,2,0))
    x1 = int(0.25*w)
    x2 = int(0.75*w)
    y1 = int(0.25*h)
    y2 = int(0.75*h)
    mapY = mapX[y1:y2,x1:x2]
    import pdb
    pdb.set_trace()

    dst = cv2.remap(image,mapY,None,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128))
    cv2.imshow("src",image)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)


def Lfont():
    # Read images : src image will be cloned into dst
    im = cv2.imread("images/wood-texture.jpg")
    obj= cv2.imread("images/iloveyouticket.jpg")
    
    # Create an all white mask
    mask = 255 * np.ones(obj.shape, obj.dtype)
    
    # The location of the center of the src in the dst
    width, height, channels = im.shape
    center = (height/2, width/2)
    
    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
    mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
    
    # Write results
    cv2.imwrite("images/opencv-normal-clone-example.jpg", normal_clone)
    cv2.imwrite("images/opencv-mixed-clone-example.jpg", mixed_clone)


 
# Read images
src = cv2.imread("320px-Japan.airlines.b777-300.ja733j.arp.jpg")
dst = cv2.imread("indoor_103.png")
 
 
# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))
# This is where the CENTER of the airplane will be placed
center = (500,500)
 
# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
 
# Save result
cv2.imwrite("output.jpg", output)
cv2.imshow("images",output)
cv2.waitKey(0)

import os
import cv2
import math
import numpy as np

from face_detection import face_detect
import tensorflow as tf

def Limage(img):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    l = hls[...,1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # import pdb
    # pdb.set_trace()
    l_scalar = cv2.mean(l)
    hls_scalar = cv2.mean(hls)
    print("hls_scalar: ",hls_scalar)
    cv2.imshow("img",img)
    cv2.imshow("l",l)
    cv2.waitKey(0)

def image_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))

    #rg = R - G
    rg = np.absolute(R - G)

    #yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    #计算rg和yb的平均值和标准差
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    #计算rgyb的标准差和平均值
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # 返回颜色丰富度C
    return stdRoot + (0.3 * meanRoot)

def DefRto(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height,width = gray.shape
    num = height*width
    temp = 0
    for j in range(height-1):
        for i in range(width-1):
                temp = temp + math.sqrt((gray[j+1,i]-gray[j,i])*(gray[j+1,i]-gray[j,i]) + (gray[j,i+1]-gray[j,i])*(gray[j,i+1]-gray[j,i]))
                temp = temp + math.abs(gray[j+1,i]-gray[j,i]) + math.abs(gray[j,i+1]-gray[j,i])

    DR = temp/num
    return DR

def colorException(image):
    labImage = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
    histLista = [0 for i in range(256)]
    histListb = [0 for i in range(256)]
    a = 0
    b = 0
    height,width,c = labImage.shape
    for j in range(height):
        for i in range(width):
            a = a + labImage[j,i,1]-128
            b = b + labImage[j,i,2]-128
            a_value = labImage[j,i,1]
            b_value = labImage[j,i,2]
            histLista[a_value] = histLista[a_value] + 1
            histListb[b_value] = histListb[b_value] + 1
    da = 1.0*a/(height*width)
    db = 1.0*b/(height*width)
    D = math.sqrt(da*da+db*db)
    Ma = 0.0
    Mb = 0.0
    for i in range(256):
        Ma = Ma + np.abs(i-128-da)*histLista[i]
        Mb = Mb + np.abs(i-128-db)*histListb[i]
    Ma = Ma/(height*width)
    Mb = Mb/(height*width)
    M = math.sqrt(Ma*Ma+Mb*Mb)
    cast = D/M
    if cast>1:
        print("存在色偏： "，cast)
    else:
        print("不存在色偏")

def brightnessException(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    histList = [0 for i in range(256)]
    height,width = gray.shape
    mean_value = 0
    for j in range(height):
        for i in range(width):
            mean_value = mean_value + (gray[j,i]-128)
            value = gray[j,i]
            histList[value] = histList[value] + 1
    da = 1.0*mean_value/(height*width)
    D = np.abs(da)

    Ma = 0.0
    for i in range(256):
        Ma = Ma + np.abs(i-128-da)*histList[i]
    Ma = Ma/(height*width)
    M = np.abs(Ma)
    cast = D/M

    print("cast: ",cast," da: ",da)
    cv2.imshow("gray",gray)
    cv2.imshow("image",image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # lines = os.listdir("./images")
    # yl = face_detect.FaceDetection("./face_detection/yolo3.h5",0)
    # for line in lines:
    #     filename = os.path.join("./images",line)
    #     img = cv2.imread(filename)
    #     boxes = yl.detect([img])
    #     if len(boxes[0]) != 0:
    #         y1,x1,y2,x2 = map(int,boxes[0][0])
    #         image = img[y1:y2,x1:x2]
    #         Limage(image)
    lines = os.listdir("./images")
    for line in lines:
        filename = os.path.join("./images",line)
        img = cv2.imread(filename)
        brightnessException(img)
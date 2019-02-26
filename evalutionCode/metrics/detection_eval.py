import os
import numpy as np
import matplotlib.pyplot as plt

def cal(detectedbox, groundtruthbox):
    leftx_det, topy_det, width_det, height_det= detectedbox
    leftx_gt, topy_gt, width_gt, height_gt= groundtruthbox

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_det * height_det + width_gt * height_gt - intersection
        iu = intersection / union

        return iu
    else:
        return 0

def match(resultpath,groundtruthpath):
    lines = os.listdir(resultpath)
    tmplines = os.listdir(groundtruthpath)
    assert len(lines) == len(tmplines)
    num_detbox = 0
    num_grbox = 0
    max_iu_confidence = np.array([])
    for line in lines:
        absres_name = os.path.join(resultpath,line)
        absgr_name = os.path.join(groundtruthpath,line)

        with open(absres_name,'r') as resf:
            reslines = resf.readlines()
        with open(absgr_name,'r') as grf:
            grlines = grf.readlines()
        num_detbox = num_detbox + len(reslines)
        num_grbox = num_grbox + len(grlines)
        for resline in reslines:
            resbox = resline.strip().split()
            boundingbox = list(map(float,resbox))
            detbox = boundingbox[:-1]
            confidence = boundingbox[-1]
            iu_array = np.array([])

            for grline in grlines:
                grbox = grline.strip().split()
                truebox = list(map(float,grbox))[:-1]
                iu = cal(detbox,truebox)
                iu_array = np.append(iu_array,iu)
            
            value = np.max(iu_array)
            max_iu_confidence = np.append(max_iu_confidence,[value,confidence])

    max_iu_confidence = max_iu_confidence.reshape(-1,2)
    max_iu_confidence = max_iu_confidence[np.argsort(-max_iu_confidence[:,1])]

    return max_iu_confidence,num_grbox

def thresh(max_iu_confidence,threshhold = 0.5):
    ius = max_iu_confidence[:,0]
    confidences = max_iu_confidence[:,1]
    tof = (ius > threshhold)
    tf_confidence = np.array([tof,confidences])
    tf_confidence = tf_confidence.T

    return tf_confidence

def plot(tf_confidence,num_grbox):
    fps = []
    recs = []
    precisions = []
    auc = 0.0
    ap = 0.0
    tmp_tp = 0
    for num in range(len(tf_confidence)):
        arr = tf_confidence[:(num + 1), 0]
        tp = np.sum(arr)
        fp = np.sum(arr == 0)
        rec = tp/num_grbox
        precision = tp/(tp+fp)
        auc = auc + rec
        if tp-tmp_tp == 1:
            ap = ap + precision
            tmp_tp = tp
        fps.append(fp)
        recs.append(rec)
        precisions.append(precision)

    auc = auc/len(fps)
    ap = ap / num_grbox
    print("ap: ",ap)
    plt.figure()
    plt.title('ROC')
    plt.xlabel('False Positives')
    plt.ylabel('True Positive rate')
    plt.ylim(0, 1)
    plt.plot(fps, recs, label = 'AUC: ' + str(auc))
    plt.legend()

    plt.figure()
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.plot(recs, precisions, label = 'mAP: ' + str(ap))
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    max_iu_confidence,num_grbox = match("./predicted","./ground-truth")
    tfconf = thresh(max_iu_confidence)
    plot(tfconf,num_grbox)
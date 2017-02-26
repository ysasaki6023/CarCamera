import sys,os,shutil

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2,glob,sys
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

def non_max_suppression_slow(boxes, overlapThresh, scoreThresh, sameClassOnly=False):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    pick = []
    boxes = np.array(boxes)
    x1,y1,x2,y2,cls,sc = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],boxes[:,4],boxes[:,5]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(sc)
    deleted = np.array([False]*len(sc))

    for i in range(len(sc)):
        if sc[i]<scoreThresh:
            deleted[i] = True
            continue
        for j in range(len(sc)):
            if i==j                    : continue
            if deleted[i] or deleted[j]: continue
            xx1, yy1 = max(x1[i], x1[j]) , max(y1[i], y1[j])
            xx2, yy2 = min(x2[i], x2[j]) , min(y2[i], y2[j])
            w,h      = max(0, xx2 - xx1 + 1), max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / min(area[i],area[j])
            if overlap < overlapThresh : continue
            if sameClassOnly and (not cls[i] == cls[j])    : continue
            if sc[i] > sc[j]: deleted[j] = True
            else            : deleted[i] = True
    return boxes[np.logical_not(deleted)]


def process_predicts(predicts):
    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]

    p_classes = np.reshape(p_classes, (7, 7, 1, 20))
    C = np.reshape(C, (7, 7, 2, 1))
    P = C * p_classes

    def calc(k_index,k_P,k_coordinate):
        class_num = k_index[3]
        k_coordinate = np.reshape(k_coordinate, (7, 7, 2, 4))
        max_coordinate = k_coordinate[index[0], index[1], index[2], :]
        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (k_index[1] + xcenter) * (448/7.0)
        ycenter = (k_index[0] + ycenter) * (448/7.0)

        w = w * 448
        h = h * 448

        xmin = xcenter - w/2.0
        ymin = ycenter - h/2.0

        xmax = xmin + w
        ymax = ymin + h

        score = k_P[k_index]

        return xmin, ymin, xmax, ymax, class_num, score

    res = []
    for i in range(10):
        index = np.argsort(P.ravel())[-i-1]
        index = np.unravel_index(index, P.shape)
        res.append(calc(index,P,coordinate))
    return res


def Run(iFiles, oFiles):
    batchNum = 10
    common_params = {'image_size': 448, 'num_classes': 20, 'batch_size':batchNum}
    net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

    net = YoloTinyNet(common_params, net_params, test=True)

    image = tf.placeholder(tf.float32, (batchNum, 448, 448, 3))
    predicts = net.inference(image)

    sess = tf.Session()

    saver = tf.train.Saver(net.trainable_collection)
    #saver.restore(sess,'models/pretrain/yolo_tiny.ckpt')
    saver.restore(sess,'models/train_30penalty2000Iter/model.ckpt-2000')
    #saver.restore(sess,'models/train/model.ckpt-5000')
    #saver.restore(sess,'models/train/model.ckpt-1000')
    #saver.restore(sess,'models/train/model.ckpt-5')

    currentPos = 0

    while currentPos<len(iFiles):
        print(currentPos,len(iFiles))
        # Build batch file names
        batch_iFiles = []
        batch_oFiles = []
        if currentPos+batchNum<len(iFiles):
            batch_iFiles = iFiles[currentPos:currentPos+batchNum]
            batch_oFiles = oFiles[currentPos:currentPos+batchNum]
            num_files    = batchNum
        else:
            batch_iFiles = iFiles[currentPos:]
            batch_oFiles = oFiles[currentPos:]
            num_files    = len(iFiles)-currentPos

        resized_imgs = np.zeros((batchNum,448,448,3),dtype=np.float)
        np_imgs      = np.zeros((batchNum,448,448,3),dtype=np.float)
        for fIndex, fName in enumerate(batch_iFiles):
            np_img = cv2.imread(fName)
            h,w,_ = np_img.shape
            size = min(h,w)
            np_img = np_img[int(h/2-size/2):int(h/2+size/2),int(w/2-size/2):int(w/2+size/2)]

            resized_img  = cv2.resize(np_img, (448, 448))
            np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            np_img = np_img.astype(np.float32)

            np_img = np_img / 255.0 * 2 - 1
            np_img = np.reshape(np_img, (1, 448, 448, 3))
            np_imgs[fIndex] = np_img
            resized_imgs[fIndex] = resized_img

        np_predict_list = sess.run(predicts, feed_dict={image: np_imgs})

        for index in range(num_files):
            np_predict = np.expand_dims(np_predict_list[index],0)
            resList = process_predicts(np_predict)
            #resList = np.expand_dims(resList[0],0)
            resList = non_max_suppression_slow(resList, overlapThresh=0.5, scoreThresh=0.05, sameClassOnly=False)

            resized_img = resized_imgs[index]
            for iRes in resList:
                xmin, ymin, xmax, ymax, class_num, score = iRes
                class_name = classes_name[int(class_num)]
                cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
                cv2.putText(resized_img, "%s: %.3f"%(class_name, score), (int(xmin), int(ymin)), 2, 0.5, (0, 0, 255))
            cv2.imwrite(oFiles[currentPos+index], resized_img)
            #print(oFiles[currentPos+index])

        currentPos += batchNum

    sess.close()

def RunVideo(index):
    iFolder = "videos/video%d"%index
    oFolder = "%s_out"%iFolder
    if os.path.exists(oFolder):
        shutil.rmtree(oFolder)
    os.mkdir(oFolder)
    iFiles = glob.glob("%s/frame*.jpg"%iFolder)
    oFiles = [os.path.basename(x) for x in iFiles]
    oFiles = [x.replace("frame","").replace(".jpg","") for x in oFiles]
    oFiles = ["frame%06d.jpg"%int(x)   for x in oFiles]
    oFiles = ["%s/%s"%(oFolder,x) for x in oFiles]
    Run(iFiles, oFiles)

def RunVVC():
    iFolder = "JPEGImages"
    oFolder = "%s_out"%iFolder
    if os.path.exists(oFolder):
        shutil.rmtree(oFolder)
    os.mkdir(oFolder)
    iFiles = glob.glob("%s/*.jpg"%iFolder)
    oFiles = [os.path.basename(x) for x in iFiles]
    oFiles = ["%s/%s"%(oFolder,x) for x in oFiles]
    Run(iFiles, oFiles)

if __name__=="__main__":
    #RunVideo1()
    RunVideo(7)
    #RunVVC()

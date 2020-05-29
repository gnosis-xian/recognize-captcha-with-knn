import glob
import sys
import os
import time
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw,ImageFont

#train文件夹存放训练数据集
TRAIN_DIR = "train"
#test文件夹存放测试数据集
TEST_DIR = "test"

# convert contours to boxes
# each box is a rectangle consisting of 4 points
# if there is connected characters, split the contour
#定义函数get_rect_box，目的在于获得切割图片字符位置和宽度
def get_rect_box(contours):
    #定义ws和valid_contours数组，用来存放图片宽度和训练数据集中的图片。如果分割错误的话需要重新分割

    ws = []
    valid_contours = []
    for contour in contours:
        #画矩形用来框住单个字符，x,y,w,h四个参数分别是该框子的x,y坐标和长宽。因
        x, y, w, h = cv2.boundingRect(contour)
        if w < 7:
            continue
        valid_contours.append(contour)
        ws.append(w)
#w_min是二值化白色区域最小宽度，目的用来分割。
    w_min = min(ws)
# w_max是最大宽度
    w_max = max(ws)

    result = []
    #如果切割出有4个字符。说明没啥问题
    if len(valid_contours) == 4:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
            result.append(box)
    # 如果切割出有3个字符。参照文章，中间分割
    elif len(valid_contours) == 3:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == w_max:
                box_left = np.int0([[x,y], [x+w/2,y], [x+w/2,y+h], [x,y+h]])
                box_right = np.int0([[x+w/2,y], [x+w,y], [x+w,y+h], [x+w/2,y+h]])
                result.append(box_left)
                result.append(box_right)
            else:
                box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
                result.append(box)
    # 如果切割出有3个字符。参照文章，将包含了3个字符的轮廓在水平方向上三等分
    elif len(valid_contours) == 2:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == w_max and w_max >= w_min * 2:
                box_left = np.int0([[x,y], [x+w/3,y], [x+w/3,y+h], [x,y+h]])
                box_mid = np.int0([[x+w/3,y], [x+w*2/3,y], [x+w*2/3,y+h], [x+w/3,y+h]])
                box_right = np.int0([[x+w*2/3,y], [x+w,y], [x+w,y+h], [x+w*2/3,y+h]])
                result.append(box_left)
                result.append(box_mid)
                result.append(box_right)
            elif w_max < w_min * 2:
                box_left = np.int0([[x,y], [x+w/2,y], [x+w/2,y+h], [x,y+h]])
                box_right = np.int0([[x+w/2,y], [x+w,y], [x+w,y+h], [x+w/2,y+h]])
                result.append(box_left)
                result.append(box_right)
            else:
                box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
                result.append(box)
    # 如果切割出有3个字符。参照文章，对轮廓在水平方向上做4等分
    elif len(valid_contours) == 1:
        contour = valid_contours[0]
        x, y, w, h = cv2.boundingRect(contour)
        box0 = np.int0([[x,y], [x+w/4,y], [x+w/4,y+h], [x,y+h]])
        box1 = np.int0([[x+w/4,y], [x+w*2/4,y], [x+w*2/4,y+h], [x+w/4,y+h]])
        box2 = np.int0([[x+w*2/4,y], [x+w*3/4,y], [x+w*3/4,y+h], [x+w*2/4,y+h]])
        box3 = np.int0([[x+w*3/4,y], [x+w,y], [x+w,y+h], [x+w*3/4,y+h]])
        result.extend([box0, box1, box2, box3])
    elif len(valid_contours) > 4:
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
            result.append(box)
    result = sorted(result, key=lambda x: x[0][0])
    return result

# process image including denoise, thresholding
#该行数用来处理图片数据集。二值化和降噪
def process_im(im):
    rows, cols, ch = im.shape
    #转为灰度图
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #二值化，就是黑白图。字符变成白色的，背景为黑色
    ret, im_inv = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY_INV)
    #应用高斯模糊对图片进行降噪。高斯模糊的本质是用高斯核和图像做卷积。就是去除一些斑斑点点的。因为二值化难免不够完美，去燥使得二值化结果更好
    kernel = 1/16*np.array([[1,2,1], [2,4,2], [1,2,1]])
    im_blur = cv2.filter2D(im_inv,-1,kernel)
    #再进行一次二值化。
    ret, im_res = cv2.threshold(im_blur,127,255,cv2.THRESH_BINARY)
    return im_res

# split captcha code into single characters
#借助第一个函数获得待切割位置和长宽后就可以切割了
def split_code(filepath):
    #获取图片名
    filename = filepath.split("/")[-1]
    #图片名即为标签
    filename_ts = filename.split(".")[0]
    im = cv2.imread(filepath)
    im_res = process_im(im)

    im2, contours, hierarchy = cv2.findContours(im_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#这里就是用的第一个函数，获得待切割位置和长宽
    boxes = get_rect_box(contours)
#如果没有区分出四个字符，就不切割这个图片
    if len(boxes) != 4:
        print(filepath)
# 如果区分出了四个字符，说明切割正确，就可以切割这个图片。将切割后的图片保存在char文件夹下
    for box in boxes:
        cv2.drawContours(im, [box], 0, (0,0,255),2)
        roi = im_res[box[0][1]:box[3][1], box[0][0]:box[1][0]]
        roistd = cv2.resize(roi, (30, 30))
        timestamp = int(time.time() * 1e6)
        filename = "{}.jpg".format(timestamp)
        filepath = os.path.join("char", filename)
        cv2.imwrite(filepath, roistd)

    #cv2.imshow("image", im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# split all captacha codes in training set
#调用上面的split_code进行切割即可。
def split_all():
    files = os.listdir(TRAIN_DIR)
    for filename in files:
        filename_ts = filename.split(".")[0]
        patt = "label/{}_*".format(filename_ts)
        saved_chars = glob.glob(patt)
        if len(saved_chars) == 4:
            print("{} done".format(filepath))
            continue
        filepath = os.path.join(TRAIN_DIR, filename)
        split_code(filepath)

# label data in training set
# input character manually for each image
#用来标注单个字符图片，在label文件夹下，很明显可以看到_后面的就是标签。比如图片里是数字6，_后面就是6
def label_data():
    files = os.listdir("char")
    for filename in files:
        filename_ts = filename.split(".")[0]
        patt = "label/{}_*".format(filename_ts)
        saved_num = len(glob.glob(patt))
        if saved_num == 1:
            print("{} done".format(patt))
            continue
        filepath = os.path.join("char", filename)
        im = cv2.imread(filepath)
        cv2.imshow("image", im)
        key = cv2.waitKey(0)
        if key == 27:
            sys.exit()
        if key == 13:
            continue
        char = chr(key)
        filename_ts = filename.split(".")[0]
        outfile = "{}_{}.jpg".format(filename_ts, char)
        outpath = os.path.join("label", outfile)
        cv2.imwrite(outpath, im)
#和标注字符图反过来，我们需要让电脑知道这个字符叫啥名字，即让电脑知道_后面的就是他字符的名字
def analyze_label():
    files = os.listdir("label")
    label_count = {}
    for filename in files:
        label = filename.split(".")[0].split("_")[1]
        label_count.setdefault(label, 0)
        label_count[label] += 1
    print(label_count)

# load all data in training set
#读取训练数据集和标签，准备开始训练。训练前必须读入数据集
def load_data():
    filenames = os.listdir("label")
    samples = np.empty((0, 900))
    labels = []
    for filename in filenames:
        filepath = os.path.join("label", filename)
        label = filename.split(".")[0].split("_")[-1]
        labels.append(label)
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        sample = im.reshape((1, 900)).astype(np.float32)
        samples = np.append(samples, sample, 0)
    samples = samples.astype(np.float32)
    unique_labels = list(set(labels))
    unique_ids = list(range(len(unique_labels)))
    label_id_map = dict(zip(unique_labels, unique_ids))
    id_label_map = dict(zip(unique_ids, unique_labels))
    label_ids = list(map(lambda x: label_id_map[x], labels))
    label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)
    return [samples, label_ids, id_label_map]

# identify captcha image
#训练模型，用的是k相邻算法
def get_code(im):
    #将读取图片和标签
    [samples, label_ids, id_label_map] = load_data()
    #k相邻算法
    model = cv2.ml.KNearest_create()
    #开始训练
    model.train(samples, cv2.ml.ROW_SAMPLE, label_ids)
    #处理图片。即二值化和降噪
    im_res = process_im(im)
    #提取轮廓
    im2, contours, hierarchy = cv2.findContours(im_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #获取各切割区域位置和长宽
    boxes = get_rect_box(contours)
    #判断有没有识别出4个字符，如果没有识别出来，就不往下运行，直接结束了
    if len(boxes) != 4:
        print("cannot get code")

    result = []
    #如果正确分割出了4个字符，下面调用训练好的模型进行识别。
    for box in boxes:
        #获取字符长宽
        roi = im_res[box[0][1]:box[3][1], box[0][0]:box[1][0]]
        #重新设长宽。
        roistd = cv2.resize(roi, (30, 30))
        #将图片转成像素矩阵
        sample = roistd.reshape((1, 900)).astype(np.float32)
        #调用训练好的模型识别
        ret, results, neighbours, distances = model.findNearest(sample, k = 3)
        #获取对应标签id
        label_id = int(results[0,0])
        #根据id得到识别出的结果
        label = id_label_map[label_id]
        #存放识别结果
        result.append(label)
    return result

# identify captcha image in test set
#测试看看模型准确率和调用模型师识别
def test_data():
    test_files = os.listdir("test")
    total = 0
    correct = 0
    for filename in test_files:
        filepath = os.path.join("test", filename)
        im = cv2.imread(filepath)
        preds = get_code(im)
        chars = filename.split(".")[0]
        print(chars, preds)
        for i in range(len(chars)):
            if chars[i] == preds[i]:
                correct += 1
            total += 1
    print(correct/total)

#测试模型准确率
if __name__ == "__main__":
    file=os.listdir("test")
    filepath="test/"+file[7]
    im = cv2.imread(filepath)
    preds = get_code(im)
    preds="识别结果为："+preds[0]+preds[1]+preds[2]+preds[3]
    print(preds)
    canny0 = im
    img_PIL = Image.fromarray(cv2.cvtColor(canny0, cv2.COLOR_BGR2RGB))
    myfont = ImageFont.truetype(r'simfang.ttf', 18)
    draw = ImageDraw.Draw(img_PIL)
    draw.text((20, 5), str(preds), font=myfont, fill=(255, 23, 140))
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", img_OpenCV)

    key = cv2.waitKey(0)
    print(filepath)


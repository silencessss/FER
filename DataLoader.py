'''
:input | image(.jpg) and annotation(.txt)
    依照list.txt讀取data
    path='F:/DataSet/#FER/RAF-DB/basic/EmoLabel/list_patition_label.txt'
'''

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
#from imgaug import augmenters as iaa
#from PIL import Image
#from imutils import paths
#import pathlib
import Configer
(img_HEIGHT,img_WIDTH)=(224,224)

def TrainVal(data,lables,PROPORTION):
    (train_x, valid_x, train_y, valid_y) = train_test_split(
            data,
            lables,
            test_size=PROPORTION, 
            stratify=lables, 
            random_state=None, 
            shuffle=True)
    return train_x, valid_x, train_y, valid_y


def Transfer_DATA(data,label):
    print(':|INFO|: Transfer...')
    data = np.array(data,dtype='float32')
    
    label = np.array(label)
    LB = LabelEncoder()
    label = LB.fit_transform(label)
    label = to_categorical(label)
    print(list(LB.classes_))
    return data,label


def Read_JPG(Path_dir,images):
    print(':|INFO|: Read images...')
    data=[]
    for num_images in range(len(images)):
        image_NAME = images[num_images].split('.')[0] + '_aligned.jpg'
        path_each_image = os.path.join(Path_dir+image_NAME)
        #print(path_each_image)
        image = load_img(path_each_image,target_size=(img_HEIGHT,img_WIDTH,1))
        image = img_to_array(image,data_format=None)
        data.append(image)
    return data 

def Read_TXT(Path_txt):
    print(':|INFO|: Read txt file...')
    data = pd.read_csv(Path_txt,sep=" ",header=None)
    images=[]
    labels=[]
    for row in range(len(data)):
        images.append(data[0][row])
        labels.append(data[1][row])
    return images,labels

def main(Path_txt,Path_dir,SPLIT,PROPORTION):
    #Path_txt = r'F:/DataSet/#FER/RAF-DB/basic/EmoLabel/list_patition_label.txt'
    #Path_dir = r'F:/DataSet/#FER/RAF-DB/basic/Image/aligned/'
    images_list,labels_list=Read_TXT(Path_txt)


    print(':|INFO|: Read txt success!!!')
    print(len(images_list))
    print(len(labels_list))

    images = Read_JPG(Path_dir,images_list)

    
    Path_dir_OccR = r'F:/#DataSet/#FER/RAF-DB/basic/Image/Right_Occ/'
    #Path_dir_OccL = r'F:/#DataSet/#FER/RAF-DB/basic/Image/Left_Occ/'
    images_OccR = Read_JPG(Path_dir_OccR,images_list)
    #images_OccL = Read_JPG(Path_dir_OccL,images_list)

    for i in range(len(images_OccR)):
        images.append(images_OccR[i])
    
    #for i in range(len(images_OccL)):
    #    images.append(images_OccL[i])  


    label_main=labels_list.copy()

    for k in range(len(label_main)):
        labels_list.append(label_main[k])

    print(len(labels_list))

    #for k in range(len(label_main)):
    #    labels_list.append(label_main[k])
    #print(len(labels_list))
    data,labels = Transfer_DATA(images,labels_list)

    print(len(data))
    print(len(labels))

    if SPLIT==True:
        train_x, valid_x, train_y, valid_y = TrainVal(data,labels,PROPORTION)
        return train_x, valid_x, train_y, valid_y
    else:
        return data,labels
'''
Path_txt_Train = r'F:/#DataSet/#FER/RAF-DB/basic/EmoLabel/list_patition_label.txt'
Path_dir_Train = r'F:/#DataSet/#FER/RAF-DB/basic/Image/aligned/'
train_x, test_x, train_y, test_y = main(Path_txt_Train,Path_dir_Train,SPLIT=True,PROPORTION=Configer.PROPORTION_TRAIN_TEST)
'''


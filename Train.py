# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 04:04:29 2021

@author: PeterChan
"""
#----------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------#
#from nfnet import NFNet, nfnet_params
print("[INFO] LOADING Libraries...")
from imgaug.augmenters.color import MultiplySaturation
from imgaug.imgaug import pool
import numpy as np
import os
import sys
import PIL
from PIL import Image
import cv2
from tensorflow.python.framework import config
import pydotplus
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
from imutils import paths
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import Sequential, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python import keras
from tensorflow.python.keras.metrics import BinaryAccuracy, CategoricalAccuracy, FalsePositives, TrueNegatives, TruePositives
import datetime
from imgaug import augmenters as iaa
#----------------------------------------------------------#
# custom labraries
#----------------------------------------------------------#
import Configer
import DataLoader
import ModelLoader
#----------------------------------------------------------#
# GPU Detail
# [TensorFlow 2.0 硬體資源設置](https://hackmd.io/@shaoeChen/ryWIV4vkL)
#----------------------------------------------------------#
print('#--------Setting GPU-------#')
print('[INFO - GPU]: ',len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
print('[INFO] gpu setting done!')

#----------------------------------------------------------#
# LOADING Data---------------------------------------------#
#----------------------------------------------------------#
Path_txt_Train = r'F:/#DataSet/#FER/RAF-DB/basic/EmoLabel/list_patition_label.txt'
Path_dir_Train = r'F:/#DataSet/#FER/RAF-DB/basic/Image/aligned/'



train_x, test_x, train_y, test_y = DataLoader.main(Path_txt_Train,Path_dir_Train,SPLIT=True,PROPORTION=Configer.PROPORTION_TRAIN_TEST)

train_x,valid_x,train_y,valid_y = DataLoader.TrainVal(train_x,train_y,PROPORTION=Configer.PROPORTION_TRAIN_VALIDATION)


print(len(train_x))
print(len(valid_x))
print(len(test_x))

class dataAugment:
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        #iaa.Invert(0.5),
        #sometimes(iaa.Cutout(nb_iterations=2)),
        sometimes(iaa.Jigsaw(nb_rows=(10),nb_cols=(10))),
        sometimes(iaa.Jigsaw(nb_rows=(1,4),nb_cols=(1,4))),
        #sometimes(iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))),
        sometimes(iaa.Fliplr(0.5)),
        sometimes(iaa.Flipud(0.5)),
        #iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        #iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
        sometimes(iaa.Rotate((-45,180))),
        sometimes(iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
        ))  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
    aug_train = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False, zca_epsilon=1e-06, 
        rotation_range=0, 
        width_shift_range=0.0,
        height_shift_range=0.0, 
        brightness_range=None, 
        shear_range=0.0, 
        zoom_range=0.0,
        channel_shift_range=0.0, 
        fill_mode='nearest', 
        cval=0.0,
        horizontal_flip=False, 
        vertical_flip=False, 
        rescale=None,
        preprocessing_function=None, 
        data_format=None, 
        validation_split=0.0, 
        dtype=None
    )






baseModel = tf.keras.applications.efficientnet.EfficientNetB7(
                include_top = False,
                weights = 'imagenet',
                input_tensor = None,
                input_shape=Configer.INPUT_SHAPE,
                pooling = None,
                classes = Configer.CLASSES,
                classifier_activation='softmax'
            )
headModel = baseModel.output
#headModel = CoordAtt_bolck.CoordAtt(headModel,reduction = 32)
#headModel = GlobalAveragePooling2D()(headModel)
headModel = MaxPooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)#relu,tanh
headModel = Dense(32, activation="relu")(headModel)#relu,tanh
headModel = Dropout(0.2)(headModel)
headModel = Dense(7, activation="softmax")(headModel)
outModel = Model(inputs=baseModel.input, outputs=headModel)
model = outModel
#model = Sequential()
#model.load_weights(Configer.WEIGHT)
model.summary()

#----------------------------------------------------------#
# COMPILE Model
#----------------------------------------------------------#
print("[INFO] COMPILE model...")
model.compile(
    loss="categorical_crossentropy", 
    optimizer=Configer.OPT,
	metrics=Configer.METRICS
    )


if Configer.MODEL_PHASE_TRAIN==True:
    print('#--------Traing Start-------#')
    def LR_scheduler(epoch):
        if epoch < 100:
            return Configer.INIT_LR
        else:
            return Configer.INIT_LR * tf.math.exp(-0.1)


    CALLBACKS=[
        #tf.keras.callbacks.TensorBoard(log_dir=myconfig.path_log_dir,histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=Configer.PATH_CHECKPOINT,save_weights_only=True,monitor='val_accuracy',mode='max',save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(LR_scheduler)
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.8,patience=5,verbose=0,mode='auto',baseline=None,restore_best_weights=False)
    ]
    H = model.fit(
        dataAugment.aug_train.flow(train_x, train_y, batch_size=Configer.BS),
        steps_per_epoch=len(train_x) // Configer.BS,
        validation_data=dataAugment.aug_train.flow(valid_x, valid_y,batch_size=8),
        validation_steps=len(valid_x) // Configer.BS,
        #validation_batch_size = None,
        validation_freq = 1,
        epochs=Configer.EPOCHS,
        verbose = 1,
        callbacks=CALLBACKS,
        shuffle=True
        )
    print('#--------Traing Done-------#')

elif Configer.MODEL_PHASE_EVALUATE==True:
    #----------------------------------------------------------#
    # EVALUATE Model
    #----------------------------------------------------------#
    # 1.模型的BatchNormalization，Dropout，LayerNormalization等优化手段只在fit时，对训练集有用;
    # 2.在进行evaluate()的时候，这些优化都会失效，因此，再次进行evaluate(x_train,y_train),就算添加了batchsize，也不能达到相同的评估计算结果。
    # ————————————————
    # 版权声明：本文为CSDN博主「风筝不是风」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https://blog.csdn.net/weixin_45279187/article/details/110194739
    print("[INFO] EVALUATE Model...")
    try:
        eval_loss,eval_accuracy,*is_anything_else_being_returned  = model.evaluate(
            x=valid_x, y=valid_y, batch_size=32, verbose=1, sample_weight=None, steps=None,
            callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
            return_dict=False
            )
        print('[OUTPUT] eval_loss.. ',eval_loss)
        print('[OUTPUT] eval_accuracy.. ',eval_accuracy)
        try:
            print('[OUTPUT] *is_anything_else_being_returned.. ',*is_anything_else_being_returned)
        except:
            print('[ERROR] Evaluate error2')
    except:
        print('[ERROR]:ERROR: Evaluate error')

#----------------------------------------------------------#
# SAVING Model
#----------------------------------------------------------#
'''time getting'''
import time
time_now = time.localtime(time.time())
time_save = str(time_now[0])+'_'+str(time_now[1])+str(time_now[2])+'_'+str(time_now[3])+str(time_now[4])

print("[INFO] SAVING Model...")
path_save_model = Configer.PATH_MODEL+'[EPOCHS_'+str(Configer.EPOCHS)+']'+'[ACC_'+str(round(eval_accuracy,4))+']'+'[LOSS_'+str(round(eval_loss,4))+'].'+Configer.MODEL_FORMAT
print('path_save_model: ',path_save_model)
#model.save('./model_save/'+str(myData.name_model_save_dataset)+'_'+str(myconfig.backbone)+'_'+str(time_save)+'_'+str(myconfig.EPOCHS)+'_'+str(eval_accuracy)+'.'+myconfig.modelFormat, save_format=myconfig.modelFormat)
model.save(path_save_model,save_format=Configer.MODEL_FORMAT)

#----------------------------------------------------------#
# Plot
#----------------------------------------------------------#
import time
time_now = time.localtime(time.time())
time_save = str(time_now[0])+'_'+str(time_now[1])+str(time_now[2])+'_'+str(time_now[3])+str(time_now[4])

print('#--------Plot-------#')
'''plot'''
N = Configer.EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training/Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
print("[INFO] saving plot Loss...")
plt.savefig(Configer.PATH_plot_dir+str(time_save)+'_loss.jpg')

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training/Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
print("[INFO] saving plot Accuracy...")
plt.savefig(Configer.PATH_plot_dir+str(time_save)+'_acc.jpg')
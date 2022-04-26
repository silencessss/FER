from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import datetime
import time
import os
#----------------------------------------------------------#
# System
#----------------------------------------------------------#
SYSTEM_TIME = datetime.datetime.now().strftime('%Y%m%d-%H%M')
TIME_NOW = time.localtime(time.time())
#----------------------------------------------------------#
# Data
#----------------------------------------------------------#
DATASET='RAFDB'
CLASSES=7
INPUT_SHAPE=(224,224,3)
PROPORTION_TRAIN_TEST=0.2
PROPORTION_TRAIN_VALIDATION=0.1
#----------------------------------------------------------#
# Model
#----------------------------------------------------------#
BACKBONE='Efficient'
MODEL_FORMAT='h5'
#----------------------------------------------------------#
# Model Phase
#----------------------------------------------------------#
MODEL_PHASE_TRAIN=True
MODEL_PHASE_EVALUATE=True
INIT_LR = 1e-4
EPOCHS = 60
BS = 8 
OPT = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
WEIGHT=r'E:/Project/FER/logs/checkpoints/epoch41-[0.8453].h5'
METRICS = [
    #tf.keras.metrics.Accuracy(name ='accuracy'),
    #tf.keras.metrics.BinaryAccuracy(name='binaryaccuracy'),
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.TruePositives(name='TP'),
    tf.keras.metrics.TrueNegatives(name='TN'),
    tf.keras.metrics.FalsePositives(name='FP'),
    tf.keras.metrics.FalseNegatives(name='FN'),
    tf.keras.metrics.MeanSquaredError(name='mean_squared_error')
]

#----------------------------------------------------------#
# Saving
#----------------------------------------------------------#
PATH_CHECKPOINT = './logs/checkpoints/epoch.{epoch:02d}-{val_accuracy:.4f}.h5'
PATH_PRETRAINMODEL=r'E:/Project/FER/logs/checkpoints/epoch41-[0.8453].h5'
PATH_PLOT = './logs/history/'+DATASET+'_'+BACKBONE+'_'+str(EPOCHS)+'_'+SYSTEM_TIME
PATH_MODEL = os.path.join('./models',DATASET,BACKBONE,str(TIME_NOW[0]),str(TIME_NOW[1]))






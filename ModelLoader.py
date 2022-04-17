import tensorflow as tf
import tensorflow_hub as hub
import efficientnet.tfkeras as efn

class modelLoader:
    def TF2_keras_application():
        modelName = 'ResNet101'
        baseModel = tf.keras.applications.resnet.modelName(

        )
    def EfficientNet():
        model = efn.EfficientNetL2(weights='E:/Project/acne_severity_grading_tf2/weight_save/efficientnet-l2_noisy-student_notop.h5',include_top=False,input_shape=(224,224,3),drop_connect_rate=0,classes=4)
        return model

    def PnasNet():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(331,331,3)),
            hub.KerasLayer("https://tfhub.dev/google/imagenet/pnasnet_large/classification/5",trainable=False),
            tf.keras.layers.Dense(4,activation='softmax')
        ])
        model.build([None,331,331,3])
        return model
    def Inception_ResNet_v2():
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(299,299,3)),
            hub.KerasLayer('https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5',trainable=False),
            tf.keras.layers.Dense(4,activation='softmax')
        ])
        model.build([None,299,299,3])
        return model
    def Fully_Connected(baseModel):
        headModel = baseModel.output
        #headModel = CoordAtt_bolck.CoordAtt(headModel,reduction = 32)
        headModel = tf.keras.layers.GlobalAveragePooling2D()(headModel)
        #headModel = tf.keras.layers.MaxPooling2D(pool_size=(7,7))(headModel)
        #headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
        #headModel = Dense(64, activation="tanh")(headModel)#relu,tanh
        headModel = tf.keras.layers.Dense(32, activation="tanh")(headModel)#relu,tanh
        #headModel = Dropout(0.2)(headModel)
        headModel = tf.keras.layers.Dense(4, activation="softmax")(headModel)
        outModel = tf.keras.models.Model(inputs=baseModel.input, outputs=headModel)
        return outModel





'''
print('#--------Create Model-------#')
class get_model:
    def Classification_Model_Zoo(modelName):
        from classification_models.tfkeras import Classifiers
        if modelName=='seresnext101':
            Classifiers.models_names()
            MODEL, preprocess_input = Classifiers.get('seresnext101')
        elif modelName=='resnext101':
            Classifiers.models_names()
            MODEL, preprocess_input = Classifiers.get('resnext101')
        elif modelName=='senet154':
            Classifiers.models_names()
            MODEL, preprocess_input = Classifiers.get('senet154')
        elif modelName=='nasnetlarge':
            Classifiers.models_names()
            MODEL, preprocess_input = Classifiers.get('nasnetlarge')
        
        baseModel = MODEL(
            include_top = myconfig.include_top, 
            input_shape = (myconfig.Img_rows, myconfig.Img_cols, myconfig.Img_channel), 
            weights = 'imagenet', 
            input_tensor = None, 
            classes = myconfig.classes
        )
        headModel = baseModel.output
        #headModel = CoordAtt_bolck.CoordAtt(headModel,reduction = 32)
        #headModel = GlobalAveragePooling2D()(headModel)
        headModel = MaxPooling2D(pool_size=(7,7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        #headModel = Dense(64, activation="tanh")(headModel)#relu,tanh
        headModel = Dense(32, activation="tanh")(headModel)#relu,tanh
        #headModel = Dropout(0.2)(headModel)
        headModel = Dense(myconfig.classes, activation="softmax")(headModel)
        outModel = Model(inputs=baseModel.input, outputs=headModel)
        return outModel
    def TF_Keras_Application(modelName):
        if modelName=='resnet101':
            baseModel = tf.keras.applications.resnet.ResNet101(
                include_top = myconfig.include_top,
                weights = 'imagenet',
                input_tensor = None,
                input_shape=(myconfig.Img_rows,myconfig.Img_cols,myconfig.Img_channel),
                pooling = None,
                classes = myconfig.classes
            )
        elif modelName=='resnet152':
            baseModel = tf.keras.applications.resnet.ResNet152(
                include_top = myconfig.include_top,
                weights = 'imagenet',
                input_tensor = None,
                input_shape=(myconfig.Img_rows,myconfig.Img_cols,myconfig.Img_channel),
                pooling = None,
                classes = myconfig.classes
            )
        elif modelName=='efficientnetB7':
            baseModel = tf.keras.applications.efficientnet.EfficientNetB7(
                include_top = myconfig.include_top,
                weights = myconfig.weights,
                input_tensor = None,
                input_shape=(myconfig.Img_rows,myconfig.Img_cols,myconfig.Img_channel),
                pooling = None,
                classes = myconfig.classes,
                classifier_activation='softmax'
            )
        elif modelName=='nasnet':
            baseModel = tf.keras.applications.nasnet.NASNetLarge(
                include_top = myconfig.include_top,
                weights = myconfig.weights,
                input_tensor = None,
                input_shape=(myconfig.Img_rows,myconfig.Img_cols,myconfig.Img_channel),
                pooling = None,
                classes = myconfig.classes
            )
        headModel = baseModel.output
        #headModel = CoordAtt_bolck.CoordAtt(headModel,reduction = 32)
        #headModel = GlobalAveragePooling2D()(headModel)
        headModel = MaxPooling2D(pool_size=(7,7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)#relu,tanh
        headModel = Dense(32, activation="relu")(headModel)#relu,tanh
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(myconfig.classes, activation="softmax")(headModel)
        outModel = Model(inputs=baseModel.input, outputs=headModel)
              
        return outModel
    
    def Load_weights_train():
        baseModel = Sequential()
        baseModel.load_weights(myconfig.weights)
        return baseModel
    def Load_model_train():
        baseModel = models.load_model(
            myconfig.path_load_model, custom_objects=None, compile=True, options=None
            )
        return baseModel

print('[INFO] Create model..')
if myconfig.Training_mode == 'Proposed_Method':
    model = Modified_CNN001.Modified_CNN((myconfig.Img_rows,myconfig.Img_cols,myconfig.Img_channel),myconfig.classes)
elif myconfig.Training_mode == 'Load_model_train':
    model = get_model.Load_model_train()
elif myconfig.Training_mode == 'TF_Keras_Application':
    model = get_model.TF_Keras_Application(myconfig.backbone)
elif myconfig.Training_mode == 'Classification_Model_Zoo':
    model = get_model.Classification_Model_Zoo(myconfig.backbone)
elif myconfig.Training_mode == 'Proposed_Method_02':
    model = Modified_CNN002.Modified_CNN_002((myconfig.Img_rows,myconfig.Img_cols,myconfig.Img_channel),(myconfig.Img_rows,myconfig.Img_cols,myconfig.Img_channel),myconfig.classes)
else:
    print('[INFO] Please check your Training mode!!!')

'''
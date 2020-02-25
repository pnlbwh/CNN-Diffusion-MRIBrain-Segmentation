import keras
import argparse
from keras import optimizers
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
import numpy as np
import os
from keras import losses
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
#from keras.applications import Xception
from keras.utils import multi_gpu_model
import random
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import nibabel as nib
os.environ['CUDA_VISIBLE_DEVICES']="1"
import numpy as np

smooth = 1.
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_test(y_true, y_pred):

    y_true_f = np.array(y_true).flatten()
    y_pred_f =np.array(y_pred).flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def neg_dice_coef_loss(y_true, y_pred):
    return dice_coef(y_true, y_pred)


#define the model
def Comp_U_Net(input_shape,learn_rate=1e-3):

    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape,name='ip0')
    

    conv0a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    
    
    conv0a = bn()(conv0a)
    
    conv0b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv0a)

    conv0b = bn()(conv0b)

    
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0b)

    pool0 = Dropout(DropP)(pool0)


    conv1a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool0)
    
    
    conv1a = bn()(conv1a)
    
    conv1b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv1a)

    conv1b = bn()(conv1b)


    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1b)

    pool1 = Dropout(DropP)(pool1)



    

    conv2a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
    
    conv2a = bn()(conv2a)

    conv2b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv2a)

    conv2b = bn()(conv2b)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2b)

    pool2 = Dropout(DropP)(pool2)







    conv3a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)
    
    conv3a = bn()(conv3a)

    conv3b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv3a)

    conv3b = bn()(conv3b)



    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3b)

    pool3 = Dropout(DropP)(pool3)

    
    conv4a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool3)
    
    conv4a = bn()(conv4a)

    conv4b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv4a)

    conv4b = bn()(conv4b)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4b)

    pool4 = Dropout(DropP)(pool4)





    conv5a = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
    
    conv5a = bn()(conv5a)

    conv5b = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv5a)

    conv5b = bn()(conv5b)

    



    up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5b), (conv4b)],name='up6', axis=3)


    up6 = Dropout(DropP)(up6)

    conv6a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
    
    conv6a = bn()(conv6a)

    conv6b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv6a)

    conv6b = bn()(conv6b)





    up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6b),(conv3b)],name='up7', axis=3)

    up7 = Dropout(DropP)(up7)
    #add second output here

    conv7a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up7)
    
    conv7a = bn()(conv7a)

 

    conv7b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv7a)

    conv7b = bn()(conv7b)

   




    up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7b), (conv2b)],name='up8', axis=3)

    up8 = Dropout(DropP)(up8)
 
    conv8a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up8)
    
    conv8a = bn()(conv8a)

    
    conv8b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv8a)

    conv8b = bn()(conv8b)


    
    up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8b),(conv1b)],name='up9',axis=3)


    conv9a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up9)
    
    conv9a = bn()(conv9a)

    conv9b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv9a)

    conv9b = bn()(conv9b)




    up10 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv9b),(conv0b)],name='up10',axis=3)

    conv10a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up10)
    
    conv10a = bn()(conv10a)

   

    conv10b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv10a)

    conv10b = bn()(conv10b)


    
    final_op=Conv2D(1, (1, 1), activation='sigmoid',name='final_op')(conv10b)
    


    #----------------------------------------------------------------------------------------------------------------------------------

    #second branch - brain
    xup6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5b), (conv4b)],name='xup6', axis=3)

    

    xup6 = Dropout(DropP)(xup6)

    xconv6a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup6)
    
    xconv6a = bn()(xconv6a)

    

    xconv6b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xconv6a)

    xconv6b = bn()(xconv6b)





    xup7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(xconv6b),(conv3b)],name='xup7', axis=3)

    xup7 = Dropout(DropP)(xup7)
    
    xconv7a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup7)
    
    xconv7a = bn()(xconv7a)


    xconv7b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xconv7a)

    xconv7b = bn()(xconv7b)


    xup8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(xconv7b),(conv2b)],name='xup8', axis=3)

    xup8 = Dropout(DropP)(xup8)
    #add third xoutxout here
    
    xconv8a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup8)
    
    xconv8a = bn()(xconv8a)


    xconv8b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xconv8a)

    xconv8b = bn()(xconv8b)



    
    xup9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(xconv8b), (conv1b)],name='xup9',axis=3)

    xup9 = Dropout(DropP)(xup9)
    

    xconv9a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup9)
    
    xconv9a = bn()(xconv9a)

    
    xconv9b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xconv9a)

    xconv9b = bn()(xconv9b)

 
    
    xup10 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(xconv9b), (conv0b)],name='xup10',axis=3)

    xup10 = Dropout(DropP)(xup10)
    

    xconv10a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup10)
    
    xconv10a = bn()(xconv10a)


    xconv10b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xconv10a)

    xconv10b = bn()(xconv10b)

    


   
    
    xfinal_op=Conv2D(1, (1, 1), activation='sigmoid',name='xfinal_op')(xconv10b)


    #-----------------------------third branch



    #Concatenation fed to the reconstruction layer of all 3
   
    x_u_net_op0=keras.layers.concatenate([final_op,xfinal_op,keras.layers.add([final_op,xfinal_op])],name='res_a')
    

    






    res_1_conv0a = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(x_u_net_op0)
    
    
    res_1_conv0a = bn()(res_1_conv0a)
    
    res_1_conv0b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv0a)

    res_1_conv0b = bn()(res_1_conv0b)

    res_1_pool0 = MaxPooling2D(pool_size=(2, 2))(res_1_conv0b)

    res_1_pool0 = Dropout(DropP)(res_1_pool0)




    res_1_conv1a = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool0)
    
    
    res_1_conv1a = bn()(res_1_conv1a)
    
    res_1_conv1b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv1a)

    res_1_conv1b = bn()(res_1_conv1b)

    res_1_pool1 = MaxPooling2D(pool_size=(2, 2))(res_1_conv1b)

    res_1_pool1 = Dropout(DropP)(res_1_pool1)



    

    res_1_conv2a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool1)
    
    res_1_conv2a = bn()(res_1_conv2a)

    res_1_conv2b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv2a)

    res_1_conv2b = bn()(res_1_conv2b)

    
    res_1_pool2 = MaxPooling2D(pool_size=(2, 2))(res_1_conv2b)

    res_1_pool2 = Dropout(DropP)(res_1_pool2)







    res_1_conv3a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool2)
    
    res_1_conv3a = bn()(res_1_conv3a)

    res_1_conv3b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv3a)

    res_1_conv3b = bn()(res_1_conv3b)

    res_1_pool3 = MaxPooling2D(pool_size=(2, 2))(res_1_conv3b)

    res_1_pool3 = Dropout(DropP)(res_1_pool3)

    
    res_1_conv4a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool3)
    
    res_1_conv4a = bn()(res_1_conv4a)

    res_1_conv4b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv4a)

    res_1_conv4b = bn()(res_1_conv4b)

    
    res_1_pool4 = MaxPooling2D(pool_size=(2, 2))(res_1_conv4b)

    res_1_pool4 = Dropout(DropP)(res_1_pool4)





    res_1_conv5a = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_pool4)
    
    res_1_conv5a = bn()(res_1_conv5a)

    res_1_conv5b = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv5a)

    res_1_conv5b = bn()(res_1_conv5b)




    res_1_up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(res_1_conv5b), (res_1_conv4b)],name='res_1_up6', axis=3)


    res_1_up6 = Dropout(DropP)(res_1_up6)

    res_1_conv6a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up6)
    
    res_1_conv6a = bn()(res_1_conv6a)


    res_1_conv6b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv6a)

    res_1_conv6b = bn()(res_1_conv6b)



    res_1_up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(res_1_conv6b),(res_1_conv3b)],name='res_1_up7', axis=3)

    res_1_up7 = Dropout(DropP)(res_1_up7)
    #add second res_1_output here
    res_1_conv7a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up7)
    
    res_1_conv7a = bn()(res_1_conv7a)

    
    res_1_conv7b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv7a)

    res_1_conv7b = bn()(res_1_conv7b)



    res_1_up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(res_1_conv7b),(res_1_conv2b)],name='res_1_up8', axis=3)

    res_1_up8 = Dropout(DropP)(res_1_up8)
    #add third outout here
    res_1_conv8a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up8)
    
    res_1_conv8a = bn()(res_1_conv8a)


    res_1_conv8b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv8a)

    res_1_conv8b = bn()(res_1_conv8b)


    res_1_up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(res_1_conv8b), (res_1_conv1b)],name='res_1_up9',axis=3)

    res_1_up9 = Dropout(DropP)(res_1_up9)

    res_1_conv9a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up9)
    
    res_1_conv9a = bn()(res_1_conv9a)


    res_1_conv9b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv9a)

    res_1_conv9b = bn()(res_1_conv9b)




    res_1_up10 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(res_1_conv9b),(res_1_conv0b)],name='res_1_up10',axis=3)

    res_1_up10 = Dropout(DropP)(res_1_up10)
    

    res_1_conv10a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_up10)
    
    res_1_conv10a = bn()(res_1_conv10a)

   
    res_1_conv10b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv10a)

    res_1_conv10b = bn()(res_1_conv10b)


    res_1_final_op=Conv2D(1, (1, 1), activation='sigmoid',name='res_1_final_op')(res_1_conv10b)


    model=Model(inputs=[inputs],outputs=[final_op,
                                         xfinal_op,
                                         res_1_final_op,
                                     ])
    
    #print("Training using multiple GPUs...")
    #model = multi_gpu_model(model, gpus=1)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-5),loss={'final_op':dice_coef_loss,
                                                'xfinal_op':neg_dice_coef_loss,
                                                'res_1_final_op':'mse'})

    return model

#----------------------------------------------------Main--------------------------------------------------#


parser = argparse.ArgumentParser()

parser.add_argument('-f', action='store', dest='model_folder', type=str,
                        help=" folder which contain the trained model")

args = parser.parse_args()

training_data_folder = args.model_folder.rstrip('/')

train_x = training_data_folder + '/axial-traindata-dwi.npy'
train_y = training_data_folder + '/axial-traindata-mask.npy'

model = Comp_U_Net(input_shape=(256,256,1))
#print(model.summary())

x_train = np.load(train_x)
y_train = np.load(train_y)

x_train=x_train.reshape(x_train.shape+(1,))
y_train=y_train.reshape(y_train.shape+(1,))

# Log output
print ("Training dwi volume shape: ", x_train.shape)
print ("Training dwi mask volume shape: ", y_train.shape)

#tensorboard = TensorBoard('/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/data/comp/logs', histogram_freq=1)
csv_logger = CSVLogger(training_data_folder + '/axial.csv', append=True, separator=';')

# checkpoint
filepath = training_data_folder + "/weights-axial-improvement-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True)

# Trains the model for a given number of epochs (iterations on a dataset).
history_callback = model.fit([x_train], 
                             [y_train,y_train,y_train], 
                             validation_split=0.2,
                             batch_size=4, 
                             epochs=10,
                             shuffle=True, 
                             verbose=1, 
                             callbacks=[csv_logger, checkpoint])

import h5py
# serialize model to JSON
model_json = model.to_json()
with open(training_data_folder + "/CompNetBasicModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(training_data_folder + "/CompNetBasicModel_weights_DWI_axial_final.h5")
print("Saved model to disk location: ", training_data_folder)

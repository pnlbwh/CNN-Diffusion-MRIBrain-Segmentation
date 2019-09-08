from keras.models import load_model
from keras.models import model_from_json
import cv2
import numpy as np



import keras
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
import nibabel as nib


smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Negative dice to obtain region of interest (ROI-Branch loss) 
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Positive dice to minimize overlap with region of interest (Complementary branch (CO) loss)
def neg_dice_coef_loss(y_true, y_pred):
    return dice_coef(y_true, y_pred)


# load json and create model
json_file = open('../model/CompNetmodel_arch_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("../model/CompNetmodel_weights_new.h5")
print("Loaded model from disk")

# evaluate loaded model on test data

loaded_model.compile(optimizer=Adam(lr=1e-5),
                  loss={'output1': dice_coef_loss, 'output2': dice_coef_loss, 'output3': dice_coef_loss,
                        'output4': dice_coef_loss, 'conv10': dice_coef_loss, 'final_op': dice_coef_loss,
                        'xoutput1': neg_dice_coef_loss, 'xoutput2': neg_dice_coef_loss, 'xoutput3': neg_dice_coef_loss,
                        'xoutput4': neg_dice_coef_loss, 'xconv10': neg_dice_coef_loss, 'xfinal_op': neg_dice_coef_loss,
                        'xxoutput1': 'mse', 'xxoutput2': 'mse', 'xxoutput3': 'mse', 'xxoutput4': 'mse',
                        'xxconv10': 'mse', 'xxfinal_op': 'mse'})

test_x = '../output/case1_data.npy'
#test_y = '../data/y_train_1sub.npy'

x_test = np.load(test_x)
x_test = x_test.reshape(x_test.shape+(1,))

preds_train = loaded_model.predict(x_test, verbose=1)

SO = preds_train[5]   # Segmentation Output
CO = preds_train[11]  # Complementrary Output
RO = preds_train[17]  # Reconstruction Output

np.save('../output/case1SO.npy', SO)
np.save('../output/case1CO.npy', CO)
np.save('../output/case1RO.npy', RO)

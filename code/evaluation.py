# %%
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

  
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import shutil

physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", physical_devices, len(tf.config.list_physical_devices('GPU')))
# tf.test.gpu_device_name()
device_lib.list_local_devices()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)








# %%



# %%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32

import os 
os.listdir('model_saved_new1')

model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']

# result_path  = ''
summary_path = 'model_saved_new1\\'

testY_data = np.load(summary_path + 'testY_' + '.npy')
trainY_data = np.load(summary_path + 'trainY_' + '.npy')
testX_data = np.load(summary_path + 'testX_' + '.npy')
trainX_data = np.load(summary_path + 'trainX_' + '.npy')

summary_path0 = 'model_saved_new1\\' + model_list[0]+'\\'
summary_path1 = 'model_saved_new1\\' + model_list[1]+'\\'
summary_path2 = 'model_saved_new1\\' + model_list[2]+'\\'
summary_path3 = 'model_saved_new1\\' + model_list[3]+'\\'
summary_path4 = 'model_saved_new1\\' + model_list[4]+'\\'
summary_path5 = 'model_saved_new1\\' + model_list[5]+'\\'
summary_path6 = 'model_saved_new1\\ensemble\\' + model_list[6]+'\\'


### take one path is enough 
# hyper_result_path = os.listdir(summary_path0)
hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

### hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

### model_name0 = ['adam0.001', 'adam0.0001', 'sgd0.001', 'sgd0.0001'] 
model_name0 = ['model-Dense201-19-0.9525','model-Dense201-23-0.9450','model-Dense201-39-0.8950','model-Dense201-38-0.8200', 'model-Dense201-39-0.8950']
model_name1 = ['model-InceptionV3-35-0.9075','model-InceptionV3-40-0.9150','model-InceptionV3-28-0.8400','model-InceptionV3-36-0.8300','model-InceptionV3-28-0.8400']
model_name2 = ['model-MobileV2-14-0.9500','model-MobileV2-11-0.9500','model-MobileV2-25-0.8975','model-MobileV2-36-0.8450','model-MobileV2-36-0.9525']
model_name3 = ['model-Res101-36-0.9475','model-Res101-32-0.9350','model-Res101-32-0.8725','model-Res101-39-0.7975','model-Res101-32-0.8725']
model_name4 = ['model-VGG16-28-0.9350','model-VGG16-40-0.8850','model-VGG16-07-0.7825','model-VGG16-39-0.7325','model-VGG16-07-0.7825']
model_name5 = ['model-Xception-22-0.9250','model-Xception-36-0.9225','model-Xception-38-0.8300','model-Xception-40-0.7700','model-Xception-38-0.8300']


# %% [markdown]
# different model with adam0.001

# %%
#### for loss and accuracy curve
import matplotlib.pyplot as plt
import json
import pandas as pd

# model_history_json_name = summary_path +  result_path + model_name + '_history.json'
# history_json = json.load(open(model_history_json_name, 'r'))

def history_model(summary_path, hyper_result_path,model_name):
    history = summary_path + hyper_result_path + '\\' + model_name + '_history.json'
    history_json = json.load(open(history, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1
    return history_df

summary_path = 'model_saved_new1\\'
hyper_result_path = ['adam0.01','adam0.001', 'adam0.0001']
model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']
summary_path0 = 'model_saved_new1\\' + model_list[0]+'\\'
summary_path1 = 'model_saved_new1\\' + model_list[1]+'\\'
summary_path2 = 'model_saved_new1\\' + model_list[2]+'\\'
summary_path3 = 'model_saved_new1\\' + model_list[3]+'\\'
summary_path4 = 'model_saved_new1\\' + model_list[4]+'\\'
summary_path5 = 'model_saved_new1\\' + model_list[5]+'\\'
summary_path6 = 'model_saved_new1\\ensemble\\' + model_list[6]+'\\'

model_name0 = ['model-Dense201-39-0.8950','model-InceptionV3-40-0.9150','model-MobileV2-11-0.9500','model-Res101-36-0.9475','model-VGG16-07-0.7825','model-Xception-38-0.8300']

 
history_df_model0_adam0 = history_model(summary_path0, hyper_result_path[0], model_name0[0])
history_df_model1_adam0 = history_model(summary_path1, hyper_result_path[2], model_name0[1])
history_df_model2_adam0 = history_model(summary_path2, hyper_result_path[2], model_name0[2])
history_df_model3_adam0 = history_model(summary_path3, hyper_result_path[1], model_name0[3])
history_df_model4_adam0 = history_model(summary_path4, hyper_result_path[0], model_name0[4])
history_df_model5_adam0 = history_model(summary_path5, hyper_result_path[0], model_name0[5])
# history_df_model6_adam0 = history_model(summary_path6, '', 'model-ensemble4-10-0.9725')


# model_list_name = ['DenseNet201','InceptionNetv3','MobileNetV2','ResNet101V2','VGG16','Xception','Ensemble']
model_list_name = ['DenseNet201','InceptionNetv3','MobileNetV2','ResNet101V2','VGG16','Xception']


plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['loss'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['loss'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['loss'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['loss'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['loss'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['loss'],label=model_list_name[5])
# plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['loss'],label=model_list_name[6])
# plt.xlim(-10, 10)
plt.ylim(0,1)
plt.legend(loc="upper right")
plt.title("Training Loss for each model after fine-tuned")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
# annot_max(history_df_adam0['epochs'],history_df_adam0['loss'])
# plt.savefig(summary_path + 'result\\loss_acc\\' + 'training_loss_' +'indivual_vs_ensemble' + '.png', bbox_inches = 'tight')
plt.savefig(summary_path + 'result\\loss_acc\\' + 'training_loss_' +'indivual' + '.png', bbox_inches = 'tight')
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['val_loss'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['val_loss'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['val_loss'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['val_loss'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['val_loss'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['val_loss'],label=model_list_name[5])
# plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['val_loss'],label=model_list_name[6])

plt.ylim(0,1)
plt.legend(loc="upper right")
plt.title("Validation Loss for each model after fine-tuned")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
# annot_max(history_df_adam0['epochs'],history_df_adam0['loss'])
plt.savefig(summary_path + 'result\\loss_acc\\' + 'validation_loss_' +'indivual' + '.png', bbox_inches = 'tight')
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['accuracy'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['accuracy'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['accuracy'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['accuracy'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['accuracy'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['accuracy'],label=model_list_name[5])
# plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['accuracy'],label=model_list_name[6])

# plt.xlim(-10, 10)
plt.ylim(0.5,1)
plt.legend(loc="lower right")
plt.title("Tranning accuracy for each model after fine-tuned")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
# annot_max(history_df_adam0['epochs'],history_df_adam0['accuracy'])
plt.savefig(summary_path + 'result\\loss_acc\\' + 'training_accuracy_' +'indivual' + '.png', bbox_inches = 'tight')
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['val_accuracy'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['val_accuracy'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['val_accuracy'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['val_accuracy'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['val_accuracy'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['val_accuracy'],label=model_list_name[5])
# plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['val_accuracy'],label=model_list_name[6])
plt.ylim(0.5,1)
plt.legend(loc="lower right")
plt.title("Validation accuracy for each model after fine-tuned")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
# annot_max(history_df_adam0['epochs'],history_df_adam0['accuracy'])
plt.savefig(summary_path + 'result\\loss_acc\\' + 'validation_accuracy_' +'indivual' + '.png', bbox_inches = 'tight')
plt.show()




# %% [markdown]
# same model with different hyper (for densenet part only)

# %%
import matplotlib.pyplot as plt
import json
import pandas as pd
model_list_name = ['DenseNet201','InceptionNetv3','MobileNetV2','ResNet101V2','VGG16','Xception']

def history_model(summary_path, hyper_result_path,model_name):
    history = summary_path + hyper_result_path + '\\' + model_name + '_history.json'
    history_json = json.load(open(history, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1
    return history_df

def history_saved(summary_path, hyper_result_path, model_name, model_list,model_list_name):
    ### hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']
    ### summary_path = 'model_saved_new1\\'
    ### model_name[0] = model_name0[0]
    history_df_adam0 = history_model(summary_path, hyper_result_path[1], model_name[0])
    history_df_adam1 = history_model(summary_path, hyper_result_path[0], model_name[1])
    history_df_sgd0 = history_model(summary_path, hyper_result_path[3], model_name[2])
    history_df_sgd1 = history_model(summary_path, hyper_result_path[2], model_name[3])


    ### from here all json have same name so always is model_name[4] hyper_result_path follow sequence
    history_df_adam2 = history_model(summary_path, hyper_result_path[4], model_name[4])
    history_df_adam3 = history_model(summary_path, hyper_result_path[5], model_name[4])
    history_df_sgd2 = history_model(summary_path, hyper_result_path[6], model_name[4])


    #### plotting follow: 'Adam_0.1','Adam_0.01', 'Adam_0.001', 'Adam_0.0001','SGD_0.01', 'SGD_0.001', 'SGD_0.0001',
    hyper_list_name = ['Adam_0.001', 'Adam_0.0001', 'SGD_0.001', 'SGD_0.0001','Adam_0.1','Adam_0.01','SGD_0.01']
    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    plt.plot(history_df_adam2['epochs'],history_df_adam2['loss'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['loss'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['loss'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['loss'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['loss'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['loss'],label=hyper_list_name[2])
    plt.plot(history_df_sgd1['epochs'],history_df_sgd1['loss'],label=hyper_list_name[3])

    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.title("Tranning Loss for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'training_loss_' + model_list + '.png', bbox_inches = 'tight')
    plt.show()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    plt.plot(history_df_adam2['epochs'],history_df_adam2['val_loss'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['val_loss'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['val_loss'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['val_loss'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['val_loss'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['val_loss'],label=hyper_list_name[2])
    plt.plot(history_df_sgd1['epochs'],history_df_sgd1['val_loss'],label=hyper_list_name[3])
    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.title("Validation Loss for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'val_loss_' + model_list + '.png' , bbox_inches = 'tight')
    plt.show()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    plt.plot(history_df_adam2['epochs'],history_df_adam2['accuracy'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['accuracy'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['accuracy'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['accuracy'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['accuracy'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['accuracy'],label=hyper_list_name[2])
    plt.plot(history_df_sgd1['epochs'],history_df_sgd1['accuracy'],label=hyper_list_name[3])
    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title("Tranning accuracy for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("accuracy")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'training_accuracy_' + model_list + '.png', bbox_inches = 'tight')
    plt.show()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    plt.plot(history_df_adam2['epochs'],history_df_adam2['val_accuracy'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['val_accuracy'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['val_accuracy'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['val_accuracy'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['val_accuracy'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['val_accuracy'],label=hyper_list_name[2])
    plt.plot(history_df_sgd1['epochs'],history_df_sgd1['val_accuracy'],label=hyper_list_name[3])
    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title("Validation accuracy for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("accuracy")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'val_accuracy_' + model_list + '.png' , bbox_inches = 'tight')
    plt.show()

# %%
# loss_acc_name = ['training_loss_','val_loss_','training_acc_','val_acc_']
history_saved(summary_path0,hyper_result_path,model_name0, model_list[0],model_list_name[0])
# history_saved(summary_path1,hyper_result_path,model_name1, model_list[1],model_list_name[1])
# history_saved(summary_path2,hyper_result_path,model_name2, model_list[2],model_list_name[2])
# history_saved(summary_path3,hyper_result_path,model_name3, model_list[3],model_list_name[3])
# history_saved(summary_path4,hyper_result_path,model_name4, model_list[4],model_list_name[4])
# history_saved(summary_path5,hyper_result_path,model_name5, model_list[5],model_list_name[5])

# %% [markdown]
# same model with different hyper (except densenet part)

# %%
import matplotlib.pyplot as plt
import json
import pandas as pd
model_list_name = ['DenseNet201','InceptionNetv3','MobileNetV2','ResNet101V2','VGG16','Xception']

def history_model(summary_path, hyper_result_path,model_name):
    history = summary_path + hyper_result_path + '\\' + model_name + '_history.json'
    history_json = json.load(open(history, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1
    return history_df

def history_saved(summary_path, hyper_result_path, model_name, model_list,model_list_name):
    ### hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']
    ### summary_path = 'model_saved_new1\\'
    ### model_name[0] = model_name0[0]
    history_df_adam0 = history_model(summary_path, hyper_result_path[1], model_name[0])
    history_df_adam1 = history_model(summary_path, hyper_result_path[0], model_name[1])
    history_df_sgd0 = history_model(summary_path, hyper_result_path[3], model_name[2])
    # history_df_sgd1 = history_model(summary_path, hyper_result_path[2], model_name[3])


    ### from here all json have same name so always is model_name[4] hyper_result_path follow sequence
    # history_df_adam2 = history_model(summary_path, hyper_result_path[4], model_name[4])
    history_df_adam3 = history_model(summary_path, hyper_result_path[5], model_name[4])
    history_df_sgd2 = history_model(summary_path, hyper_result_path[6], model_name[4])


    #### plotting follow: 'Adam_0.1','Adam_0.01', 'Adam_0.001', 'Adam_0.0001','SGD_0.01', 'SGD_0.001', 'SGD_0.0001',
    hyper_list_name = ['Adam_0.001', 'Adam_0.0001', 'SGD_0.001', 'SGD_0.0001','Adam_0.1','Adam_0.01','SGD_0.01']
    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    # plt.plot(history_df_adam2['epochs'],history_df_adam2['loss'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['loss'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['loss'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['loss'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['loss'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['loss'],label=hyper_list_name[2])
    # plt.plot(history_df_sgd1['epochs'],history_df_sgd1['loss'],label=hyper_list_name[3])

    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.title("Tranning Loss for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'training_loss_' + model_list + '.png', bbox_inches = 'tight')
    plt.show()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    # plt.plot(history_df_adam2['epochs'],history_df_adam2['val_loss'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['val_loss'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['val_loss'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['val_loss'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['val_loss'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['val_loss'],label=hyper_list_name[2])
    # plt.plot(history_df_sgd1['epochs'],history_df_sgd1['val_loss'],label=hyper_list_name[3])
    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.title("Validation Loss for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'val_loss_' + model_list + '.png' , bbox_inches = 'tight')
    plt.show()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    # plt.plot(history_df_adam2['epochs'],history_df_adam2['accuracy'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['accuracy'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['accuracy'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['accuracy'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['accuracy'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['accuracy'],label=hyper_list_name[2])
    # plt.plot(history_df_sgd1['epochs'],history_df_sgd1['accuracy'],label=hyper_list_name[3])
    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title("Tranning accuracy for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("accuracy")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'training_accuracy_' + model_list + '.png', bbox_inches = 'tight')
    plt.show()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    # plt.plot(history_df_adam2['epochs'],history_df_adam2['val_accuracy'],label=hyper_list_name[4])
    plt.plot(history_df_adam3['epochs'],history_df_adam3['val_accuracy'],label=hyper_list_name[5])
    plt.plot(history_df_adam0['epochs'],history_df_adam0['val_accuracy'],label=hyper_list_name[0])
    plt.plot(history_df_adam1['epochs'],history_df_adam1['val_accuracy'],label=hyper_list_name[1])
    plt.plot(history_df_sgd2['epochs'],history_df_sgd2['val_accuracy'],label=hyper_list_name[6])
    plt.plot(history_df_sgd0['epochs'],history_df_sgd0['val_accuracy'],label=hyper_list_name[2])
    # plt.plot(history_df_sgd1['epochs'],history_df_sgd1['val_accuracy'],label=hyper_list_name[3])
    # plt.xlim(-10, 10)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title("Validation accuracy for " + model_list_name + 
              " different Optimizer and learning rate")
    plt.xlabel("Epoch #")
    plt.ylabel("accuracy")
    plt.savefig('model_saved_new1\\' + 'result\\loss_acc\\' + 'val_accuracy_' + model_list + '.png' , bbox_inches = 'tight')
    plt.show()

# %%
# loss_acc_name = ['training_loss_','val_loss_','training_acc_','val_acc_']
# history_saved(summary_path0,hyper_result_path,model_name0, model_list[0],model_list_name[0])
history_saved(summary_path1,hyper_result_path,model_name1, model_list[1],model_list_name[1])
history_saved(summary_path2,hyper_result_path,model_name2, model_list[2],model_list_name[2])
history_saved(summary_path3,hyper_result_path,model_name3, model_list[3],model_list_name[3])
history_saved(summary_path4,hyper_result_path,model_name4, model_list[4],model_list_name[4])
history_saved(summary_path5,hyper_result_path,model_name5, model_list[5],model_list_name[5])

# %% [markdown]
# Confusion Matrix with correct version for dense part only different hyper

# %%
def load_model_result(summary_path, hyper_result_path, model_name): 
    model0_adam0 = load_model(summary_path + hyper_result_path[1] + '\\' + model_name[0] + '.h5')
    model0_adam1 = load_model(summary_path + hyper_result_path[0] + '\\' + model_name[1] + '.h5')
    model0_sgd0 = load_model(summary_path + hyper_result_path[3] + '\\' + model_name[2] + '.h5')
    model0_sgd1 = load_model(summary_path + hyper_result_path[2] + '\\' + model_name[3] + '.h5')
    model0_adam2 = load_model(summary_path + hyper_result_path[4] + '\\' + model_name[4] + '.h5')
    model0_adam3 = load_model(summary_path + hyper_result_path[5] + '\\' + model_name[5] + '.h5')
    model0_sgd2 = load_model(summary_path + hyper_result_path[6] + '\\' + model_name[6] + '.h5')

    return model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam2,model0_adam3,model0_sgd2

def confusion_plot(model, filename):
    batch_size = 32
    # make predications on the testing set
    print("evaluating network for " + filename + ".....")
    predIdxs = model.predict(testX_data, batch_size=batch_size)

    threshold = 0.5

    # Convert predicted probabilities to binary values using threshold
    predIdxs = (predIdxs > threshold).astype(int)


    #### for confusion matrix
    # compute the confusion matrix nad use it 
    # to drive the raw accuracy, sensitivity 
    # and specificity
    # cm = confusion_matrix(testY_data.argmax(axis=1), predIdxs)

    cm = confusion_matrix(testY_data, predIdxs)

    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]

    print('True positive = ', tp)
    print('True negative = ', tn)
    print('False positive = ', fp)
    print('False negative = ', fn)

    total_test = sum(sum(cm))
    accurancy = (cm[0, 0] + cm[1, 1]) / total_test	### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])	### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])	### Specificity (true negatives / all actual negatives) =TN / TN + FP
    f1score = 2*(precision*sensitivity)/(precision+sensitivity)
    npv = tn/(tn+fn)


    print(cm)
    print("accurancy: {:.4f}".format(accurancy))
    print("precision: {:.4f}".format(precision))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    print("f1-score: {:.4f}".format(f1score))
    print("NPV: {:.4f}".format(npv))



    import seaborn as sns
    ### fmt ='g' to turn off scientific number 
    sns.heatmap(cm, cmap="Blues",annot=True, fmt='g',xticklabels=["Normal", "Covid-19"], yticklabels=["Normal", "Covid-19"])
    plt.title("Confusion Matrix: " + filename)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.savefig(summary_path + 'result\\' + filename + '.png')
    plt.show()
    print('##########################################################################################################')

    accurancy = "%.4f" % accurancy
    precision = "%.4f" % precision
    sensitivity = "%.4f" % sensitivity
    specificity = "%.4f" % specificity
    f1score = "%.4f" % f1score
    npv = "%.4f" % npv

    evaluation_result = [accurancy,precision,sensitivity,specificity,f1score,npv]
    return evaluation_result

def different_model_evaluation(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam2,model0_adam3,model0_sgd2,model_list, filename):
    evaluation_result_name = ['Adam_0.001', 'Adam_0.0001', 'SGD_0.001', 'SGD_0.0001','Adam_0.1','Adam_0.01','SGD_0.01']
    evaluation_result_adam0 = [evaluation_result_name[0]] + confusion_plot(model0_adam0, model_list + '_' + hyper_result_path[1])
    evaluation_result_adam1 = [evaluation_result_name[1]] + confusion_plot(model0_adam1, model_list + '_' + hyper_result_path[0])
    evaluation_result_sgd0 = [evaluation_result_name[2]] + confusion_plot(model0_sgd0, model_list + '_' + hyper_result_path[3])
    evaluation_result_sgd1 = [evaluation_result_name[3]] + confusion_plot(model0_sgd1, model_list + '_' + hyper_result_path[2])
    evaluation_result_adam2 = [evaluation_result_name[4]] + confusion_plot(model0_adam2, model_list + '_' + hyper_result_path[4])
    evaluation_result_adam3 = [evaluation_result_name[5]] + confusion_plot(model0_adam3, model_list + '_' + hyper_result_path[5])
    evaluation_result_sgd2 = [evaluation_result_name[6]] + confusion_plot(model0_sgd2, model_list + '_' + hyper_result_path[6])


    ### evaluation_result_adam0 = ['Adam_0.001', '0.9525', '0.9250', '0.9788', '0.9289', '0.9512', '0.9788']

    import csv
    evaluation_method = ['Method','Accurancy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'NPV']
    evaluation_result = [evaluation_result_adam0,evaluation_result_adam1,evaluation_result_sgd0,evaluation_result_sgd1, evaluation_result_adam2, evaluation_result_adam3, evaluation_result_sgd2]

    with open(summary_path + 'result\\' + filename, 'w') as f:    
        csv_writer = csv.writer(f)
        csv_writer.writerow(evaluation_method)
        csv_writer.writerows(evaluation_result)
    f.close


# %%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32

import os 
os.listdir('model_saved_new1')

model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']

# result_path  = ''
summary_path = 'model_saved_new1\\'

testY_data = np.load(summary_path + 'testY_' + '.npy')
trainY_data = np.load(summary_path + 'trainY_' + '.npy')
testX_data = np.load(summary_path + 'testX_' + '.npy')
trainX_data = np.load(summary_path + 'trainX_' + '.npy')

summary_path0 = 'model_saved_new1\\' + model_list[0]+'\\'
summary_path1 = 'model_saved_new1\\' + model_list[1]+'\\'
summary_path2 = 'model_saved_new1\\' + model_list[2]+'\\'
summary_path3 = 'model_saved_new1\\' + model_list[3]+'\\'
summary_path4 = 'model_saved_new1\\' + model_list[4]+'\\'
summary_path5 = 'model_saved_new1\\' + model_list[5]+'\\'
summary_path6 = 'model_saved_new1\\ensemble\\' + model_list[6]+'\\'


### take one path is enough 

### cannot use summary_path0, it have extra folder
# hyper_result_path = os.listdir(summary_path1)
hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

### model_name0 = ['adam0.001', 'adam0.0001', 'sgd0.001', 'sgd0.0001']
model_name0 = ['model-Dense201-19-0.9525','model-Dense201-23-0.9450','model-Dense201-39-0.8950','model-Dense201-38-0.8200','model-Dense201-13-0.9550','model-Dense201-32-0.9625','model-Dense201-31-0.9300']
model_name1 = ['model-InceptionV3-35-0.9075','model-InceptionV3-40-0.9150','model-InceptionV3-28-0.8400','model-InceptionV3-36-0.8300']
model_name2 = ['model-MobileV2-14-0.9500','model-MobileV2-11-0.9500','model-MobileV2-25-0.8975','model-MobileV2-36-0.8450']
model_name3 = ['model-Res101-36-0.9475','model-Res101-32-0.9350','model-Res101-32-0.8725','model-Res101-39-0.7975']
model_name4 = ['model-VGG16-28-0.9350','model-VGG16-40-0.8850','model-VGG16-07-0.7825','model-VGG16-39-0.7325']
model_name5 = ['model-Xception-22-0.9250','model-Xception-36-0.9225','model-Xception-38-0.8300','model-Xception-40-0.7700']


### adam2 = adam0.01; adam3 = adam0.1 sgd2 = sgd0.01
### cannot load all not enough memory
#dense
model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam2, model0_adam3,model0_sgd2 = load_model_result(summary_path0,hyper_result_path,model_name0)
# ##inception
# model1_adam0,model1_adam1,model1_sgd0,model1_sgd1 = load_model_result(summary_path1,hyper_result_path,model_name1)

# model2_adam0,model2_adam1,model2_sgd0,model2_sgd1 = load_model_result(summary_path2,hyper_result_path,model_name2)

# model3_adam0,model3_adam1,model3_sgd0,model3_sgd1 = load_model_result(summary_path3,hyper_result_path,model_name3)

# model4_adam0,model4_adam1,model4_sgd0,model4_sgd1 = load_model_result(summary_path4,hyper_result_path,model_name4)

# model5_adam0,model5_adam1,model5_sgd0,model5_sgd1 = load_model_result(summary_path5,hyper_result_path,model_name5)
# model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception']
different_model_evaluation(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam2,model0_adam3,model0_sgd2,model_list[0],model_list[0]+'_evaluation.csv')
# different_model_evaluation(model1_adam0,model1_adam1,model1_sgd0,model1_sgd1,model_list[1],model_list[1]+'_evaluation.csv')
# different_model_evaluation(model2_adam0,model2_adam1,model2_sgd0,model2_sgd1,model_list[2],model_list[2]+'_evaluation.csv')
# different_model_evaluation(model3_adam0,model3_adam1,model3_sgd0,model3_sgd1,model_list[3],model_list[3]+'_evaluation.csv')
# different_model_evaluation(model4_adam0,model4_adam1,model4_sgd0,model4_sgd1,model_list[4],model_list[4]+'_evaluation.csv')
# different_model_evaluation(model5_adam0,model5_adam1,model5_sgd0,model5_sgd1,model_list[5],model_list[5]+'_evaluation.csv')


# %% [markdown]
# roc testing individual

# %%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def load_model_result(summary_path, hyper_result_path, model_name): 
    model0_adam0 = load_model(summary_path + hyper_result_path[1] + '\\' + model_name[0] + '.h5')
    model0_adam1 = load_model(summary_path + hyper_result_path[0] + '\\' + model_name[1] + '.h5')
    model0_sgd0 = load_model(summary_path + hyper_result_path[3] + '\\' + model_name[2] + '.h5')
    model0_sgd1 = load_model(summary_path + hyper_result_path[2] + '\\' + model_name[3] + '.h5')
    model0_adam2 = load_model(summary_path + hyper_result_path[4] + '\\' + model_name[4] + '.h5')
    model0_adam3 = load_model(summary_path + hyper_result_path[5] + '\\' + model_name[5] + '.h5')
    model0_sgd2 = load_model(summary_path + hyper_result_path[6] + '\\' + model_name[6] + '.h5')

    return model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam2,model0_adam3,model0_sgd2


batch_size = 32

import os 
os.listdir('model_saved_new1')

model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']

# result_path  = ''
summary_path = 'model_saved_new1\\'

testY_data = np.load(summary_path + 'testY_' + '.npy')
trainY_data = np.load(summary_path + 'trainY_' + '.npy')
testX_data = np.load(summary_path + 'testX_' + '.npy')
trainX_data = np.load(summary_path + 'trainX_' + '.npy')

summary_path0 = 'model_saved_new1\\' + model_list[0]+'\\'
summary_path1 = 'model_saved_new1\\' + model_list[1]+'\\'
summary_path2 = 'model_saved_new1\\' + model_list[2]+'\\'
summary_path3 = 'model_saved_new1\\' + model_list[3]+'\\'
summary_path4 = 'model_saved_new1\\' + model_list[4]+'\\'
summary_path5 = 'model_saved_new1\\' + model_list[5]+'\\'
summary_path6 = 'model_saved_new1\\ensemble\\' + model_list[6]+'\\'


### take one path is enough 

### cannot use summary_path0, it have extra folder
# hyper_result_path = os.listdir(summary_path1)

def roc(summary_path, model,filename,modellist,testX_data,batch_size):
    predIdxs = model.predict(testX_data, batch_size=batch_size)
    print(predIdxs)

    # y_pred_proba = model0_adam0.predict_proba(testX_data)[:,1]
    fpr, tpr, _ = metrics.roc_curve(testY_data,  predIdxs)

    hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.title("ROC curve" + filename)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('model_saved_new1\\' + 'result\\ROC\\' + modellist + filename + '.png')
    plt.show()
    print('##########################################################################################################')

filename = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']
hyper = ['adam0','adam1','sgd0','sgd1','adam3','sgd2']
model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']


hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

### adam2 = adam0.01; adam3 = adam0.1 sgd2 = sgd0.01
### cannot load all not enough memory
#dense
model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam2, model0_adam3,model0_sgd2 = load_model_result(summary_path0,hyper_result_path,model_name0)
# ##inception
model_name0 = ['model-Dense201-19-0.9525','model-Dense201-23-0.9450','model-Dense201-39-0.8950','model-Dense201-38-0.8200','model-Dense201-13-0.9550','model-Dense201-32-0.9625','model-Dense201-31-0.9300']
model_name1 = ['model-InceptionV3-35-0.9075','model-InceptionV3-40-0.9150','model-InceptionV3-28-0.8400','model-InceptionV3-36-0.8300','model-InceptionV3-33-0.9075','model-InceptionV3-38-0.8975']
model_name2 = ['model-MobileV2-14-0.9500','model-MobileV2-11-0.9500','model-MobileV2-25-0.8975','model-MobileV2-36-0.8450','model-MobileV2-23-0.9450','model-MobileV2-18-0.9425']
model_name3 = ['model-Res101-36-0.9475','model-Res101-32-0.9350','model-Res101-32-0.8725','model-Res101-39-0.7975','model-Res101-31-0.9425','model-Res101-22-0.9225']
model_name4 = ['model-VGG16-28-0.9350','model-VGG16-40-0.8850','model-VGG16-07-0.7825','model-VGG16-39-0.7325','model-VGG16-32-0.9375','model-VGG16-36-0.8150']
model_name5 = ['model-Xception-22-0.9250','model-Xception-36-0.9225','model-Xception-38-0.8300','model-Xception-40-0.7700','model-Xception-17-0.9250','model-Xception-38-0.9025']
### cannot load all not enough memory
#dense
##inception
# model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam3,model0_sgd2 = load_model_result(summary_path1,hyper_result_path,model_name1)
# model2_adam0,model2_adam1,model2_sgd0,model2_sgd1, model2_adam3,model2_sgd2 = load_model_result(summary_path2,hyper_result_path,model_name2)
# model3_adam0,model3_adam1,model3_sgd0,model3_sgd1, model3_adam3,model3_sgd2 = load_model_result(summary_path3,hyper_result_path,model_name3)
# model4_adam0,model4_adam1,model4_sgd0,model4_sgd1, model4_adam3,model4_sgd2 = load_model_result(summary_path4,hyper_result_path,model_name4)
# model5_adam0,model5_adam1,model5_sgd0,model5_sgd1, model5_adam3,model5_sgd2 = load_model_result(summary_path5,hyper_result_path,model_name5)


roc(summary_path0, model0_adam0,filename[0],model_list[0],testX_data,32)
roc(summary_path0, model0_adam1,filename[1],model_list[0],testX_data,32)
roc(summary_path0, model0_adam2,filename[2],model_list[0],testX_data,32)
roc(summary_path0, model0_adam3,filename[3],model_list[0],testX_data,32)
roc(summary_path0, model0_sgd0,filename[4],model_list[0],testX_data,32)
roc(summary_path0, model0_sgd1,filename[5],model_list[0],testX_data,32)
roc(summary_path0, model0_sgd2,filename[6],model_list[0],testX_data,32)







# %% [markdown]
# roc all

# %%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

## this for dense model only
def load_model_result_dense(summary_path, hyper_result_path, model_name): 
    model0_adam0 = load_model(summary_path + hyper_result_path[1] + '\\' + model_name[0] + '.h5')
    model0_adam1 = load_model(summary_path + hyper_result_path[0] + '\\' + model_name[1] + '.h5')
    model0_sgd0 = load_model(summary_path + hyper_result_path[3] + '\\' + model_name[2] + '.h5')
    model0_sgd1 = load_model(summary_path + hyper_result_path[2] + '\\' + model_name[3] + '.h5')
    model0_adam2 = load_model(summary_path + hyper_result_path[4] + '\\' + model_name[4] + '.h5')
    model0_adam3 = load_model(summary_path + hyper_result_path[5] + '\\' + model_name[5] + '.h5')
    model0_sgd2 = load_model(summary_path + hyper_result_path[6] + '\\' + model_name[6] + '.h5')

    return model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam2,model0_adam3,model0_sgd2


batch_size = 32

import os 
os.listdir('model_saved_new1')

model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']

# result_path  = ''
summary_path = 'model_saved_new1\\'

testY_data = np.load(summary_path + 'testY_' + '.npy')
trainY_data = np.load(summary_path + 'trainY_' + '.npy')
testX_data = np.load(summary_path + 'testX_' + '.npy')
trainX_data = np.load(summary_path + 'trainX_' + '.npy')

summary_path0 = 'model_saved_new1\\' + model_list[0]+'\\'
summary_path1 = 'model_saved_new1\\' + model_list[1]+'\\'
summary_path2 = 'model_saved_new1\\' + model_list[2]+'\\'
summary_path3 = 'model_saved_new1\\' + model_list[3]+'\\'
summary_path4 = 'model_saved_new1\\' + model_list[4]+'\\'
summary_path5 = 'model_saved_new1\\' + model_list[5]+'\\'
summary_path6 = 'model_saved_new1\\ensemble\\' + model_list[6]+'\\'


### take one path is enough 

### cannot use summary_path0, it have extra folder
# hyper_result_path = os.listdir(summary_path1)

def roc(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam2, model0_adam3,model0_sgd2,testX_data,batch_size,model_list):
    predIdxs0 = model0_adam0.predict(testX_data, batch_size=batch_size)
    predIdxs1 = model0_adam1.predict(testX_data, batch_size=batch_size)
    predIdxs2 = model0_adam2.predict(testX_data, batch_size=batch_size)
    predIdxs3 = model0_adam3.predict(testX_data, batch_size=batch_size)
    predIdxs4 = model0_sgd0.predict(testX_data, batch_size=batch_size)
    predIdxs5 = model0_sgd1.predict(testX_data, batch_size=batch_size)
    predIdxs6 = model0_sgd2.predict(testX_data, batch_size=batch_size)

    fpr0, tpr0, _ = metrics.roc_curve(testY_data,  predIdxs0)
    fpr1, tpr1, _ = metrics.roc_curve(testY_data,  predIdxs1)
    fpr2, tpr2, _ = metrics.roc_curve(testY_data,  predIdxs2)
    fpr3, tpr3, _ = metrics.roc_curve(testY_data,  predIdxs3)
    fpr4, tpr4, _ = metrics.roc_curve(testY_data,  predIdxs4)
    fpr5, tpr5, _ = metrics.roc_curve(testY_data,  predIdxs5)
    fpr6, tpr6, _ = metrics.roc_curve(testY_data,  predIdxs6)
    hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']
    #create ROC curve
    plt.plot(fpr0,tpr0,label=hyper_result_path[1])
    plt.plot(fpr1,tpr1,label=hyper_result_path[0])
    plt.plot(fpr2,tpr2,label=hyper_result_path[4])
    plt.plot(fpr3,tpr3,label=hyper_result_path[5])
    plt.plot(fpr4,tpr4,label=hyper_result_path[3])
    plt.plot(fpr5,tpr5,label=hyper_result_path[2])
    plt.plot(fpr6,tpr6,label=hyper_result_path[6])
    plt.legend(loc='best')
    plt.title("ROC curve: " + model_list)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='best')
    plt.savefig('model_saved_new1\\' + 'result\\ROC\\' + model_list + '.png')
    plt.show()
    print('##########################################################################################################')

def load_model_result(summary_path, hyper_result_path, model_name): 
    model0_adam0 = load_model(summary_path + hyper_result_path[1] + '\\' + model_name[0] + '.h5')
    model0_adam1 = load_model(summary_path + hyper_result_path[0] + '\\' + model_name[1] + '.h5')
    model0_sgd0 = load_model(summary_path + hyper_result_path[3] + '\\' + model_name[2] + '.h5')
    model0_sgd1 = load_model(summary_path + hyper_result_path[2] + '\\' + model_name[3] + '.h5')
    # model0_adam2 = load_model(summary_path + hyper_result_path[4] + '\\' + model_name[4] + '.h5')
    model0_adam3 = load_model(summary_path + hyper_result_path[5] + '\\' + model_name[4] + '.h5')
    model0_sgd2 = load_model(summary_path + hyper_result_path[6] + '\\' + model_name[5] + '.h5')

    return model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam3,model0_sgd2


def roc_exceptdense(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam3,model0_sgd2,testX_data,batch_size,model_list):

    predIdxs0 = model0_adam0.predict(testX_data, batch_size=batch_size)
    predIdxs1 = model0_adam1.predict(testX_data, batch_size=batch_size)
    # predIdxs2 = model0_adam2.predict(testX_data, batch_size=batch_size)
    predIdxs3 = model0_adam3.predict(testX_data, batch_size=batch_size)
    predIdxs4 = model0_sgd0.predict(testX_data, batch_size=batch_size)
    predIdxs5 = model0_sgd1.predict(testX_data, batch_size=batch_size)
    predIdxs6 = model0_sgd2.predict(testX_data, batch_size=batch_size)

    # predIdxs = [predIdxs0,predIdxs1,predIdxs2,predIdxs3,predIdxs4,predIdxs5,predIdxs6]

    # y_pred_proba = model0_adam0.predict_proba(testX_data)[:,1]

    fpr0, tpr0, _ = metrics.roc_curve(testY_data,  predIdxs0)
    fpr1, tpr1, _ = metrics.roc_curve(testY_data,  predIdxs1)
    # fpr2, tpr2, _ = metrics.roc_curve(testY_data,  predIdxs2)
    fpr3, tpr3, _ = metrics.roc_curve(testY_data,  predIdxs3)
    fpr4, tpr4, _ = metrics.roc_curve(testY_data,  predIdxs4)
    fpr5, tpr5, _ = metrics.roc_curve(testY_data,  predIdxs5)
    fpr6, tpr6, _ = metrics.roc_curve(testY_data,  predIdxs6)


    hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

    #create ROC curve
    plt.plot(fpr0,tpr0,label=hyper_result_path[1])
    plt.plot(fpr1,tpr1,label=hyper_result_path[0])
    # plt.plot(fpr2,tpr2,label=hyper_result_path[4])
    plt.plot(fpr3,tpr3,label=hyper_result_path[5])
    plt.plot(fpr4,tpr4,label=hyper_result_path[3])
    plt.plot(fpr5,tpr5,label=hyper_result_path[2])
    plt.plot(fpr6,tpr6,label=hyper_result_path[6])
    plt.title("ROC curve: " + model_list)
    plt.legend(loc='best')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('model_saved_new1\\' + 'result\\ROC\\' + model_list + '.png')
    plt.show()
    print('##########################################################################################################')


filename = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']
hyper = ['adam0','adam1','sgd0','sgd1','adam3','sgd2']
model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']


hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

### adam2 = adam0.01; adam3 = adam0.1 sgd2 = sgd0.01
### cannot load all not enough memory
#dense
model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam2, model0_adam3,model0_sgd2 = load_model_result_dense(summary_path0,hyper_result_path,model_name0)
# ##inception
model_name0 = ['model-Dense201-19-0.9525','model-Dense201-23-0.9450','model-Dense201-39-0.8950','model-Dense201-38-0.8200','model-Dense201-13-0.9550','model-Dense201-32-0.9625','model-Dense201-31-0.9300']
model_name1 = ['model-InceptionV3-35-0.9075','model-InceptionV3-40-0.9150','model-InceptionV3-28-0.8400','model-InceptionV3-36-0.8300','model-InceptionV3-33-0.9075','model-InceptionV3-38-0.8975']
model_name2 = ['model-MobileV2-14-0.9500','model-MobileV2-11-0.9500','model-MobileV2-25-0.8975','model-MobileV2-36-0.8450','model-MobileV2-23-0.9450','model-MobileV2-18-0.9425']
model_name3 = ['model-Res101-36-0.9475','model-Res101-32-0.9350','model-Res101-32-0.8725','model-Res101-39-0.7975','model-Res101-31-0.9425','model-Res101-22-0.9225']
model_name4 = ['model-VGG16-28-0.9350','model-VGG16-40-0.8850','model-VGG16-07-0.7825','model-VGG16-39-0.7325','model-VGG16-32-0.9375','model-VGG16-36-0.8150']
model_name5 = ['model-Xception-22-0.9250','model-Xception-36-0.9225','model-Xception-38-0.8300','model-Xception-40-0.7700','model-Xception-17-0.9250','model-Xception-38-0.9025']
### cannot load all not enough memory
##inception
# model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam3,model0_sgd2 = load_model_result(summary_path1,hyper_result_path,model_name1)
# model2_adam0,model2_adam1,model2_sgd0,model2_sgd1, model2_adam3,model2_sgd2 = load_model_result(summary_path2,hyper_result_path,model_name2)
# model3_adam0,model3_adam1,model3_sgd0,model3_sgd1, model3_adam3,model3_sgd2 = load_model_result(summary_path3,hyper_result_path,model_name3)
# model4_adam0,model4_adam1,model4_sgd0,model4_sgd1, model4_adam3,model4_sgd2 = load_model_result(summary_path4,hyper_result_path,model_name4)
# model5_adam0,model5_adam1,model5_sgd0,model5_sgd1, model5_adam3,model5_sgd2 = load_model_result(summary_path5,hyper_result_path,model_name5)

# roc_exceptdense(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam3,model0_sgd2,testX_data,batch_size,model_list[1])
# roc_exceptdense(model2_adam0,model2_adam1,model2_sgd0,model2_sgd1, model2_adam3,model2_sgd2,testX_data,batch_size,model_list[2])


roc(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam2, model0_adam3,model0_sgd2,testX_data,batch_size,model_list[0])
# roc_exceptdense(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam3,model0_sgd2,testX_data,batch_size,model_list[1])
# roc_exceptdense(model2_adam0,model2_adam1,model2_sgd0,model2_sgd1, model2_adam3,model2_sgd2,testX_data,batch_size,model_list[2])
# roc_exceptdense(model3_adam0,model3_adam1,model3_sgd0,model3_sgd1, model3_adam3,model3_sgd2,testX_data,batch_size,model_list[3])
# roc_exceptdense(model4_adam0,model4_adam1,model4_sgd0,model4_sgd1, model4_adam3,model4_sgd2,testX_data,batch_size,model_list[4])
# roc_exceptdense(model5_adam0,model5_adam1,model5_sgd0,model5_sgd1, model5_adam3,model5_sgd2,testX_data,batch_size,model_list[5])


# %% [markdown]
# Confusion Matrix with correct version for other model except densenet different hyper

# %%
def load_model_result(summary_path, hyper_result_path, model_name): 
    model0_adam0 = load_model(summary_path + hyper_result_path[1] + '\\' + model_name[0] + '.h5')
    model0_adam1 = load_model(summary_path + hyper_result_path[0] + '\\' + model_name[1] + '.h5')
    model0_sgd0 = load_model(summary_path + hyper_result_path[3] + '\\' + model_name[2] + '.h5')
    model0_sgd1 = load_model(summary_path + hyper_result_path[2] + '\\' + model_name[3] + '.h5')
    # model0_adam2 = load_model(summary_path + hyper_result_path[4] + '\\' + model_name[4] + '.h5')
    model0_adam3 = load_model(summary_path + hyper_result_path[5] + '\\' + model_name[4] + '.h5')
    model0_sgd2 = load_model(summary_path + hyper_result_path[6] + '\\' + model_name[5] + '.h5')

    return model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam3,model0_sgd2

def confusion_plot(model, filename):
    batch_size = 32
    # make predications on the testing set
    print("evaluating network for " + filename + ".....")
    predIdxs = model.predict(testX_data, batch_size=batch_size)

    threshold = 0.5

    # Convert predicted probabilities to binary values using threshold
    predIdxs = (predIdxs > threshold).astype(int)


    #### for confusion matrix
    # compute the confusion matrix nad use it 
    # to drive the raw accuracy, sensitivity 
    # and specificity
    # cm = confusion_matrix(testY_data.argmax(axis=1), predIdxs)

    cm = confusion_matrix(testY_data, predIdxs)

    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]

    print('True positive = ', tp)
    print('True negative = ', tn)
    print('False positive = ', fp)
    print('False negative = ', fn)

    total_test = sum(sum(cm))
    accurancy = (cm[0, 0] + cm[1, 1]) / total_test	### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])	### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])	### Specificity (true negatives / all actual negatives) =TN / TN + FP
    f1score = 2*(precision*sensitivity)/(precision+sensitivity)
    npv = tn/(tn+fn)



    # show the cnfusion matrix, accuracy with 4 digits
    #sensitivity, and specificity
    print(cm)
    print("accurancy: {:.4f}".format(accurancy))
    print("precision: {:.4f}".format(precision))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    print("f1-score: {:.4f}".format(f1score))
    print("NPV: {:.4f}".format(npv))



    import seaborn as sns
    ### fmt ='g' to turn off scientific number 
    sns.heatmap(cm, cmap="Blues",annot=True, fmt='g',xticklabels=["Normal", "Covid-19"], yticklabels=["Normal", "Covid-19"])
    plt.title("Confusion Matrix: " + filename)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.savefig(summary_path + 'result\\' + filename + '.png')
    plt.show()
    print('##########################################################################################################')

    accurancy = "%.4f" % accurancy
    precision = "%.4f" % precision
    sensitivity = "%.4f" % sensitivity
    specificity = "%.4f" % specificity
    f1score = "%.4f" % f1score
    npv = "%.4f" % npv

    evaluation_result = [accurancy,precision,sensitivity,specificity,f1score,npv]
    return evaluation_result

def different_model_evaluation(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam3,model0_sgd2,model_list, filename):
    evaluation_result_name = ['Adam_0.0001', 'Adam_0.001', 'SGD_0.0001', 'SGD_0.001','Adam_0.1','Adam_0.01','SGD_0.01']
    ### hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']
    # evaluation_result_name = ['Adam_0.01','Adam_0.001', 'Adam_0.0001','SGD_0.01', 'SGD_0.001', 'SGD_0.0001',]

    evaluation_result_adam0 = [evaluation_result_name[1]] + confusion_plot(model0_adam0, model_list + '_' + hyper_result_path[1])
    evaluation_result_adam1 = [evaluation_result_name[0]] + confusion_plot(model0_adam1, model_list + '_' + hyper_result_path[0])
    evaluation_result_sgd0 = [evaluation_result_name[3]] + confusion_plot(model0_sgd0, model_list + '_' + hyper_result_path[3])
    evaluation_result_sgd1 = [evaluation_result_name[2]] + confusion_plot(model0_sgd1, model_list + '_' + hyper_result_path[2])
    # evaluation_result_adam2 = [evaluation_result_name[4]] + confusion_plot(model0_adam2, model_list + '_' + hyper_result_path[4])
    evaluation_result_adam3 = [evaluation_result_name[5]] + confusion_plot(model0_adam3, model_list + '_' + hyper_result_path[5])
    evaluation_result_sgd2 = [evaluation_result_name[6]] + confusion_plot(model0_sgd2, model_list + '_' + hyper_result_path[6])


    ### evaluation_result_adam0 = ['Adam_0.001', '0.9525', '0.9250', '0.9788', '0.9289', '0.9512', '0.9788']

    import csv
    evaluation_method = ['Method','Accurancy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'NPV']
    evaluation_result = [evaluation_result_adam3,evaluation_result_adam0,evaluation_result_adam1,evaluation_result_sgd2,evaluation_result_sgd0,evaluation_result_sgd1]

    with open(summary_path + 'result\\' + filename, 'w') as f:    
        csv_writer = csv.writer(f)
        csv_writer.writerow(evaluation_method)
        csv_writer.writerows(evaluation_result)
    f.close


# %%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32

import os 
# os.listdir('model_saved_new1')

model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']

# result_path  = ''
summary_path = 'model_saved_new1\\'

testY_data = np.load(summary_path + 'testY_' + '.npy')
trainY_data = np.load(summary_path + 'trainY_' + '.npy')
testX_data = np.load(summary_path + 'testX_' + '.npy')
trainX_data = np.load(summary_path + 'trainX_' + '.npy')

summary_path0 = 'model_saved_new1\\' + model_list[0]+'\\'
summary_path1 = 'model_saved_new1\\' + model_list[1]+'\\'
summary_path2 = 'model_saved_new1\\' + model_list[2]+'\\'
summary_path3 = 'model_saved_new1\\' + model_list[3]+'\\'
summary_path4 = 'model_saved_new1\\' + model_list[4]+'\\'
summary_path5 = 'model_saved_new1\\' + model_list[5]+'\\'
summary_path6 = 'model_saved_new1\\ensemble\\' + model_list[6]+'\\'


### take one path is enough 

### cannot use summary_path0, it have extra folder
# hyper_result_path = os.listdir(summary_path1)
hyper_result_path = ['adam0.0001', 'adam0.001', 'sgd0.0001', 'sgd0.001', 'adam0.1','adam0.01','sgd0.01']

### model_name0 = ['adam0.001', 'adam0.0001', 'sgd0.001', 'sgd0.0001']
model_name0 = ['model-Dense201-19-0.9525','model-Dense201-23-0.9450','model-Dense201-39-0.8950','model-Dense201-38-0.8200','model-Dense201-13-0.9550','model-Dense201-32-0.9625','model-Dense201-31-0.9300']
model_name1 = ['model-InceptionV3-35-0.9075','model-InceptionV3-40-0.9150','model-InceptionV3-28-0.8400','model-InceptionV3-36-0.8300','model-InceptionV3-33-0.9075','model-InceptionV3-38-0.8975']
model_name2 = ['model-MobileV2-14-0.9500','model-MobileV2-11-0.9500','model-MobileV2-25-0.8975','model-MobileV2-36-0.8450','model-MobileV2-23-0.9450','model-MobileV2-18-0.9425']
model_name3 = ['model-Res101-36-0.9475','model-Res101-32-0.9350','model-Res101-32-0.8725','model-Res101-39-0.7975','model-Res101-31-0.9425','model-Res101-22-0.9225']
model_name4 = ['model-VGG16-28-0.9350','model-VGG16-40-0.8850','model-VGG16-07-0.7825','model-VGG16-39-0.7325','model-VGG16-32-0.9375','model-VGG16-36-0.8150']
model_name5 = ['model-Xception-22-0.9250','model-Xception-36-0.9225','model-Xception-38-0.8300','model-Xception-40-0.7700','model-Xception-17-0.9250','model-Xception-38-0.9025']


### adam2 = adam0.01; adam3 = adam0.1 sgd2 = sgd0.01
### cannot load all not enough memory
#dense
### model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam2, model0_adam3,model0_sgd2 = load_model_result(summary_path0,hyper_result_path,model_name0)
# ##inception
# model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam3,model0_sgd2 = load_model_result(summary_path1,hyper_result_path,model_name1)

# model2_adam0,model2_adam1,model2_sgd0,model2_sgd1, model2_adam3,model2_sgd2 = load_model_result(summary_path2,hyper_result_path,model_name2)

# model3_adam0,model3_adam1,model3_sgd0,model3_sgd1, model3_adam3,model3_sgd2 = load_model_result(summary_path3,hyper_result_path,model_name3)

model4_adam0,model4_adam1,model4_sgd0,model4_sgd1, model4_adam3,model4_sgd2 = load_model_result(summary_path4,hyper_result_path,model_name4)
model5_adam0,model5_adam1,model5_sgd0,model5_sgd1, model5_adam3,model5_sgd2 = load_model_result(summary_path5,hyper_result_path,model_name5)
# model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception']
### different_model_evaluation(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model0_adam2,model0_adam3,model0_sgd2,model_list[0],model_list[0]+'_evaluation.csv')
# different_model_evaluation(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1, model0_adam3,model0_sgd2,model_list[1],model_list[1]+'_evaluation.csv')
# different_model_evaluation(model2_adam0,model2_adam1,model2_sgd0,model2_sgd1, model2_adam3,model2_sgd2,model_list[2],model_list[2]+'_evaluation.csv')
# different_model_evaluation(model3_adam0,model3_adam1,model3_sgd0,model3_sgd1, model3_adam3,model3_sgd2,model_list[3],model_list[3]+'_evaluation.csv')
different_model_evaluation(model4_adam0,model4_adam1,model4_sgd0,model4_sgd1, model4_adam3,model4_sgd2,model_list[4],model_list[4]+'_evaluation.csv')
different_model_evaluation(model5_adam0,model5_adam1,model5_sgd0,model5_sgd1, model5_adam3,model5_sgd2,model_list[5],model_list[5]+'_evaluation.csv')


# %% [markdown]
# stop here

# %% [markdown]
# data_aug result
# 

# %%
#### for loss and accuracy curve
import matplotlib.pyplot as plt
import json
import pandas as pd

# model_history_json_name = summary_path +  result_path + model_name + '_history.json'
# history_json = json.load(open(model_history_json_name, 'r'))

def history_model(summary_path, hyper_result_path,model_name):
    history = summary_path + hyper_result_path + '\\' + model_name + '_history.json'
    history_json = json.load(open(history, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1
    return history_df


summary_path = 'model_saved_new1\\dense\\data_aug\\'
hyper_result_path = ['com_1','com_2','com_3','com_4','com_5','com_6','com_7']
model_name0 = ['model-Dense201-39-0.8950']


history_df_model0_adam0 = history_model(summary_path, hyper_result_path[0], model_name0[0])
history_df_model1_adam0 = history_model(summary_path, hyper_result_path[1], model_name0[0])
history_df_model2_adam0 = history_model(summary_path, hyper_result_path[2], model_name0[0])
history_df_model3_adam0 = history_model(summary_path, hyper_result_path[3], model_name0[0])
history_df_model4_adam0 = history_model(summary_path, hyper_result_path[4], model_name0[0])
history_df_model5_adam0 = history_model(summary_path, hyper_result_path[5], model_name0[0])
history_df_model6_adam0 = history_model(summary_path, hyper_result_path[6], model_name0[0])



# model_list_name = ['1st Set','2nd Set', '3rd Set','4th Set','5th Set','6th Set','7th Set']
model_list_name = ['1st set of Data Augmentation','2nd set of Data Augmentation','3rd set of Data Augmentation','4th set of Data Augmentation','5th set of Data Augmentation','6th set of Data Augmentation','Without Data Augmentation']


plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['loss'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['loss'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['loss'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['loss'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['loss'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['loss'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['loss'],label=model_list_name[6])

# plt.xlim(-10, 10)
plt.ylim(0,1)
plt.legend(loc="upper right")
plt.title("Tranning Loss for different data augmentation techniques")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
# annot_max(history_df_adam0['epochs'],history_df_adam0['loss'])
plt.savefig(summary_path + 'result\\' + 'training_loss_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['val_loss'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['val_loss'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['val_loss'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['val_loss'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['val_loss'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['val_loss'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['val_loss'],label=model_list_name[6])

plt.ylim(0,1)
plt.legend(loc="upper right")
plt.title("Validation Loss for different data augmentation techniques")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
# annot_max(history_df_adam0['epochs'],history_df_adam0['loss'])
plt.savefig(summary_path + 'result\\' + 'validation_loss_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['accuracy'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['accuracy'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['accuracy'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['accuracy'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['accuracy'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['accuracy'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['accuracy'],label=model_list_name[6])

# plt.xlim(-10, 10)
plt.ylim(0,1)
plt.legend(loc="lower right")
plt.title("Tranning accuracy for different data augmentation techniques")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
# annot_max(history_df_adam0['epochs'],history_df_adam0['accuracy'])
plt.savefig(summary_path + 'result\\' + 'training_accuracy_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['val_accuracy'],label=model_list_name[0])
plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['val_accuracy'],label=model_list_name[1])
plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['val_accuracy'],label=model_list_name[2])
plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['val_accuracy'],label=model_list_name[3])
plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['val_accuracy'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['val_accuracy'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['val_accuracy'],label=model_list_name[6])

plt.ylim(0,1)
plt.legend(loc="lower right")
plt.title("Validation accuracy for different data augmentation techniques")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
# annot_max(history_df_adam0['epochs'],history_df_adam0['accuracy'])
plt.savefig(summary_path + 'result\\' + 'validation_accuracy_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()






# %% [markdown]
# data aug best and without compare only

# %%
#### for loss and accuracy curve
import matplotlib.pyplot as plt
import json
import pandas as pd

# model_history_json_name = summary_path +  result_path + model_name + '_history.json'
# history_json = json.load(open(model_history_json_name, 'r'))

def history_model(summary_path, hyper_result_path,model_name):
    history = summary_path + hyper_result_path + '\\' + model_name + '_history.json'
    history_json = json.load(open(history, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1
    return history_df


summary_path = 'model_saved_new1\\dense\\data_aug\\'
hyper_result_path = ['com_1','com_2','com_3','com_4','com_5','com_6','com_7']
model_name0 = ['model-Dense201-39-0.8950']


# history_df_model0_adam0 = history_model(summary_path, hyper_result_path[0], model_name0[0])
# history_df_model1_adam0 = history_model(summary_path, hyper_result_path[1], model_name0[0])
# history_df_model2_adam0 = history_model(summary_path, hyper_result_path[2], model_name0[0])
# history_df_model3_adam0 = history_model(summary_path, hyper_result_path[3], model_name0[0])
# history_df_model4_adam0 = history_model(summary_path, hyper_result_path[4], model_name0[0])
history_df_model5_adam0 = history_model(summary_path, hyper_result_path[5], model_name0[0])
history_df_model6_adam0 = history_model(summary_path, hyper_result_path[6], model_name0[0])



# model_list_name = ['1st Set','2nd Set', '3rd Set','4th Set','5th Set','6th Set','7th Set']
model_list_name = ['1st set of Data Augmentation','2nd set of Data Augmentation','3rd set of Data Augmentation','4th set of Data Augmentation','5th set of Data Augmentation','Applied Data Augmentation','Without Data Augmentation']


plt.style.use("ggplot")
plt.figure()
plt.plot()
# plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['loss'],label=model_list_name[0])
# plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['loss'],label=model_list_name[1])
# plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['loss'],label=model_list_name[2])
# plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['loss'],label=model_list_name[3])
# plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['loss'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['loss'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['loss'],label=model_list_name[6])

# plt.xlim(-10, 10)
plt.ylim(0,1)
plt.legend(loc="upper right")
plt.title("Tranning Loss: \n Without vs Applied Data Augmentation")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
# annot_max(history_df_adam0['epochs'],history_df_adam0['loss'])
plt.savefig(summary_path + 'result\\' + 'training_loss_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot()
# plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['val_loss'],label=model_list_name[0])
# plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['val_loss'],label=model_list_name[1])
# plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['val_loss'],label=model_list_name[2])
# plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['val_loss'],label=model_list_name[3])
# plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['val_loss'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['val_loss'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['val_loss'],label=model_list_name[6])

plt.ylim(0,1)
plt.legend(loc="upper right")
plt.title("Validation Loss: \n Without vs Applied Data Augmentation")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
# annot_max(history_df_adam0['epochs'],history_df_adam0['loss'])
plt.savefig(summary_path + 'result\\' + 'validation_loss_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot()
# plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['accuracy'],label=model_list_name[0])
# plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['accuracy'],label=model_list_name[1])
# plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['accuracy'],label=model_list_name[2])
# plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['accuracy'],label=model_list_name[3])
# plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['accuracy'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['accuracy'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['accuracy'],label=model_list_name[6])

# plt.xlim(-10, 10)
plt.ylim(0,1)
plt.legend(loc="lower right")
plt.title("Tranning accuracy: \n Without vs Applied Data Augmentation")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
# annot_max(history_df_adam0['epochs'],history_df_adam0['accuracy'])
plt.savefig(summary_path + 'result\\' + 'training_accuracy_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot()
# plt.plot(history_df_model0_adam0['epochs'],history_df_model0_adam0['val_accuracy'],label=model_list_name[0])
# plt.plot(history_df_model1_adam0['epochs'],history_df_model1_adam0['val_accuracy'],label=model_list_name[1])
# plt.plot(history_df_model2_adam0['epochs'],history_df_model2_adam0['val_accuracy'],label=model_list_name[2])
# plt.plot(history_df_model3_adam0['epochs'],history_df_model3_adam0['val_accuracy'],label=model_list_name[3])
# plt.plot(history_df_model4_adam0['epochs'],history_df_model4_adam0['val_accuracy'],label=model_list_name[4])
plt.plot(history_df_model5_adam0['epochs'],history_df_model5_adam0['val_accuracy'],label=model_list_name[5])
plt.plot(history_df_model6_adam0['epochs'],history_df_model6_adam0['val_accuracy'],label=model_list_name[6])

plt.ylim(0,1)
plt.legend(loc="lower right")
plt.title("Validation accuracy: \n Without vs Applied Data Augmentation")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
# annot_max(history_df_adam0['epochs'],history_df_adam0['accuracy'])
plt.savefig(summary_path + 'result\\' + 'validation_accuracy_' +'different_data_augmentation1' + '.png', bbox_inches = 'tight')
plt.show()




# %% [markdown]
# data aug for cm 

# %%
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt



# result_path  = ''
summary_path = 'model_saved_new1\\'

testY_data = np.load(summary_path + 'testY_' + '.npy')
trainY_data = np.load(summary_path + 'trainY_' + '.npy')
testX_data = np.load(summary_path + 'testX_' + '.npy')
trainX_data = np.load(summary_path + 'trainX_' + '.npy')


summary_path = 'model_saved_new1\\dense\\data_aug\\'
hyper_result_path = ['com_1','com_2','com_3','com_4','com_5','com_6','com_7']
model_name = ['model-Dense201-19-0.9375','model-Dense201-27-0.9325','model-Dense201-23-0.9375','model-Dense201-28-0.9450','model-Dense201-23-0.9500','model-Dense201-19-0.9525','model-Dense201-03-0.9225']



model0 = load_model(summary_path + hyper_result_path[0] + '\\' + model_name[0] + '.h5')
model1 = load_model(summary_path + hyper_result_path[1] + '\\' + model_name[1] + '.h5')
model2 = load_model(summary_path + hyper_result_path[2] + '\\' + model_name[2] + '.h5')
model3 = load_model(summary_path + hyper_result_path[3] + '\\' + model_name[3] + '.h5')
model4 = load_model(summary_path + hyper_result_path[4] + '\\' + model_name[4] + '.h5')
model5 = load_model(summary_path + hyper_result_path[5] + '\\' + model_name[5] + '.h5')
model6 = load_model(summary_path + hyper_result_path[6] + '\\' + model_name[6] + '.h5')


def confusion_plot(model, filename,i):
    batch_size = 32
    # make predications on the testing set
    print("evaluating network for " + filename + ".....")
    predIdxs = model.predict(testX_data, batch_size=batch_size)

    threshold = 0.5

    # Convert predicted probabilities to binary values using threshold
    predIdxs = (predIdxs > threshold).astype(int)


    #### for confusion matrix
    # compute the confusion matrix nad use it 
    # to drive the raw accuracy, sensitivity 
    # and specificity
    # cm = confusion_matrix(testY_data.argmax(axis=1), predIdxs)

    cm = confusion_matrix(testY_data, predIdxs)

    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]

    print('True positive = ', tp)
    print('True negative = ', tn)
    print('False positive = ', fp)
    print('False negative = ', fn)

    total_test = sum(sum(cm))
    accurancy = (cm[0, 0] + cm[1, 1]) / total_test	### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])	### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])	### Specificity (true negatives / all actual negatives) =TN / TN + FP
    f1score = 2*(precision*sensitivity)/(precision+sensitivity)
    npv = tn/(tn+fn)

    ### original wrong version
    # total_test = sum(sum(cm))
    # accurancy = (cm[0, 0] + cm[1, 1]) / total_test	### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    # # precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP
    # precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP


    # sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0])	### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    # specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])	### Specificity (true negatives / all actual negatives) =TN / TN + FP
    # f1score = 2*(precision*sensitivity)/(precision+sensitivity)
    # npv = tn/(tn+fn)

    # show the cnfusion matrix, accuracy with 4 digits
    #sensitivity, and specificity
    print(cm)
    print("accurancy: {:.4f}".format(accurancy))
    print("precision: {:.4f}".format(precision))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    print("f1-score: {:.4f}".format(f1score))
    print("NPV: {:.4f}".format(npv))



    import seaborn as sns
    ### fmt ='g' to turn off scientific number 
    sns.heatmap(cm, cmap="Blues",annot=True, fmt='g',xticklabels=["Normal", "Covid-19"], yticklabels=["Normal", "Covid-19"])
    namelist = ['1st set of Data Augmentation','2nd set of Data Augmentation','3rd set of Data Augmentation','4th set of Data Augmentation','5th set of Data Augmentation','6th set of Data Augmentation','Without Data Augmentation']
    plt.title("Confusion Matrix: \n" + namelist[i])
    plt.xlabel("Prediction")
    plt.ylabel("Growth Truth")
    plt.savefig(summary_path + 'result\\' + filename + '_new.png')
    plt.show()
    print('##########################################################################################################')
    
    accurancy = "%.4f" % accurancy
    precision = "%.4f" % precision
    sensitivity = "%.4f" % sensitivity
    specificity = "%.4f" % specificity
    f1score = "%.4f" % f1score
    npv = "%.4f" % npv

    evaluation_result = [accurancy,precision,sensitivity,specificity,f1score,npv]
    return evaluation_result

evaluation_result_name = ['1st Set','2nd Set', '3rd Set','4th Set','5th Set','6th Set','7th Set']
model_list = ''

# evaluation_result0 = [evaluation_result_name[0]] + confusion_plot(model0, model_list + '_' + hyper_result_path[0],0)
# evaluation_result1 = [evaluation_result_name[1]] + confusion_plot(model1, model_list + '_' + hyper_result_path[1],1)
# evaluation_result2 = [evaluation_result_name[2]] + confusion_plot(model2, model_list + '_' + hyper_result_path[2],2)
# evaluation_result3 = [evaluation_result_name[3]] + confusion_plot(model3, model_list + '_' + hyper_result_path[3],3)
# evaluation_result4 = [evaluation_result_name[4]] + confusion_plot(model4, model_list + '_' + hyper_result_path[4],4)
# evaluation_result5 = [evaluation_result_name[5]] + confusion_plot(model5, model_list + '_' + hyper_result_path[5],5)
evaluation_result6 = [evaluation_result_name[6]] + confusion_plot(model6, model_list + '_' + hyper_result_path[6],6)



# import csv
# evaluation_method = ['Method','Accurancy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'NPV']
# evaluation_result =[evaluation_result0,evaluation_result1,evaluation_result2,evaluation_result3,evaluation_result4,evaluation_result5,evaluation_result6]
# filename = evaluation_result_name

# for i in range(7):
#     with open(summary_path + 'result\\' + filename[i]+'_evaluation.csv', 'w') as f:    
#         csv_writer = csv.writer(f)
#         csv_writer.writerow(evaluation_method)
#         csv_writer.writerows(evaluation_result)
#     f.close


# # model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception']
# different_model_evaluation(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model_list[0],model_list[0]+'_evaluation.csv')
# different_model_evaluation(model1_adam0,model1_adam1,model1_sgd0,model1_sgd1,model_list[0],model_list[1]+'_evaluation.csv')
# different_model_evaluation(model2_adam0,model2_adam1,model2_sgd0,model2_sgd1,model_list[0],model_list[2]+'_evaluation.csv')
# different_model_evaluation(model3_adam0,model3_adam1,model3_sgd0,model3_sgd1,model_list[0],model_list[3]+'_evaluation.csv')
# different_model_evaluation(model4_adam0,model4_adam1,model4_sgd0,model4_sgd1,model_list[0],model_list[4]+'_evaluation.csv')
# different_model_evaluation(model5_adam0,model5_adam1,model5_sgd0,model5_sgd1,model_list[0],model_list[5]+'_evaluation.csv')

# %% [markdown]
# roc for individual model
# 

# %%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


batch_size = 32
import os 
model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception', 'ensemble4']

summary_path = 'model_saved_new1\\ensemble\\' 
testY_data = np.load(summary_path + 'testY_' + '.npy')
trainY_data = np.load(summary_path + 'trainY_' + '.npy')
testX_data = np.load(summary_path + 'testX_' + '.npy')
trainX_data = np.load(summary_path + 'trainX_' + '.npy')

model_name = ['model-Dense201-32-0.9625','model-InceptionV3-40-0.9150','model-MobileV2-11-0.9500','model-Res101-36-0.9475','model-VGG16-32-0.9375','model-Xception-17-0.9250']
model0 = load_model(summary_path + model_name[0] + '.h5')
model1 = load_model(summary_path + model_name[1] + '.h5')
model2 = load_model(summary_path + model_name[2] + '.h5')
model3 = load_model(summary_path + model_name[3] + '.h5')
model4 = load_model(summary_path + model_name[4] + '.h5')
model5 = load_model(summary_path + model_name[5] + '.h5')

def roc(model0,model1,model2,model3,model4,model5,testX_data,batch_size):

    predIdxs0 = model0.predict(testX_data, batch_size=batch_size)
    predIdxs1 = model1.predict(testX_data, batch_size=batch_size)
    predIdxs2 = model2.predict(testX_data, batch_size=batch_size)
    predIdxs3 = model2.predict(testX_data, batch_size=batch_size)
    predIdxs4 = model4.predict(testX_data, batch_size=batch_size)
    predIdxs5 = model5.predict(testX_data, batch_size=batch_size)

    # predIdxs = [predIdxs0,predIdxs1,predIdxs2,predIdxs3,predIdxs4,predIdxs5,predIdxs6]

    # y_pred_proba = model0_adam0.predict_proba(testX_data)[:,1]

    fpr0, tpr0, _ = metrics.roc_curve(testY_data,  predIdxs0)
    fpr1, tpr1, _ = metrics.roc_curve(testY_data,  predIdxs1)
    fpr2, tpr2, _ = metrics.roc_curve(testY_data,  predIdxs2)
    fpr3, tpr3, _ = metrics.roc_curve(testY_data,  predIdxs3)
    fpr4, tpr4, _ = metrics.roc_curve(testY_data,  predIdxs4)
    fpr5, tpr5, _ = metrics.roc_curve(testY_data,  predIdxs5)


    hyper_result_path = ['DenseNet_Adam_0.01','InceptionV3_Adam_0.0001','MobileNet_Adam_0.0001','ResNet101V2_Adam_0.001','VGG16_Adam_0.01','Xception_Adam_0.01', 'ensemble4']

    #create ROC curve
    plt.plot(fpr0,tpr0,label=hyper_result_path[0])
    plt.plot(fpr1,tpr1,label=hyper_result_path[1])
    plt.plot(fpr2,tpr2,label=hyper_result_path[2])
    plt.plot(fpr3,tpr3,label=hyper_result_path[3])
    plt.plot(fpr4,tpr4,label=hyper_result_path[4])
    plt.plot(fpr5,tpr5,label=hyper_result_path[5])
    plt.title("ROC curve: " + 'All best model')
    plt.legend(loc='best')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('model_saved_new1\\' + 'result\\ROC\\' + 'all_best_model' + '.png')
    plt.show()
    print('##########################################################################################################')


roc(model0,model1,model2,model3,model4,model5,testX_data,batch_size)



# %% [markdown]
# ensemble cm

# %%
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt



# result_path  = ''
summary_path = 'model_saved_new1\\ensemble1\\'
summary_path1 = 'model_saved_new1\\'

# testY_data = np.load(summary_path1 + 'testY_' + '.npy')
# trainY_data = np.load(summary_path1 + 'trainY_' + '.npy')
# testX_data = np.load(summary_path1 + 'testX_' + '.npy')
# trainX_data = np.load(summary_path1 + 'trainX_' + '.npy')


hyper_result_path = ['ensemble2','ensemble3','ensemble4','ensemble5','ensemble6','ensemble7']
model_name = ['model-ensemble-04-0.9700','model-ensemble3-40-0.9700','model-ensemble4-10-0.9725','model-ensemble5-16-0.9500' ,'model-ensemble6-15-0.9675']





# model0 = load_model(summary_path + hyper_result_path[0] + '\\' + model_name[0] + '.h5')
# model1 = load_model(summary_path + hyper_result_path[1] + '\\' + model_name[1] + '.h5')
# model2 = load_model(summary_path + hyper_result_path[2] + '\\' + model_name[2] + '.h5')
# model3 = load_model(summary_path + hyper_result_path[3] + '\\' + model_name[3] + '.h5')
# model4 = load_model(summary_path + hyper_result_path[4] + '\\' + model_name[4] + '.h5')
# model5 = load_model(summary_path + hyper_result_path[5] + '\\' + model_name[5] + '.h5')
# model6 = load_model(summary_path + hyper_result_path[6] + '\\' + model_name[6] + '.h5')


X = [trainX_data for _ in range(len(model0.input))]
X_1 = [testX_data for _ in range(len(model0.input))]

# 

def confusion_plot(model, filename,i, testX_data):
    batch_size = 32
    # make predications on the testing set
    print("evaluating network for " + filename + ".....")
    predIdxs = model.predict(testX_data, batch_size=batch_size)

    threshold = 0.5

    # Convert predicted probabilities to binary values using threshold
    predIdxs = (predIdxs > threshold).astype(int)


    #### for confusion matrix
    # compute the confusion matrix nad use it 
    # to drive the raw accuracy, sensitivity 
    # and specificity
    # cm = confusion_matrix(testY_data.argmax(axis=1), predIdxs)

    cm = confusion_matrix(testX_data, predIdxs)

    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]

    print('True positive = ', tp)
    print('True negative = ', tn)
    print('False positive = ', fp)
    print('False negative = ', fn)

    total_test = sum(sum(cm))
    accurancy = (cm[0, 0] + cm[1, 1]) / total_test	### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])	### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])	### Specificity (true negatives / all actual negatives) =TN / TN + FP
    f1score = 2*(precision*sensitivity)/(precision+sensitivity)
    npv = tn/(tn+fn)

    ### original wrong version
    # total_test = sum(sum(cm))
    # accurancy = (cm[0, 0] + cm[1, 1]) / total_test	### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    # # precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP
    # precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])    ### (true positives / predicted positives) = TP / TP + FP


    # sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0])	### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    # specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])	### Specificity (true negatives / all actual negatives) =TN / TN + FP
    # f1score = 2*(precision*sensitivity)/(precision+sensitivity)
    # npv = tn/(tn+fn)

    # show the cnfusion matrix, accuracy with 4 digits
    #sensitivity, and specificity
    print(cm)
    print("accurancy: {:.4f}".format(accurancy))
    print("precision: {:.4f}".format(precision))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    print("f1-score: {:.4f}".format(f1score))
    print("NPV: {:.4f}".format(npv))



    import seaborn as sns
    ### fmt ='g' to turn off scientific number 
    sns.heatmap(cm, cmap="Blues",annot=True, fmt='g',xticklabels=["Normal", "Covid-19"], yticklabels=["Normal", "Covid-19"])
    namelist = ['Ensemble Model 1','Ensemble Model 2','Ensemble Model 3','Ensemble Model 4','Ensemble Model 5']
    plt.title("Confusion Matrix: \n" + namelist[i])
    plt.xlabel("Prediction")
    plt.ylabel("Growth Truth")
    plt.savefig(summary_path + 'result\\' + filename + '_new.png')
    plt.show()
    print('##########################################################################################################')
    
    accurancy = "%.4f" % accurancy
    precision = "%.4f" % precision
    sensitivity = "%.4f" % sensitivity
    specificity = "%.4f" % specificity
    f1score = "%.4f" % f1score
    npv = "%.4f" % npv

    evaluation_result = [accurancy,precision,sensitivity,specificity,f1score,npv]
    return evaluation_result

evaluation_result_name = ['1st Set','2nd Set', '3rd Set','4th Set','5th Set','6th Set','7th Set']
model_list = ''

evaluation_result0 = [evaluation_result_name[0]] + confusion_plot(model0, model_list + '_' + hyper_result_path[0],0,X_1)
# evaluation_result1 = [evaluation_result_name[1]] + confusion_plot(model1, model_list + '_' + hyper_result_path[1],1,X_1)
# evaluation_result2 = [evaluation_result_name[2]] + confusion_plot(model2, model_list + '_' + hyper_result_path[2],2)
# evaluation_result3 = [evaluation_result_name[3]] + confusion_plot(model3, model_list + '_' + hyper_result_path[3],3)
# evaluation_result4 = [evaluation_result_name[4]] + confusion_plot(model4, model_list + '_' + hyper_result_path[4],4)
# evaluation_result5 = [evaluation_result_name[5]] + confusion_plot(model5, model_list + '_' + hyper_result_path[5],5)
# evaluation_result6 = [evaluation_result_name[6]] + confusion_plot(model6, model_list + '_' + hyper_result_path[6],6)



# import csv
# evaluation_method = ['Method','Accurancy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'NPV']
# evaluation_result =[evaluation_result0,evaluation_result1,evaluation_result2,evaluation_result3,evaluation_result4,evaluation_result5,evaluation_result6]
# filename = evaluation_result_name

# for i in range(7):
#     with open(summary_path + 'result\\' + filename[i]+'_evaluation.csv', 'w') as f:    
#         csv_writer = csv.writer(f)
#         csv_writer.writerow(evaluation_method)
#         csv_writer.writerows(evaluation_result)
#     f.close


# # model_list = ['dense','inceptionv3','mobile','resnet','vgg','xception']
# different_model_evaluation(model0_adam0,model0_adam1,model0_sgd0,model0_sgd1,model_list[0],model_list[0]+'_evaluation.csv')
# different_model_evaluation(model1_adam0,model1_adam1,model1_sgd0,model1_sgd1,model_list[0],model_list[1]+'_evaluation.csv')
# different_model_evaluation(model2_adam0,model2_adam1,model2_sgd0,model2_sgd1,model_list[0],model_list[2]+'_evaluation.csv')
# different_model_evaluation(model3_adam0,model3_adam1,model3_sgd0,model3_sgd1,model_list[0],model_list[3]+'_evaluation.csv')
# different_model_evaluation(model4_adam0,model4_adam1,model4_sgd0,model4_sgd1,model_list[0],model_list[4]+'_evaluation.csv')
# different_model_evaluation(model5_adam0,model5_adam1,model5_sgd0,model5_sgd1,model_list[0],model_list[5]+'_evaluation.csv')



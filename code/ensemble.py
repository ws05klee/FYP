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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
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





# %% [markdown]
# start here when building new ensemble model

# %%
def load_all_models(model_path_name):
    all_models = []
    # model_path_name = []
    model_names = model_path_name
    for model_name in model_names:
        filename = os.path.join(model_name)
        model = tf.keras.models.load_model(filename)
        all_models.append(model)
        print('loaded:', filename)
    return all_models


summary_path = 'model_saved_new1\\ensemble\\'
model_name0 = summary_path+'model-Dense201-32-0.9625.h5'
# model_name1 = summary_path+'model-InceptionV3-40-0.9150.h5'
model_name1 = summary_path+'model-MobileV2-11-0.9500.h5'
model_name4 = summary_path+'model-VGG16-32-0.9375.h5'
model_name5 = summary_path+'model-Xception-17-0.9250.h5'
# model_name6 = summary_path+'model-Res101-21-0.9542.h5'

models = load_all_models([model_name0,model_name5,model_name1,model_name4])


# models = load_all_models([model_name0,model_name1,model_name2,model_name4])
print(models)

# %%
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
def ensemble_model(models):
    for i, layer in enumerate(models[0].layers):
        layer._name = models[0].layers[i]._name + "_dense"
    for i, layer in enumerate(models[1].layers):
        layer._name = models[1].layers[i]._name + "_xcep"
    for i, layer in enumerate(models[2].layers):
        layer._name = models[2].layers[i]._name + "_mobile"
    for i, layer in enumerate(models[3].layers):
        layer._name = models[3].layers[i]._name + "_vgg"
    # for i, layer in enumerate(models[3].layers):
    #     layer._name = models[3].layers[i]._name + "_xcep"

    for i, model in enumerate(models):
        for layer in model.layers:
            layer.trainable = False
            from tensorflow.keras import layers

    ensemble_visible = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]
    merge = tf.keras.layers.concatenate(ensemble_outputs)
    merge = tf.keras.layers.Dense(10, activation='relu')(merge)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(merge)
    model = tf.keras.models.Model(inputs=ensemble_visible, outputs=output)
    # plot_model(model, show_shapes=True, to_file='model_graph.png')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    return model

# %%
# model_used = ['InceptionV3', 'VGG16']
# model_best_acc = ['-02-0.8875','-05-0.9100']

# model_weight_loaded1 =  model_saved_folder_path + model_weight_path + 'model-' + model_used[0] + model_best_acc[0] + '.h5'
# model_weight_loaded2 =  model_saved_folder_path + model_weight_path + 'model-' + model_used[1] + model_best_acc[1] + '.h5'
# model_weight_name =[model_weight_loaded1,model_weight_loaded2]

# print(model_weight_name)

# models = load_all_models(model_weight_name)
# models.summary()
model = ensemble_model(models)
model.summary()
model.save(summary_path + 'model_ensemble4_2.h5')




# %% [markdown]
# start from here before saving np
# 

# %%
import matplotlib.pyplot as plt
from tensorflow.keras import layers


initial_lr = 1e-3 # learning rate
epochs = 40 # no of eopches to train
batch_size = 32 # batch size 
img_size = 224

print("LOADING IMAGES.......")
imagePaths = list(paths.list_images("data1"))
data = []
labels =[]


#loop for the image paths
for imagePath in imagePaths:
	#extract the class label from the file
	label = imagePath.split(os.path.sep)[-2]

	#load the image
	#swap color channels and resize it
	#fixed 224*224 pixels while ingorning aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (img_size, img_size))

	#update the data and labes lists
	data.append(image)
	labels.append(label)

print("original labels: ", labels[0])
print("original labels: ", labels[-1])
# print("original labels to binary: ",labels)

#convert the data and labels to numpay
#while scalling the pixel to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)


# performs one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print("original labels to binary: ",labels[0])
print("original labels to binary: ",labels[-1])
# print("original labels to binary: ",labels)


#partition the data for tranning(80%)
#and testing(20%) using splits
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

data_generator_with_aug = ImageDataGenerator(
#  rotation_range=45,
#  fill_mode="nearest", 
#  horizontal_flip=True, 
 width_shift_range = 0.1, 
 height_shift_range = 0.1, 
#  shear_range=16
)



import pandas as pd
import numpy as np

model_name =  'model-ensemble'
summary_path = 'model_saved_new\\ensemble\\'


# print(testY.argmax(axis=1))
# print(testY)
np.save(summary_path +'testY_' + model_name + '.npy', testY)
np.save(summary_path +'trainY_' + model_name + '.npy', trainY)
np.save(summary_path +'testX_' + model_name + '.npy', testX)
np.save(summary_path +'trainX_' + model_name + '.npy', trainX)

# %% [markdown]
# **start from here after saving np**
# 

# %%
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

summary_path = 'model_saved_new1\\ensemble\\'
model = load_model(summary_path + 'model_ensemble4_2.h5')

import pandas as pd
import numpy as np

# result_path  = ''
# model_name = 'model_ensemble'

initial_lr = 1e-2 # learning rate
epochs = 40 # no of eopches to train
batch_size = 8 # batch size 
img_size = 224

# opt_adam= Adam(learning_rate=initial_lr, decay=initial_lr / epochs)
# opt_sgd = SGD(learning_rate=initial_lr, decay=initial_lr / epochs)

# opt = opt_sgd

testY = np.load('model_saved_new1\\testY_.npy')
trainY = np.load('model_saved_new1\\trainY_.npy')
testX = np.load('model_saved_new1\\testX_.npy')
trainX = np.load('model_saved_new1\\trainX_.npy')

X = [trainX for _ in range(len(model.input))]
X_1 = [testX for _ in range(len(model.input))]

del trainX
del testX

data_generator_with_aug = ImageDataGenerator(
#  rotation_range=45,
#  fill_mode="nearest", 
#  horizontal_flip=True, 
 width_shift_range = 0.1, 
 height_shift_range = 0.1, 
#  shear_range=16
)

# %%
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

batch_size=8

# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


model_filepath = summary_path + 'ensemble4_2\\' + "model-ensemble2-{epoch:02d}-{val_accuracy:.4f}.h5"

checkpoint = ModelCheckpoint(
            filepath=model_filepath,
            monitor = 'val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
            )
history = model.fit(
        data_generator_with_aug(X, trainY,batch_size=batch_size),
        steps_per_epoch=1600 // batch_size,
        epochs=epochs,
        validation_data=(X_1, testY),
        validation_steps=400 // batch_size,
        callbacks=[checkpoint]
        )

# %%
import pandas as pd
import numpy as np

summary_path = 'model_saved_new1\\ensemble\\ensemble4_2\\'

model_name =  'model-ensemble4'

# # # print(testY.argmax(axis=1))
# # # print(testY)
# np.save(summary_path +'testY_' + model_name + '.npy', testY)
# np.save(summary_path +'trainY_' + model_name + '.npy', trainY)
# np.save(summary_path +'testX_' + model_name + '.npy', testX)
# np.save(summary_path +'trainX_' + model_name + '.npy', trainX)

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = summary_path +model_name + '_history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
# or save to csv: 
hist_csv_file = summary_path +model_name + '_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)



# %% [markdown]
# ***Evaulation***

# %%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

testY = np.load('model_saved_new1\\testY_.npy')
trainY = np.load('model_saved_new1\\trainY_.npy')
testX_data = np.load('model_saved_new1\\testX_.npy')
trainX_data = np.load('model_saved_new1\\trainX_.npy')

batch_size = 8

# result_path  = ''

summary_path = 'model_saved_new1\\ensemble\\ensemble4\\'

# result_path use this to store different condition
result_path  = ''
model_name = 'model-ensemble-04-0.9700'
model_name_data = 'model-ensemble'



model = load_model(summary_path +result_path + model_name + '.h5')


X = [trainX_data for _ in range(len(model.input))]
X_1 = [testX_data for _ in range(len(model.input))]

trainX_data = X
testX_data = X_1
# make predications on the testing set
print("evaluating network.....")
predIdxs = model.predict(testX_data, batch_size=batch_size)

threshold = 0.5

# Convert predicted probabilities to binary values using threshold
predIdxs = (predIdxs > threshold).astype(int)


cm = confusion_matrix(testY, predIdxs)

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
precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0])	### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])	### Specificity (true negatives / all actual negatives) =TN / TN + FP
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
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.savefig(summary_path + 'dense_adam0.001.png')
plt.show()


accurancy = "%.4f" % accurancy
precision = "%.4f" % precision
sensitivity = "%.4f" % sensitivity
specificity = "%.4f" % specificity
f1score = "%.4f" % f1score
npv = "%.4f" % npv

evaluation_result = [accurancy,precision,sensitivity,specificity,f1score,npv]

import csv
evaluation_method = ['Method','Accurancy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'NPV']

with open(summary_path + 'evaluation.csv', 'w') as f:    
    csv_writer = csv.writer(f)
    csv_writer.writerow(evaluation_method)
    csv_writer.writerows(evaluation_result)
f.close





# %% [markdown]
# loss and acc graph

# %%
#### for loss and accuracy curve
import matplotlib.pyplot as plt
import json
import pandas as pd

model_history_json_name = summary_path +  result_path + model_name + '_history.json'
history_json = json.load(open(model_history_json_name, 'r'))
history_df=pd.DataFrame(history_json)
history_df=history_df.reset_index()
history_df.columns.values[0]='epochs'


for i in range(len(history_df['epochs'])):
    history_df['epochs'][i]=int(history_df['epochs'][i])+1


plt.style.use("ggplot")
plt.figure()
plt.plot()
plt.plot(history_df['epochs'],history_df['loss'],label='Training Loss')
plt.plot(history_df['epochs'],history_df['val_loss'],label='Validation Loss')
plt.ylim(0,1)
plt.legend(loc="upper right")
plt.title("Tranning and Validation Loss for Integrated Ensemble")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig(summary_path + 'val_train_loss_' +'integrated_ensemble6' + '.png', bbox_inches = 'tight')
# annot_max(history_df['epochs'],history_df['loss'])
plt.show()

plt.figure()
plt.plot()
plt.plot(history_df['epochs'],history_df['accuracy'],label='Training Accuracy')
plt.plot(history_df['epochs'],history_df['val_accuracy'],label='Validation Accuracy')
plt.ylim(0,1)
plt.legend(loc="lower right")
plt.title("Tranning and Validation Accuracy for Integrated Ensemble")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.savefig(summary_path + 'val_train_accuracy_' +'integrated_ensemble6' + '.png', bbox_inches = 'tight')
# annot_max(history_df['epochs'],history_df['accuracy'])
# annot_max(history_df['epochs'],history_df['val_accuracy'])
plt.show()



#fn here: actual is 0(covideff) but identify to 1(normalleff) 0-->1: 7 0-->0:73
# for current confusion matrix become:
# original labels:  covideff
# original labels:  normaleff
# original labels to binary:  [0]
# original labels to binary:  [1]
# binary labels to category:  [1. 0.]
# binary labels to category:  [0. 1.]
# normaleff:	tn fp
# covideff:	fn tp


## confusion matrix (normally online is like below, my diff with online)
## tp fp
## fn tn

### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
### Misclassification (all incorrect / all) = FP + FN / TP + TN + FP + FN
### Precision (true positives / predicted positives) = TP / TP + FP
### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
### Specificity (true negatives / all actual negatives) =TN / TN + FP




# %% [markdown]
# each ensemble seprate plot

# %%
#### for loss and accuracy curve
import matplotlib.pyplot as plt
import json
import pandas as pd

summary_path = 'model_saved_new1\\dense\data_aug\\'
result_path = ['com_1','com_2','com_3','com_4','com_5','com_6','com_7']
model_name = 'model-Dense201-39-0.8950'

for k in range(len(result_path)):
    model_history_json_name = summary_path +  result_path[k] + '\\' + model_name + '_history.json'
    history_json = json.load(open(model_history_json_name, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'


    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1


    plt.style.use("ggplot")
    plt.figure()
    plt.plot()
    plt.plot(history_df['epochs'],history_df['loss'],label='Training Loss')
    plt.plot(history_df['epochs'],history_df['val_loss'],label='Validation Loss')
    plt.ylim(0,1)
    plt.legend(loc="upper right")
    plt.title("Tranning and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.savefig(summary_path + 'val_train_loss_' + result_path[k] + '.png', bbox_inches = 'tight')
    # annot_max(history_df['epochs'],history_df['loss'])
    plt.show()

    plt.figure()
    plt.plot()
    plt.plot(history_df['epochs'],history_df['accuracy'],label='Training Accuracy')
    plt.plot(history_df['epochs'],history_df['val_accuracy'],label='Validation Accuracy')
    plt.ylim(0,1)
    plt.legend(loc="lower right")
    plt.title("Tranning and Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.savefig(summary_path + 'val_train_accuracy_' +result_path[k] + '.png', bbox_inches = 'tight')
    # annot_max(history_df['epochs'],history_df['accuracy'])
    # annot_max(history_df['epochs'],history_df['val_accuracy'])
    plt.show()



    #fn here: actual is 0(covideff) but identify to 1(normalleff) 0-->1: 7 0-->0:73
    # for current confusion matrix become:
    # original labels:  covideff
    # original labels:  normaleff
    # original labels to binary:  [0]
    # original labels to binary:  [1]
    # binary labels to category:  [1. 0.]
    # binary labels to category:  [0. 1.]
    # normaleff:	tn fp
    # covideff:	fn tp


    ## confusion matrix (normally online is like below, my diff with online)
    ## tp fp
    ## fn tn

    ### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    ### Misclassification (all incorrect / all) = FP + FN / TP + TN + FP + FN
    ### Precision (true positives / predicted positives) = TP / TP + FP
    ### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    ### Specificity (true negatives / all actual negatives) =TN / TN + FP




# %%
#### for loss and accuracy curve
import matplotlib.pyplot as plt
import json
import pandas as pd

summary_path = 'model_saved_new1\\ensemble1\\'
result_path = ['ensemble2','ensemble3','ensemble4','ensemble5','ensemble6']
model_name = ['model-ensemble-04-0.9700','model-ensemble3-40-0.9700','model-ensemble4-10-0.9725','model-ensemble5-16-0.9500','model-ensemble6-15-0.9675']




plt.style.use("ggplot")
plt.figure()
for k in range(len(result_path)):
    model_history_json_name = summary_path +  result_path[k] + '\\' + model_name[k] + '_history.json'
    history_json = json.load(open(model_history_json_name, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1

    plt.plot()
    plt.plot(history_df['epochs'],history_df['loss'],label= result_path[k])
    # # plt.plot(history_df['epochs'],history_df['val_loss'],label='Validation Loss')
    plt.ylim(0,1)
    plt.legend(loc="best")
    plt.title("Tranning Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    # plt.show()

plt.savefig(summary_path + 'train_loss_' + 'ensemble' + '.png', bbox_inches = 'tight')



plt.style.use("ggplot")
plt.figure()
for k in range(len(result_path)):
    model_history_json_name = summary_path +  result_path[k] + '\\' + model_name[k] + '_history.json'
    history_json = json.load(open(model_history_json_name, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1

    plt.plot()
    plt.plot(history_df['epochs'],history_df['val_loss'],label= result_path[k])
    # # plt.plot(history_df['epochs'],history_df['val_loss'],label='Validation Loss')
    plt.ylim(0,1)
    plt.legend(loc="best")
    plt.title("Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    # plt.show()

plt.savefig(summary_path + 'val_loss_' + 'ensemble' + '.png', bbox_inches = 'tight')



plt.style.use("ggplot")
plt.figure()
for k in range(len(result_path)):
    model_history_json_name = summary_path +  result_path[k] + '\\' + model_name[k] + '_history.json'
    history_json = json.load(open(model_history_json_name, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1

    plt.plot()
    plt.plot(history_df['epochs'],history_df['val_accuracy'],label= result_path[k])
    # # plt.plot(history_df['epochs'],history_df['val_loss'],label='Validation Loss')
    plt.ylim(0.5,1)
    plt.legend(loc="best")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    # plt.show()

plt.savefig(summary_path + 'val_acc_' + 'ensemble' + '.png', bbox_inches = 'tight')


plt.style.use("ggplot")
plt.figure()
for k in range(len(result_path)):
    model_history_json_name = summary_path +  result_path[k] + '\\' + model_name[k] + '_history.json'
    history_json = json.load(open(model_history_json_name, 'r'))
    history_df=pd.DataFrame(history_json)
    history_df=history_df.reset_index()
    history_df.columns.values[0]='epochs'

    for i in range(len(history_df['epochs'])):
        history_df['epochs'][i]=int(history_df['epochs'][i])+1

    plt.plot()
    plt.plot(history_df['epochs'],history_df['accuracy'],label= result_path[k])
    # # plt.plot(history_df['epochs'],history_df['val_loss'],label='Validation Loss')
    plt.ylim(0.5,1)
    plt.legend(loc="best")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    # plt.show()

plt.savefig(summary_path + 'acc_' + 'ensemble' + '.png', bbox_inches = 'tight')



    # # annot_max(history_df['epochs'],history_df['loss'])
    # plt.show()

    # plt.figure()
    # plt.plot()
    # plt.plot(history_df['epochs'],history_df['accuracy'],label='Training Accuracy')
    # plt.plot(history_df['epochs'],history_df['val_accuracy'],label='Validation Accuracy')
    # plt.ylim(0,1)
    # plt.legend(loc="lower right")
    # plt.title("Tranning and Validation Accuracy")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Accuracy")
    # plt.savefig(summary_path + 'val_train_accuracy_' +result_path[k] + '.png', bbox_inches = 'tight')
    # # annot_max(history_df['epochs'],history_df['accuracy'])
    # # annot_max(history_df['epochs'],history_df['val_accuracy'])
    # plt.show()



    #fn here: actual is 0(covideff) but identify to 1(normalleff) 0-->1: 7 0-->0:73
    # for current confusion matrix become:
    # original labels:  covideff
    # original labels:  normaleff
    # original labels to binary:  [0]
    # original labels to binary:  [1]
    # binary labels to category:  [1. 0.]
    # binary labels to category:  [0. 1.]
    # normaleff:	tn fp
    # covideff:	fn tp


    ## confusion matrix (normally online is like below, my diff with online)
    ## tp fp
    ## fn tn

    ### Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    ### Misclassification (all incorrect / all) = FP + FN / TP + TN + FP + FN
    ### Precision (true positives / predicted positives) = TP / TP + FP
    ### Sensitivity aka Recall (true positives / all actual positives) = TP / TP + FN
    ### Specificity (true negatives / all actual negatives) =TN / TN + FP




# %%
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
# from keras.utils import vis_utils
import os

# model1 = models[0]
# model2 = models[1]

summary_path = 'model_saved_new1\\ensemble\\ensemble4\\'

# result_path use this to store different condition
result_path  = ''
model_name = 'model-ensemble4-10-0.9725'
model_name_data = 'model-ensemble'
summary_save_path = 'model_summary/'



model = load_model(summary_path +result_path + model_name + '.h5')


plot_model(model,to_file= summary_save_path + 'model_ensemble4_plot.png' , show_shapes=True)
# plot_model(model1, to_file= summary_save_path + 'model_1_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(model2, to_file= summary_save_path + 'model_2_plot.png', show_shapes=True, show_layer_names=True)

# model1.summary()
# model2.summary()
# model.summary()

# %%
from tensorflow.keras.utils import plot_model
# from keras.utils import vis_utils
import os

model1 = models[0]
model2 = models[1]

summary_save_path = 'model_summary/'
### create file here need to manual create folder at window explorer

plot_model(model,to_file= summary_save_path + 'model_ensemble_plot.png' , show_shapes=True, show_layer_names=True)
plot_model(model1, to_file= summary_save_path + 'model_1_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model2, to_file= summary_save_path + 'model_2_plot.png', show_shapes=True, show_layer_names=True)

# model1.summary()
# model2.summary()
# model.summary()



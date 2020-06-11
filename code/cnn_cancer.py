#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:10:55 2019

@author: smartlab
"""

from keras import layers
from keras import models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers, regularizers
from keras.layers.core import Activation

######################################
RESOLUTION = 250
BATCH_SIZE = 3

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory('/home/smartlab/Desktop/TRAIN', target_size=(RESOLUTION, RESOLUTION),
batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

val_generator = val_datagen.flow_from_directory(
        '/home/smartlab/Desktop/TRAIN', target_size=(RESOLUTION, RESOLUTION),
batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

#####################################

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory('/home/smartlab/Desktop/TRAIN', target_size=(RESOLUTION, RESOLUTION), batch_size=BATCH_SIZE, class_mode='categorical')

##################################################################################

###############################################################

test_generator.filenames

#####################################################

model = Sequential()
model.add(Conv2D(96, kernel_size = (11, 11), input_shape=(RESOLUTION, RESOLUTION, 3)))
convout1 = Activation('relu')
model.add(convout1)
convout2 = MaxPooling2D(pool_size=(3,3))
model.add(convout2)
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(5,5)))
convout3 = Activation('relu')
model.add(convout3)
convout4 = MaxPooling2D(pool_size=(3,3))
model.add(convout4)
model.add(BatchNormalization())

model.add(Conv2D(384, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(BatchNormalization())

model.add(Conv2D(384, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(BatchNormalization())

model.add(Conv2D(384, kernel_size=(2,2)))
convout5 = Activation('relu')
model.add(convout5)
convout6 = MaxPooling2D(pool_size=(1,1))
model.add(convout6)
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

##########################################################

model.summary()

##################################################

N_TRAIN = 62
N_VAL = 6
history = model.fit_generator(
        train_generator,
        steps_per_epoch=(N_TRAIN // BATCH_SIZE),
        epochs=100,
        validation_data=val_generator,
        validation_steps=(N_VAL // BATCH_SIZE)
        )

##########################################################

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='best')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='best')

########################################################################

model.save('radio_galaxies_CNN_minimum.h5')
model.save_weights('radio_galaxies_CNN_Weights_minimum.h5')

######################################################################

model.pop()
model.pop()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])
model.summary()

#######################################################################

Y_pred = model.predict_generator(test_generator, steps = 5)

#############################################################################

np.save('features_minimum.npy', Y_pred)

############################################################################

output_file = []
for name, arr_ in zip(test_generator.filenames, Y_pred):
    output_file.append((name, arr_))

###########################################################################

with open('output_features.txt', 'w') as f:
    for i in output_file:
        f.write('{}, {}\n'.format(i[0], i[1]))
        
##########################################################################


# In[ ]:





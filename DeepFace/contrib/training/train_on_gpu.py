#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
tf.compat.v1.disable_eager_execution()


initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
# opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
opt = tf.keras.optimizers.legacy.SGD(learning_rate=initial_learning_rate, momentum=0.9, nesterov=True)

IMAGE_SIZE = (152, 152)
CHANNELS = 3
NUM_CLASSES = 8631

BATCH_SIZE = 8
LEARN_RATE = 0.01 * (BATCH_SIZE / 128)
MOMENTUM = 0.9
EPOCHS = 15


CL_PATH = 'D:/VGG-Face2/data/vggface2_train/train/VGGFace2-class_labels_train.txt'
# Note that the folder 'tfrecords' contains two sub folders :train, test
# train (contains training tfrecords), test (contains testing tfrecords)
DATASET_PATH = 'D:/code test/DeepFace2/TFRecord'  
TB_PATH = 'D:/code test/DeepFace2/LOG'

keras.backend.clear_session()

# Change the import statement for deepface
from deepface.dataset import get_train_test_dataset, SHUFFLE_BUFFER

train, val = get_train_test_dataset(CL_PATH, DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)
# these are essential values that have to be set
# in order to determine the right number of steps per epoch
train_samples, val_samples = 2307424, 25893
# this value is set so as to ensure
#  proper shuffling of dataset
SHUFFLE_BUFFER = train_samples
print('train.num_classes == ', train.num_classes)
print('validate.num_classes == ', val.num_classes)
#assert train.num_classes == val.num_classes == NUM_CLASSES

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    patience=1, min_lr=0.0001, verbose=1) # mandatory step in training, as specified in paper
tensorboard = keras.callbacks.TensorBoard(TB_PATH)
# checkpoints = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}_{val_acc:.4f}.hdf5',
#     monitor='val_acc', save_weights_only=True)
checkpoints = keras.callbacks.ModelCheckpoint(
    'weights.{epoch:02d}_{val_accuracy:.4f}.hdf5',
    monitor='val_accuracy',  # Change 'val_acc' to 'val_accuracy'
    save_weights_only=True
)


cbs = [reduce_lr, checkpoints, tensorboard]

# Change the import statement for deepface
from deepface.deepface import create_deepface



model=create_deepface(IMAGE_SIZE, CHANNELS, NUM_CLASSES, LEARN_RATE, MOMENTUM) 

# model.compile(loss='binary_crossentropy', metrics=['accuracy'])
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=initial_learning_rate, momentum=0.9, nesterov=True)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# model.fit(train.data, steps_per_epoch=train_samples // BATCH_SIZE + 1,
#     validation_data=val.data, validation_steps=val_samples // BATCH_SIZE + 1,
#     callbacks=cbs, epochs=EPOCHS)
# model.fit_generator(
#     train.data,
#     steps_per_epoch=min(train_samples // BATCH_SIZE + 1, 1500),  # Adjust the number of steps as needed
#     validation_data=val.data,
#     validation_steps=val_samples // BATCH_SIZE + 1,
#     callbacks=cbs,
#     epochs=EPOCHS
# )

model.fit(train.data, steps_per_epoch= min (train_samples // BATCH_SIZE + 1, 2000),
    validation_data=val.data, validation_steps=val_samples // BATCH_SIZE + 1,
    callbacks=cbs, epochs=EPOCHS)

model.save('model.h5')
# -*- coding: utf-8 -*-
import random
import string
import math
import itertools
import os
import numpy as np
import imgaug
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection
import keras_ocr
import glob
import cv2
import keras
print('GPU available :',tf.test.is_gpu_available())

recognizer_alphabet = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ '
recognizer = keras_ocr.recognition.Recognizer(
    alphabet=recognizer_alphabet,weights=None)
recognizer.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
#Read dataset
paths = glob.glob("out_train/*")
labels = [str(os.path.basename(i).split("_")[0]).strip() for i in paths]
print(max([len(i) for i in labels]))
train_labels = [(filepath, word) for filepath, word in zip(paths,labels)  if len(word)<221]
print("number trains",len(train_labels))
# Create dataloader
batch_size = 16
augmenter = imgaug.augmenters.Sequential([
   
])
#  imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0)),
train_labels, validation_labels = sklearn.model_selection.train_test_split(train_labels, test_size=0.2, random_state=42)
(training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
    (
        keras_ocr.datasets.get_recognizer_image_generator_new(
            labels=labels,
            height=recognizer.model.input_shape[1],
            width=recognizer.model.input_shape[2],
            alphabet=recognizer.alphabet,
            augmenter=augmenter
        ),
        len(labels) // batch_size
    ) for labels, augmenter in [(train_labels, augmenter), (validation_labels, None)]
]
training_gen, validation_gen = [
    recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=batch_size
    )
    for image_generator in [training_image_gen, validation_image_gen]
]

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('recognizer_borndigital.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('recognizer_borndigital.csv')
]
recognizer.training_model.fit_generator(
    generator=training_gen,
    steps_per_epoch=training_steps,
    validation_steps=validation_steps,
    validation_data=validation_gen,
    callbacks=callbacks,
    epochs=100,
)

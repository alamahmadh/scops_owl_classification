import os
import ast
import pickle
from time import localtime, strftime
import pandas as pd

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import config
#from models.resnet import ResNetBuilder
from models.cnn import CNNBaseline, CNNDual
from generator import *

def generate_generator_multiple(generator, sound_path, batch_size, target_size1, target_size2):
    gen_x1 = generator.flow_from_directory(sound_path,
                                           target_size=target_size1,
                                           batch_size=batch_size,
                                           class_mode='categorical',
                                           audio_mode=audio_mode1
                                           #save_to_dir='./visuals/augmented_samples'
                                          )
    
    gen_x2 = generator.flow_from_directory(sound_path,
                                           target_size=target_size2,
                                           batch_size=batch_size,
                                           class_mode='categorical',
                                           audio_mode=audio_mode2
                                          )
    while True:
            X1i = gen_x1.next()
            X2i = gen_x2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

# Setup Callbacks for History
class HistoryCollector(Callback):
    def __init__(self, name):
        self.name = name
        self.data = []

    def on_epoch_end(self, batch, logs={}):
        self.data.append(logs.get(self.name))

if __name__ == "__main__":
    model_name = config.model_name
    nb_classes = config.nb_classes
    inputmode = config.inputmode
    input_shape = config.input_shape
    input_shape1 = config.input_shape1
    input_shape2 = config.input_shape2
    optimizer = config.optimizer
    first_epoch = config.first_epoch

    model = None
    if model_name == 'cnn_baseline':
        model = CNNBaseline(nb_classes, input_shape)
    elif model_name == 'cnn_dual':
        model = CNNDual(nb_classes, input_shape1, input_shape2)
    elif model_name == 'resnet_18':
        model = ResNetBuilder.build_resnet_18(input_shape, nb_classes)
    elif model_name == 'resnet_34':
        model = ResNetBuilder.build_resnet_34(input_shape, nb_classes)
    elif model_name == 'resnet_50':
        model = ResNetBuilder.build_resnet_50(input_shape, nb_classes)
    elif model_name == 'resnet_101':
        model = ResNetBuilder.build_resnet_101(input_shape, nb_classes)
    elif model_name == 'resnet_152':
        model = ResNetBuilder.build_resnet_152(input_shape, nb_classes)
    else:
        raise ValueError("Can not find model ", model_name, ".")

    if optimizer == 'sgd':
        sgd = SGD(lr=config.learning_rate, 
                  #decay=config.decay, 
                  momentum=config.momentum,
                  nesterov=config.nesterov)
        model.compile(loss=config.loss_function,
                      optimizer=sgd,
                      metrics=['accuracy'])
    elif optimizer == 'adam'
        adam = Adam(learning_rate=config.learning_rate)
        model.compile(loss=config.loss_function,
                      optimizer=adam,
                      metrics=['accuracy'])
    
    if first_epoch=='False':
        # load weights
        model.load_weights(config.weight_file_path)
        print("loading weigths from: " + config.weight_file_path)
    else:
        print("using initial weights")

    # Callback history collectors
    trainLossHistory = HistoryCollector('loss')
    validLossHistory = HistoryCollector('val_loss')
    trainAccHistory = HistoryCollector('accuracy')
    validAccHistory = HistoryCollector('val_accuracy')
    best_weight_file_path = "best_weights_dual.h5"
    checkpoint = ModelCheckpoint(best_weight_file_path, 
                                 monitor='val_accuracy', verbose=0,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto')

    #Create data generator
    train_datagen = SoundDataGenerator(rescale=1./255,
                                       time_shift=TimeShiftAugmentation,
                                       pitch_shift=PitchShiftAugmentation)

    valid_datagen = SoundDataGenerator(rescale=1./255)

    if inputmode=="dual":
        train_dual_generator = generate_generator_multiple(train_datagen, 
                                                           config.train_path, 
                                                           config.batch_size, 
                                                           config.target_size1,
                                                           config.target_size2)
        valid_dual_generator = generate_generator_multiple(valid_datagen, 
                                                           config.valid_path,
                                                           config.batch_size,
                                                           config.target_size1,
                                                           config.target_size2)
    elif inputmode=="single":
        train_generator = train_datagen.flow_from_directory(config.train_path,
                                                            target_size=config.target_size,
                                                            batch_size=config.batch_size,
                                                            class_mode='categorical',
                                                            audio_mode=config.audio_mode)
        valid_generator = valid_datagen.flow_from_directory(config.valid_path,
                                                            target_size=config.target_size,
                                                            batch_size=config.batch_size,
                                                            class_mode='categorical',
                                                            audio_mode=config.audio_mode)
    else:
        raise ValueError("Wrong input mode ",inputmode, ".")
    
    # Fit the model
    dst_dir = 'data'
    history = model.fit(
        train_dual_generator, 
        steps_per_epoch= sum([len(files) for root, directories, files in os.walk(dst_dir)])// config.batch_size,
        epochs=config.nb_epoch,
        validation_data=valid_dual_generator,
        validation_steps= sum([len(files) for root, directories, files in os.walk(config.valid_path)])// config.batch_size,
        callbacks=[trainLossHistory, validLossHistory, trainAccHistory, validAccHistory, checkpoint])
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    rows = zip(acc, val_acc, loss, val_loss)

    df = pd.DataFrame(list(rows), 
               columns =['acc', 'val_acc', 'loss', 'val_loss']) 

    df.to_csv(config.training_csv)
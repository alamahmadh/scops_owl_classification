from tensorflow.keras.layers import (Input, Dense, Activation, Dropout, Flatten, Concatenate, concatenate, 
                                     Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras import backend as K

def CNNBaseline(nb_classes, input_shape):
    """ Instantiate a CNN Baseline architecture
    # Arguments
        nb_classes  : the number of classification classes
        input_shape : the shape of the input layer (rows, columns)
    # Returns
        A Keras model instance
    """

    img_input = Input(shape=input_shape)

    bn_axis = 3

    #============================================================================
    # x = Dropout(0.2)(img_input)
    x1 = BatchNormalization(axis=bn_axis, name='bn_conv1_1')(img_input)
    x1 = Conv2D(32, 5, strides=(1, 2), activation='relu',
                kernel_initializer="he_normal", padding='same', name='conv_1_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_conv2_1')(x1)
    x1 = Conv2D(32, 5, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding='same', name='conv2_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_conv3_1')(x1)
    x1 = Conv2D(64, 5, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding='same', name='conv3_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_conv4_1')(x1)
    x1 = Conv2D(128, 3, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding='same', name='conv4_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_dense_1')(x1)
    # flatten 3D feature maps to 1D feature vectors
    x1 = Flatten(name='flatten_1')(x1)
    x1 = Dropout(0.5)(x1)
    # dense layer
    x1 = Dense(1024, activation='relu', kernel_regularizer=L2(0.0001), name='dense')(x1)

    # soft max layer
    x1 = Dropout(0.5)(x1)
    x1 = Dense(nb_classes, activation='softmax', name='softmax')(x1)

    model = Model(img_input, x1)
    
    return model

def CNNDual(nb_classes, input_shape1, input_shape2):
    """ Instantiate a CNN Dual architecture
    # Arguments
        nb_classes  : the number of classification classes
        input_shape : the shape of the input layer (rows, columns)
    # Returns
        A Keras model instance
    """

    img_input1 = Input(shape=input_shape1)
    img_input2 = Input(shape=input_shape2)

    bn_axis = 3

    #============================================================================
    # x = Dropout(0.2)(img_input)
    x1 = BatchNormalization(axis=bn_axis, name='bn_conv1_1')(img_input1)
    x1 = Conv2D(32, 5, strides=(1, 2), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv_1_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_conv2_1')(x1)
    x1 = Conv2D(32, 5, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv2_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_conv3_1')(x1)
    x1 = Conv2D(64, 5, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv3_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_conv4_1')(x1)
    x1 = Conv2D(128, 3, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv4_1')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    #x1 = BatchNormalization(axis=bn_axis, name='bn_conv5_1')(x1)
    #x1 = Convolution2D(128, 3, 3, subsample=(1, 1), activation='relu',
    #                  init="he_normal", border_mode="same", name='conv5_1')(x1)
    #x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    #x1 = BatchNormalization(axis=bn_axis, name='bn_dense_1')(x1)
    # flatten 3D feature maps to 1D feature vectors
    #x1 = Flatten(name='flatten_1')(x1)
    #============================================================================

    x2 = BatchNormalization(axis=bn_axis, name='bn_conv1_2')(img_input2)
    x2 = Conv2D(32, 5, strides=(1, 2), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv_1_2')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)

    x2 = BatchNormalization(axis=bn_axis, name='bn_conv2_2')(x2)
    x2 = Conv2D(32, 5, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv2_2')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)

    x2 = BatchNormalization(axis=bn_axis, name='bn_conv3_2')(x2)
    x2 = Conv2D(64, 5, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv3_2')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)

    x2 = BatchNormalization(axis=bn_axis, name='bn_conv4_2')(x2)
    x2 = Conv2D(128, 3, strides=(1, 1), activation='relu',
                kernel_initializer="he_normal", padding="same", name='conv4_2')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)

    #x2 = BatchNormalization(axis=bn_axis, name='bn_conv5_2')(x2)
    #x2 = Convolution2D(128, 3, 3, subsample=(1, 1), activation='relu',
    #                  init="he_normal", border_mode="same", name='conv5_2')(x2)
    #x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)

    #x2 = BatchNormalization(axis=bn_axis, name='bn_dense_2')(x2)
    # flatten 3D feature maps to 1D feature vectors
    #x2 = Flatten(name='flatten_2')(x2)
    #=============================================================================
    
    xtot = concatenate([x1, x2])
    xtot = BatchNormalization(axis=bn_axis, name='bn_dense')(xtot)
    xtot = Flatten(name='flatten')(xtot)
    xtot = Dropout(0.5)(xtot)
    xtot = Dense(1024, activation='relu', kernel_regularizer=L2(0.0001), name='dense')(xtot)
    xtot = Dropout(0.5)(xtot)

    # soft max layer
    xtot = Dense(nb_classes, activation='softmax', name='softmax')(xtot)

    model = Model(inputs=[img_input1, img_input2], outputs=xtot)

    return model

train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test"
weight_file_path = "weights_dual.h5"
best_weight_file_path = "best_weights_dual.h5"
training_csv = "training_dual.csv"

nb_classes = 7

optimizer = "sgd"
learning_rate = 0.0001
decay = 1e-6
momentum = 0.0
nesterov = False
loss_function = "categorical_crossentropy"

batch_size = 16
nb_epoch = 80

TimeShiftAugmentation = True
PitchShiftAugmentation = True
SameClassAugmentation = False

input_shape = (256, 512, 1)
input_shape1 = (256, 512, 1)
input_shape2 = (256, 512, 1)
img_rows, img_cols, nb_channels =  input_shape
img_rows1, img_cols1, nb_channels1 =  input_shape1
img_rows2, img_cols2, nb_channels2 =  input_shape2
target_size = (img_rows, img_cols)
target_size1 = (img_rows1, img_cols1)
target_size2 = (img_rows2, img_cols2)

#dual mode input
audio_mode1 = "melspectrogram"
audio_mode2 = "mfcc"
audio_mode = "mfcc" #this is for single mode input

inputmode = 'dual'
first_epoch = "True"

if inputmode == 'dual':
    model_name = "cnn_dual"
elif inputmode == 'single':
    model_name = "cnn_baseline"
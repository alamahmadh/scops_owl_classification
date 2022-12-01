#split the dataset into a training set, validation, and testing sets

import numpy as np
np.random.seed(1337)

import argparse
import os
import glob
import shutil

#create the parser
parser = argparse.ArgumentParser(
    description='Split the dataset into train, val, and test sets from a source data'
)
parser.add_argument('--indir', '-i', type=str, default='xeno/mono', help='The directory of the source data')
parser.add_argument('--outdir', '-o', type=str, default='data', help='The directory of the source data')
parser.add_argument('--train', '-t', type=float, default=0.7, help='the desired percentage of the train set. Default 0.7')
parser.add_argument('--test', '-v', type=float, default=0.1, help='the desired percentage of the valid set. Default 0.1')

#Parse the argument
args = parser.parse_args()

train_percentage = float(args.train)
test_percentage = float(args.test)
valid_percentage = 1.0 - train_percentage - test_percentage 

src_dir = args.indir
sound_classes = [x for x in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, x))]

train_dir = "{}/train".format(args.outdir)
valid_dir = "{}/valid".format(args.outdir)
test_dir = "{}/test".format(args.outdir)

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

#copy all data into the train folder
for sound_class in sound_classes:
    sound_class_src_dir = os.path.join(src_dir, sound_class)
    sound_class_dst_dir = os.path.join(train_dir, sound_class)
    shutil.copytree(sound_class_src_dir, sound_class_dst_dir)
    
def split_data(in_dir, out_dir, out_percentage=0.1):
    '''
    This function is to split the train data (aka total data we just copied from xeno directory) 
    into val/test set

    '''
    for sound_class in sound_classes:
        sound_class_in_dir = os.path.join(in_dir, sound_class)
        sound_class_out_dir = os.path.join(out_dir, sound_class)
    
        if not os.path.exists(sound_class_out_dir):
            os.makedirs(sound_class_out_dir)
        
        sound_class_segments = glob.glob(os.path.join(sound_class_in_dir, "*.wav"))
        n_samples = len(sound_class_segments)
        n_out_samples = int(np.ceil(out_percentage * n_samples))
        n_in_samples = n_samples - n_out_samples
        tmp_segments = np.asarray(sound_class_segments)
        
        out_samples = np.random.choice(sound_class_segments, 
                                       n_out_samples, 
                                       replace=False)
            
        for out_sample in out_samples:
            out_dst_sample = os.path.join(sound_class_out_dir, os.path.basename(out_sample))
            shutil.move(out_sample, out_dst_sample)

if __name__ == "__main__":
    #First generate the test set from the train set (i.e., the full data set) 
    split_data(train_dir, test_dir, out_percentage=test_percentage)
    
    #The validation set is generated from the remaining train set
    split_data(train_dir, valid_dir, out_percentage=valid_percentage)
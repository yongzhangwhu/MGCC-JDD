import os
import numpy as np
import shutil
import random

def generate_image_list(dataset_path, img_list_path):
    
    file = open(img_list_path, 'w')
    img_path = os.listdir(dataset_path)
    img_path.sort()

    for i in range(len(img_path)):
        img = os.path.join(dataset_path, img_path[i])
        file.write(img + '\n')


# generate the image list of MSR datasets
def generate_image_list_from_MSR():
    test_path =  '../dataset/Dataset_LINEAR_with_noise/bayer_panasonic/test.txt'
    train_path = '../dataset/Dataset_LINEAR_with_noise/bayer_panasonic/train.txt'
    valid_path = '../dataset/Dataset_LINEAR_with_noise/bayer_panasonic/validation.txt'

    test_path_list = './datasets/MSR_test.txt'
    train_path_list = './datasets/MSR_train.txt'
    valid_path_list = './datasets/MSR_valid.txt'

    input_path = '../dataset/Dataset_LINEAR_with_noise/bayer_panasonic/input'
    groundtruth_path = '../dataset/Dataset_LINEAR_with_noise/bayer_panasonic/groundtruth'

    with open (train_path, 'r') as f:
        f_input = open(train_path_list, 'w')
        for line in f.readlines():
            line = line.strip()
            img_name = line + '.png'
            img_input_path = os.path.join(input_path, img_name)
            img_ground_path = os.path.join(groundtruth_path, img_name)
            f_input.write(img_input_path + ' ' + img_ground_path +'\n')      
        f_input.close()
    
    with open (valid_path, 'r') as f:
        f_input = open(valid_path_list, 'w')
        for line in f.readlines():
            line = line.strip()
            img_name = line + '.png'
            img_input_path = os.path.join(input_path, img_name)
            img_ground_path = os.path.join(groundtruth_path, img_name)
            f_input.write(img_input_path + ' ' + img_ground_path +'\n')      
        f_input.close()

    with open (test_path, 'r') as f:
        f_input = open(test_path_list, 'w')
        for line in f.readlines():
            line = line.strip()
            img_name = line + '.png'
            img_input_path = os.path.join(input_path, img_name)
            img_ground_path = os.path.join(groundtruth_path, img_name)
            f_input.write(img_input_path + ' ' + img_ground_path +'\n')      
        f_input.close()

if __name__ == '__main__':
   img_list_path = './datasets/valid_urban.txt'
   dataset_path = '../dataset/urban100/'
   generate_image_list(dataset_path, img_list_path)


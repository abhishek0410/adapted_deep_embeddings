import os
import sys
import struct
from array import array
import random
import numpy as np
import pdb

class MNIST():
    def __init__(self, path):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

        self.num_classes = 10

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = self.process_images(ims)
        self.test_labels = self.process_labels(labels)

        return self.test_images, self.test_labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = self.process_images(ims)
        self.train_labels = self.process_labels(labels)

        return self.train_images, self.train_labels

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        return np.array(labels)

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    def kntl_data_form(self, t1_train, t1_valid, k, n, t2_test):
        
        #self = <data.mnist.read_mnist.MNIST object at 0x7f3bf9e7bba8>
        #t1_train = 8000; t1_valid = 3000 ;k =10 ;n=5 ;t2_test =10000
        ##t1_train, t1_valid , t2_test is just a number.

       
        ##self.train_images = 60000 
        ##self.test_images =10000


        ##1.)self.train image is then partitioned into less than 5 or greater then 5 :
        ##self.x_train_lt5_full = 30596 (Subset of self.train_images)
        ## self.x_train_gte5_full = 29404 (Subset of self.train_images)

        ##2.)self.test image is partitioned into less than 5 or greater then 5:
        ##self.x_test_lt5_full = x_test_lt5 =5139 (Subset of self.test_images))


        ##3.)Total number of self.x_lessThan5 = x_lessThan5Intraining + x_lessThan5InTesting    
        ## self.x_lt5 =  self.x_train_lt5_full+ self.x_test_lt5_full = 35735
        
        ##self.x_gte5 =   534215

        ##self.x_valid_lt5 = 3000 (subset of self.x_lt5) 3000 is pre decided
        ##self.x_train_lt5 =8000 (subset of self.x_lt5) 8000 is pre-decided

        
        self.load_testing() #Returns test images and test labels
        self.load_training() #Returns train images and train labels

        self.x_train_lt5_full = self.train_images[self.train_labels < 5] ##Size = 30596 , This is the total  number of samples availabel
        self.y_train_lt5_full = self.train_labels[self.train_labels < 5] ##Size = 30596
        shuffle = np.random.permutation(len(self.y_train_lt5_full))
        self.x_train_lt5_full, self.y_train_lt5_full = self.x_train_lt5_full[shuffle], self.y_train_lt5_full[shuffle]

        x_test_lt5 = self.test_images[self.test_labels < 5] ##5139 ##Total number of test samples
        y_test_lt5 = self.test_labels[self.test_labels < 5] ##5139
        shuffle = np.random.permutation(len(y_test_lt5))
        self.x_test_lt5_full, self.y_test_lt5_full = x_test_lt5[shuffle], y_test_lt5[shuffle]

        self.x_lt5 = np.concatenate((self.x_train_lt5_full, self.x_test_lt5_full), axis=0) ##35735
        self.y_lt5 = np.concatenate((self.y_train_lt5_full, self.y_test_lt5_full), axis=0) ##35735

        print('Task 1 full: {0}'.format(len(self.x_lt5))) 

        shuffle = np.random.permutation(len(self.y_lt5))
        self.x_lt5, self.y_lt5 = self.x_lt5[shuffle], self.y_lt5[shuffle] 
        self.x_valid_lt5, self.y_valid_lt5 = self.x_lt5[:t1_valid], self.y_lt5[:t1_valid] ## len(self.x_valid_lt5) = 3000
        self.x_train_lt5, self.y_train_lt5 = self.x_lt5[t1_valid:t1_valid + t1_train], self.y_lt5[t1_valid:t1_valid + t1_train] ##len(self.x_train_lt5) =8000


        print('Task 1 training: {0}'.format(len(self.x_train_lt5)))
        print('Task 1 validation: {0}'.format(len(self.x_valid_lt5)))

        self.x_train_gte5_full = self.train_images[self.train_labels >= 5] ##29404
        self.y_train_gte5_full = self.train_labels[self.train_labels >= 5] - 5
        self.x_test_gte5_full = self.test_images[self.test_labels >= 5] #4861
        self.y_test_gte5_full = self.test_labels[self.test_labels >= 5] - 5

        self.x_gte5 = np.concatenate((self.x_train_gte5_full, self.x_test_gte5_full), axis=0)
        self.y_gte5 = np.concatenate((self.y_train_gte5_full, self.y_test_gte5_full), axis=0)

        print('Task 2 full: {0}'.format(len(self.x_gte5)))

        self.x_train_gte5 = []
        self.y_train_gte5 = []
        classes = np.unique(self.y_gte5)
        chosen_classes = np.random.choice(classes, n, replace=False)
        for c in chosen_classes:
            idx = np.random.choice(np.where(self.y_gte5 == c)[0], k, replace=False)
            self.x_train_gte5.extend(self.x_gte5[idx])
            self.y_train_gte5.extend(self.y_gte5[idx])
            self.x_gte5 = np.delete(self.x_gte5, idx, axis=0)
            self.y_gte5 = np.delete(self.y_gte5, idx, axis=0)

        self.x_train_gte5 = np.array(self.x_train_gte5)
        self.y_train_gte5 = np.array(self.y_train_gte5)

        assert t2_test <= len(self.y_gte5)

        shuffle = np.random.permutation(len(self.y_gte5))
        self.x_gte5, self.y_gte5 = self.x_gte5[shuffle], self.y_gte5[shuffle]
        self.x_test_gte5, self.y_test_gte5 = self.x_gte5[:t2_test], self.y_gte5[:t2_test]

        print('k = {0}, n = {1}'.format(k, n))
        print('Task 2 training: {0}'.format(len(self.x_train_gte5)))
        print('Task 2 test: {0}\n'.format(len(self.x_test_gte5)))

        # pdb.set_trace()

        return (self.x_train_lt5, self.y_train_lt5), (self.x_valid_lt5, self.y_valid_lt5), (self.x_train_gte5, self.y_train_gte5), (self.x_test_gte5, self.y_test_gte5)

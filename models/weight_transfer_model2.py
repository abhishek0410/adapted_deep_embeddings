'''
Adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import os
import sys
import tensorflow as tf
import pdb



model = Sequential()


from .model import Model
from .utils import assign_to_device, _conv, define_scope, _fully_connected, get_available_gpus, _max_pooling, _relu, _softmax

class WeightTransferModel(Model):

    def __init__(self, config):
        self.config = self.get_config(config)
        self.saver = None
        self.learning_rate = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)

    def create_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def save_model(self, sess, step):
        # self.saver.save(sess, os.path.join(self.config.save_dir_by_rep, 'model.ckpt'), global_step=step)
        self.saver.save(sess, os.path.join(self.config.save_dir_by_rep, 'model.h5'), global_step=step)


    def restore_model(self, sess):
        checkpoint = tf.train.latest_checkpoint(self.config.save_dir_by_rep)
        if checkpoint is None:
            sys.exit('Cannot restore model that does not exist')
        self.saver.restore(sess, checkpoint)

        
    ##The below function saves the last checkpoint for the SOURCE - TRAINING :
    def get_last_sourceTraining_checkpoint(self):
        self.last_sourceTraining_checkpoint =  tf.train.latest_checkpoint(self.config.save_dir_by_rep)

    ##This function  restores the model to the last source training Checkpoint
    def restore_model_SOURCE_Trained(self, sess):
        checkpoint = self.last_sourceTraining_checkpoint
        if checkpoint is None:
            sys.exit('Cannot restore model that does not exist')
        self.saver.restore(sess, checkpoint)

    def get_single_device(self):
        devices = get_available_gpus()
        d = self.config.controller
        if devices:
            d = devices[0]
        return d

    @define_scope
    def optimize(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
           
            pred = self.prediction
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, 
                labels=tf.one_hot(self.target, self.config.n)))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(cost)
            #pdb.set_trace()
            return train_op, cost

    @define_scope(scope='stream_metrics')
    def metrics(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            pred = self.prediction
            acc, update_acc = tf.metrics.accuracy(self.target, tf.argmax(_softmax(pred), axis=1))
            return update_acc


# class MNISTWeightTransferModel(WeightTransferModel):
#     def __init__(self, config):
#         model = Sequential()
#         #add model layers
#         model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(x.get_shape()[-1], 32, 1)))
#         model.add(Conv2D(32, kernel_size=3, activation=’relu’))
#         model.add(Flatten())
#         model.add(Dense(10, activation=’softmax’))

#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         super().__init__(config)
#         self.input = tf.placeholder(tf.float32, [None, 784])
#         self.target = tf.placeholder(tf.int32, [None])
#         self.is_task1 = tf.placeholder(tf.bool)
#         self.prediction
#         self.optimize
#         self.metrics
    


class MNISTWeightTransferModel(WeightTransferModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.target = tf.placeholder(tf.int32, [None])
        self.is_task1 = tf.placeholder(tf.bool)
        self.prediction
        self.optimize
        self.metrics

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = tf.reshape(x, [-1, 28, 28, 1])
            x = _relu(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1))
            x = _max_pooling('pool2', _relu(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            x = tf.contrib.layers.flatten(x)
            x = _relu(_fully_connected('fc1', x, 128))
            x1 = lambda: _fully_connected('fc3', x, self.config.n)
            x2 = lambda: _fully_connected('fc4', x, self.config.n)
            x = tf.cond(tf.equal(self.is_task1, tf.constant(True)), x1, x2)
            return x


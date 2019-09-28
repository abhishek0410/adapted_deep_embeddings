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
        self.saver.save(sess, os.path.join(self.config.save_dir_by_rep, 'model.ckpt'), global_step=step)
        # self.saver.save(sess, os.path.join(self.config.save_dir_by_rep, 'model.h5'), global_step=step)


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
        # pdb.set_trace()
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



            # pdb.set_trace()
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(cost)
            #pdb.set_trace()
            return train_op, cost
    @define_scope
    def optimize_with_diff_LR(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
           
            pred = self.prediction
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, 
                labels=tf.one_hot(self.target, self.config.n)))


            ##Using different learning rates for different layers, pass the different layer parameters in the Optimizers
            ##Changing the Learning Rates for C.N.N layers to be 0.001 and for F.C layer to be 0.1 : 
            variables = tf.trainable_variables()
            pdb.set_trace()
            sess = tf.Session()
            graph = tf.get_default_graph()
            # conv1_params_w = sess.graph.get_tensor_by_name("prediction/conv1/conv_weights:0")
            # conv1_params_b = sess.graph.get_tensor_by_name("prediction/conv1/conv_biases:0")

            tf.initialize_all_variables().run()

            conv1_params_w = variables[0]
            conv1_params_b = variables[1]


            conv2_params_w = variables[2]
            conv2_params_b = variables[3]
  

            conv_vars = [conv1_params_w,conv1_params_b,conv2_params_w,conv2_params_b]

            fc1_params_w = variables[4]
            fc1_params_b = variables[5]

            fc3_params_w = variables[6]
            fc3_params_b = variables[7]

            fc_vars = [fc1_params_w,fc1_params_b,fc3_params_w,fc3_params_b]

            opt1 = tf.train.AdamOptimizer(0.001)
            opt2 = tf.train.AdamOptimizer(0.01)

            grads = tf.gradients(cost, conv_vars + fc_vars)

            grads1 = grads[:len(conv_vars)]
            grads2 = grads[len(conv_vars):]

            train_op1 = opt1.apply_gradients(zip(grads1, conv_vars))
            train_op2 = opt2.apply_gradients(zip(grads2, fc_vars))

            train_op = tf.group(train_op1, train_op2)
            tf.initialize_all_variables().run()


            optimizer = train_op

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer
            # pdb.set_trace()
            return train_op, cost

    @define_scope(scope='stream_metrics')
    def metrics(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            pred = self.prediction
            acc, update_acc = tf.metrics.accuracy(self.target, tf.argmax(_softmax(pred), axis=1))
            return update_acc

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

class IsoletWeightTransferModel(WeightTransferModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 617])
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
            x = _relu(_fully_connected('fc1', x, 128))
            x = _relu(_fully_connected('fc2', x, 64))
            x1 = lambda: _fully_connected('fc3', x, self.config.n)
            x2 = lambda: _fully_connected('fc4', x, self.config.n)
            x = tf.cond(tf.equal(self.is_task1, tf.constant(True)), x1, x2)
            return x

class OmniglotWeightTransferModel(WeightTransferModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.target = tf.placeholder(tf.int32, [None])
        self.is_task1 = tf.placeholder(tf.bool)
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.9, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize
        self.metrics

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = tf.reshape(x, [-1, 28, 28, 1])
            x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
            x = tf.contrib.layers.flatten(x)
            x = _relu(_fully_connected('fc1', x, 128))
            x1 = lambda: _fully_connected('fc2', x, self.config.n)
            x2 = lambda: _fully_connected('fc3', x, self.config.n)
            x = tf.cond(tf.equal(self.is_task1, tf.constant(True)), x1, x2)
            return x

class TinyImageNetWeightTransferModel(WeightTransferModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.target = tf.placeholder(tf.int32, [None])
        self.is_task1 = tf.placeholder(tf.bool)
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.9, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize
        self.metrics

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _max_pooling('pool3', _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _relu(self.batch_norm(_conv('conv4', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
            x = tf.contrib.layers.flatten(x)
            x = _relu(_fully_connected('fc1', x, 128))
            x1 = lambda: _fully_connected('fc2', x, self.config.n)
            x2 = lambda: _fully_connected('fc3', x, self.config.n)
            x = tf.cond(tf.equal(self.is_task1, tf.constant(True)), x1, x2)
            return x

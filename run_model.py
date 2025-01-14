import logging
import math
import numpy as np
import os
import random
import tensorflow as tf
import pdb


from data.mnist.read_mnist import MNIST
from data.isolet.read_isolet import Isolet
from data.omniglot.read_omniglot import Omniglot
from data.tiny_imagenet.read_tiny_imagenet import TinyImageNet
from data.episode_generator import generate_training_episode, generate_evaluation_episode 
from losses.histogram_loss import train_batches
from models.proto_model import *
from models.weight_transfer_model import *
from models.baseline_model import *
from models.recall_at_kappa import recall_at_kappa_leave_one_out, recall_at_kappa_support_query
from utils import classification_batch_evaluation, hist_loss_batch_eval, proto_episodic_performance, proto_performance

def train_classification(sess, model, data, params, weight_transfer=True):
    (x_train, y_train), (x_valid, y_valid), (x_train2, y_train2), (x_test2, y_test2) = data
    flag = False
    ##len(x_train) = 8000
    ##len(y_train) = 8000
    ##len(x_valid) =3000
    ##len(x_train2) =50
    ##len(y_train2) =50
    ##len(x_test2) =10000
    ##len(y_test2) = 10000
    # pdb.set_trace()
  
    temp_learning_rate_source_training = params['learning_rate']
    if weight_transfer:
        initial_best_epoch = {'epoch': -1, 'valid_acc': -1}
#     with tf.Session() as sess:
    folder_to_output_file_to  = "/home/abhishek/Desktop/V2_MNIST_" + str(params["k"])+"_"+str(params["n"])
    # pdb.set_trace()
    #path_of_the_source_training = os.path.join("/home/abhishek/Desktop","V2_MNIST_",str(params["k"]),str(params["n"])) +".txt"
    # with open("/home/abhishek/Desktop/{}_{}_{}.txt".format(params["dataset"],params["k"],params["n"]), "a") as f:
    # with open(folder_to_output_file_to+"/{}_{}_{}.txt".format(params["dataset"],params["k"],params["n"]), "a") as f:
    
    ##If the source training has been done already ,skip the following :
    if (os.path.exists("/home/abhishek/Desktop/common_source_training/SOURCE_TRAINING.txt")):
        print("Source has already been trained")
        # with tf.Session() as sess:
            # tf.reset_default_graph()
        # saver = tf.train.Saver()
        # # Restore variables from disk.
        # saver.restore(sess, "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/MODEL/model.ckpt-2")
        # print("Model restored.")

    else:
        with open("/home/abhishek/Desktop/common_source_training/SOURCE_TRAINING.txt", "a") as f:
            print("SOURCE -TRAINING BEGINS",file =f)
            if flag==False:
                # for epoch in range(1,3):
                for epoch in range(1, params['epochs'] + 1):
                    shuffle = np.random.permutation(len(y_train))
                    x_train, y_train = x_train[shuffle], y_train[shuffle]


                    for i in range(0, len(y_train), params['batch_size']):
                        x_train_mb, y_train_mb = x_train[i:i + params['batch_size']], y_train[i:i + params['batch_size']]

        #                 sess.run(model.optimize, feed_dict={model.input: x_train_mb, model.target: y_train_mb, model.is_task1: True, model.is_train: True, model.learning_rate: params['learning_rate']})
                        sess.run(model.optimize, feed_dict={model.input: x_train_mb, model.target: y_train_mb, model.is_task1: True, model.is_train: True, model.learning_rate: temp_learning_rate_source_training})
        #                 pdb.set_trace()


                    valid_acc = classification_batch_evaluation(sess, model, model.metrics, params['batch_size'], True, x_valid, y=y_valid, stream=True)
                    ##model.metrics = <tf.Tensor 'stream_metrics/accuracy/update_op:0' shape=() dtype=float32>

                    print('valid [{} / {}] valid accuracy: {} learning Rate :{}'.format(epoch, params['epochs'] + 1, valid_acc,temp_learning_rate_source_training),file =f)
                    print('valid [{} / {}] valid accuracy: {} learning Rate :{}'.format(epoch, params['epochs'] + 1, valid_acc,temp_learning_rate_source_training))
                    logging.info('valid [{} / {}] valid accuracy: {}'.format(epoch, params['epochs'] + 1, valid_acc))

                    if valid_acc > initial_best_epoch['valid_acc']:
                        initial_best_epoch['epoch'] = epoch
                        initial_best_epoch['valid_acc'] = valid_acc
                        model.saver.save(sess, os.path.join("/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/common_source_model", 'model.ckpt'), global_step=epoch)
                  
                        # model.save_model(sess, epoch) 
                        # model.get_last_sourceTraining_checkpoint()
                        ##Saves the model at the following location : 
                        ##trained_models/mnist/mnist_10_5/weight_transfer/replication1

                        
                        ##Trying to save model as an HDF5 file
        #                 model.save('my_model.h5')

                    if epoch - initial_best_epoch['epoch'] >= params['patience']:
                        print('Early Stopping Epoch: {}\n'.format(epoch))
                        logging.info('Early Stopping Epoch: {}\n'.format(epoch))
                        break

            print('Initial training done \n',file=f)
            logging.info('Initial training done \n')


            # model.restore_model(sess) ##Restores the model after creating it .

        #pdb.set_trace()
    ##Restoring the model : 
    saver = tf.train.Saver()
    # Restore variables from disk.
    all_files = os.listdir("/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/common_source_model")
    model_path = os.path.splitext("/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/common_source_model/"+all_files[0])[0]
    saver.restore(sess, model_path)
    # saver.restore(sess, "/home/abhishek/Desktop/ANU/comp_6470/adapted_deep_embeddings/trained_models/mnist/MODEL/model.ckpt-36")
    print("Model restored.")
    flag =True
    transfer_best_epoch = {'epoch': -1, 'train_acc': -1, 'test_acc': -1}
    es_acc = 0.0
#     model.freeze = True
#     print("Model parameters are ",model.freeze)

    ##Getting the trainable variables : 
    ''' [<tf.Variable 'prediction/conv1/conv_weights:0' shape=(3, 3, 1, 32) dtype=float32_ref>, 
        <tf.Variable 'prediction/conv1/conv_biases:0' shape=(32,) dtype=float32_ref>, 
        <tf.Variable 'prediction/conv2/conv_weights:0' shape=(3, 3, 32, 32) dtype=float32_ref>,
        <tf.Variable 'prediction/conv2/conv_biases:0' shape=(32,) dtype=float32_ref>,
        <tf.Variable 'prediction/fc1/fc_weights:0' shape=(6272, 128) dtype=float32_ref>,
        <tf.Variable 'prediction/fc1/fc_biases:0' shape=(128,) dtype=float32_ref>, 
        <tf.Variable 'prediction/fc3/fc_weights:0' shape=(128, 5) dtype=float32_ref>, 
        <tf.Variable 'prediction/fc3/fc_biases:0' shape=(5,) dtype=float32_ref>,
        <tf.Variable 'prediction/fc4/fc_weights:0' shape=(128, 5) dtype=float32_ref>,
        <tf.Variable 'prediction/fc4/fc_biases:0' shape=(5,) dtype=float32_ref>]
    ''' 
    ##Specify the Learning Rates for different layers : 
    model.learning_rate_CNN = 0.001
    


    for temp_learning_rate_target_training in (0.005,0.001,0.01): ## break after first iteration

        for decay_after_epoch in (10,3,5): 
            train_loss = []
            test_loss = []
            learning_rate = temp_learning_rate_target_training
            model.learning_rate_FN = temp_learning_rate_target_training
            # model.restore_model_SOURCE_Trained(sess) ##Restores the model after creating it .
        

            # with open("/home/abhishek/Desktop/{}_{}_{}.txt".format(params["dataset"],params["k"],params["n"])) as f1:
            # with open(folder_to_output_file_to+"/{}_{}_{}.txt".format(params["dataset"],params["k"],params["n"])) as f1:
            with open("/home/abhishek/Desktop/common_source_training/SOURCE_TRAINING.txt", "r+") as f1:
                with open("/home/abhishek/Desktop/Results_Exp2"+"/{}_{}_{}_{}_{}.txt".format(params["dataset"],params["k"],params["n"],temp_learning_rate_target_training,decay_after_epoch), "a") as f:
                    for x in f1.readlines():
                        f.write(x)
                    print("Target Training Begins",file=f)
                    # for epoch in range(1, 3):
                    for epoch in range(1, params['epochs'] + 1):
                        shuffle = np.random.permutation(len(y_train2))
                        x_train2, y_train2 = x_train2[shuffle], y_train2[shuffle]


                        if epoch%decay_after_epoch==0 and epoch <=decay_after_epoch:
                            learning_rate = learning_rate *0.1
                            model.learning_rate_FN = model.learning_rate_FN * 0.1
                        elif (epoch-decay_after_epoch)%30==0:
                            learning_rate = learning_rate *0.1
                            model.learning_rate_FN = model.learning_rate_FN * 0.1

                        
                        for i in range(0, len(y_train2), params['batch_size']):
                            x_train_mb, y_train_mb = x_train2[i:i + params['batch_size']], y_train2[i:i + params['batch_size']]
                            sess.run(model.optimize_with_diff_LR, feed_dict={model.input: x_train_mb, model.target: y_train_mb, model.is_task1: False, model.is_train: True, model.learning_rate: learning_rate})

                        train_acc = classification_batch_evaluation(sess, model, model.metrics, params['batch_size'], False, x_train2, y=y_train2, stream=True)
                        # sess.close()
                        train_loss = 1- train_acc
                        logging.info('train [{} / {}] train accuracy: {}'.format(epoch, params['epochs'] + 1, train_acc))
                        test_acc = classification_batch_evaluation(sess, model, model.metrics, params['batch_size'], False, x_test2, y=y_test2, stream=True)
                        test_loss_temp = 1- test_acc
                        test_loss.append(test_loss_temp)
                        print('train [{} / {}] train accuracy: {} train losss:{} test accuracy :{} test loss :{} learning Rate:{} '.format(epoch, params['epochs'] + 1, train_acc,train_loss,test_acc,test_loss_temp,model.learning_rate_FN),file=f)
                        print('train [{} / {}] train accuracy: {} train losss:{} test accuracy :{} test loss :{} learning Rate:{} '.format(epoch, params['epochs'] + 1, train_acc,train_loss,test_acc,test_loss_temp,model.learning_rate_FN))

                        if epoch >4 :
                            if(test_loss[epoch-1]==test_loss[epoch-5]):
                                if(test_loss[epoch-2]==test_loss[epoch-4]):
                                    if(test_loss[epoch-1]==test_loss[epoch-3]):
                                        if(test_loss[epoch-1]==test_loss[epoch-2]):
                                            print("Early Stopping",file =f)
                                            print("Early Stopping ")
                                            break

                            
            
                    print('Transfer training done \n',file=f)
                    print('TARGET test accuracy: {}'.format(transfer_best_epoch['test_acc']),file=f)
                    logging.info('Transfer training done \n')
                    logging.info('test accuracy: {}'.format(transfer_best_epoch['test_acc']))  
                    break  
        break  ## break out after 1st iteration only 
def transfer_learningA():
	pass

def train_histogram_loss(sess, model, data, params):
    (x_train, y_train), (x_valid, y_valid), (x_train2, y_train2), (x_test2, y_test2) = data

    initial_best_epoch = {'epoch': -1, 'valid_acc': -1, 'test_acc': -1}

    for epoch in range(1, params['epochs'] + 1):
        shuffle = np.random.permutation(len(y_train))
        x_train, y_train = x_train[shuffle], y_train[shuffle]
        for batch_id, fd_items in train_batches(x_train, y_train, params['batch_size']):
            feed_dict = {
                model.input: fd_items['x'],
                model.pos_comps: fd_items['pos_comps'],
                model.neg_comps: fd_items['neg_comps'],
                model.n_pos_comps: fd_items['n_pos_comps'],
                model.n_neg_comps: fd_items['n_neg_comps'],
                model.is_train: True,
                model.learning_rate: params['learning_rate']
            }
            sess.run(model.optimize, feed_dict=feed_dict)

        valid_norm_preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], x_valid)
        valid_recall_at_one = np.mean(recall_at_kappa_leave_one_out(valid_norm_preds, y_valid, kappa=params['kappa'], dist=params['dist']))

        print('valid [{} / {}] valid accuracy: {}'.format(epoch, params['epochs'] + 1, valid_recall_at_one))
        logging.info('valid [{} / {}] valid accuracy: {}'.format(epoch, params['epochs'] + 1, valid_recall_at_one))

        if valid_recall_at_one > initial_best_epoch['valid_acc']:
            initial_best_epoch['epoch'] = epoch
            initial_best_epoch['valid_acc'] = valid_recall_at_one

            if not params['adaptive'] or params['k'] <= 1:
                preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], np.concatenate((x_train2, x_test2)))
                preds = np.split(preds, [len(x_train2), len(x_train2) + len(x_test2)])
                support_preds = preds[0]
                query_preds = preds[1]
                test_recall_at_one = np.mean(recall_at_kappa_support_query(support_preds, y_train2, query_preds, y_test2, kappa=params['kappa'], dist=params['dist']))
                initial_best_epoch['test_acc'] = test_recall_at_one

            model.save_model(sess, epoch)

        if epoch - initial_best_epoch['epoch'] >= params['patience']:
            print('Early Stopping Epoch: {}\n'.format(epoch))
            logging.info('Early Stopping Epoch: {}\n'.format(epoch))
            break

    if not params['adaptive'] or params['k'] <= 1:
        print('Optimization Finished \n')
        print('test accuracy: {}'.format(initial_best_epoch['test_acc']))
        logging.info('Optimization Finished \n')
        logging.info('test accuracy: {}'.format(initial_best_epoch['test_acc']))
        return

    print('Initial training done \n')
    logging.info('Initial training done \n')

    model.restore_model(sess)

    transfer_best_epoch = {'epoch': -1, 'train_acc': -1, 'test_acc': -1}
    es_acc = 0.0

    for epoch in range(1, params['epochs'] + 1):
        shuffle = np.random.permutation(len(x_train2))
        x_train2, y_train2 = x_train2[shuffle], y_train2[shuffle]
        for batch_id, fd_items in train_batches(x_train2, y_train2, params['batch_size']):
            feed_dict = {
                model.input: fd_items['x'],
                model.pos_comps: fd_items['pos_comps'],
                model.neg_comps: fd_items['neg_comps'],
                model.n_pos_comps: fd_items['n_pos_comps'],
                model.n_neg_comps: fd_items['n_neg_comps'],
                model.is_train: True,
                model.learning_rate: params['learning_rate']
            }
            sess.run(model.optimize, feed_dict=feed_dict)

        train_norm_preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], x_train2)
        train_recall_at_one = np.mean(recall_at_kappa_leave_one_out(train_norm_preds, y_train2, kappa=params['kappa'], dist=params['dist']))

        print('train [{} / {}] train accuracy: {}'.format(epoch, params['epochs'] + 1, train_recall_at_one))
        logging.info('train [{} / {}] train accuracy: {}'.format(epoch, params['epochs'] + 1, train_recall_at_one))

        if train_recall_at_one > transfer_best_epoch['train_acc']:
            transfer_best_epoch['epoch'] = epoch
            transfer_best_epoch['train_acc'] = train_recall_at_one
            preds = hist_loss_batch_eval(sess, model, model.prediction, params['batch_size'], np.concatenate((x_train2, x_test2)))
            preds = np.split(preds, [len(x_train2), len(x_train2) + len(x_test2)])
            support_preds = preds[0]
            query_preds = preds[1]
            test_recall_at_one = np.mean(recall_at_kappa_support_query(support_preds, y_train2, query_preds, y_test2, kappa=params['kappa'], dist=params['dist']))
            transfer_best_epoch['test_acc'] = test_recall_at_one

        if epoch % params['patience'] == 0:
            acc_diff = transfer_best_epoch['train_acc'] - es_acc
            if acc_diff < params['percentage_es'] * es_acc:
                print('Early Stopping Epoch: {}\n'.format(epoch))
                logging.info('Early Stopping Epoch: {}\n'.format(epoch))
                break
            es_acc = transfer_best_epoch['train_acc']

    print('Transfer training done \n')
    print('test accuracy: {}'.format(transfer_best_epoch['test_acc']))
    logging.info('Transfer training done \n')
    logging.info('test accuracy: {}'.format(transfer_best_epoch['test_acc']))

def train_proto_nets(sess, model, data, params):
    (x_train, y_train), (x_valid, y_valid), (x_train2, y_train2), (x_test2, y_test2) = data

    i = 1
    best_episode = {'episode': -1, 'valid_acc': -1, 'test_acc': -1}
    for support_batch, query_batch, query_labels_batch in generate_training_episode(x_train, y_train, params['classes_per_episode'], params['k'], params['query_train_per_class'], params['training_episodes'], batch_size=params['query_batch_size']):
        feed_dict = {
            model.query: query_batch,
            model.label: query_labels_batch,
            model.is_train: True,
            model.learning_rate: params['learning_rate']
        }

        if params['dataset'] == 'tiny_imagenet':
            prototypes = model.compute_batch_prototypes(sess, support_batch, params['classes_per_episode'])
            feed_dict[model.p] = prototypes
        else:
            feed_dict[model.support] = support_batch

        sess.run(model.optimize, feed_dict=feed_dict)

        if i % 200 == 1:
            valid_cost, valid_acc = proto_performance(sess, model, x_train, y_train, x_valid, y_valid, batch_size=params['query_batch_size'])
            valid_cost, valid_acc = float(valid_cost), float(valid_acc)
            print('valid [{}] valid cost: {} valid accuracy: {}'.format(i, valid_cost, valid_acc))
            logging.info('valid [{}] valid cost: {} valid accuracy: {}'.format(i, valid_cost, valid_acc))

            if valid_acc > best_episode['valid_acc']:
                best_episode['episode'] = i
                best_episode['valid_acc'] = valid_acc

                if not params['adaptive'] or params['k'] <= 1:
                    test_cost, test_acc = proto_performance(sess, model, x_train2, y_train2, x_test2, y_test2, batch_size=params['query_batch_size'])
                    best_episode['test_acc'] = float(test_acc)

                model.save_model(sess, i)

            if i - best_episode['episode'] >= params['patience']:
                print('Early Stopping Episode: {}\n'.format(i))
                logging.info('Early Stopping Episode: {}\n'.format(i))
                break

        i += 1

    if not params['adaptive'] or params['k'] <= 1:
        print('Optimization Finished \n')
        print('test accuracy: {}'.format(best_episode['test_acc']))
        logging.info('Optimization Finished \n')
        logging.info('test accuracy: {}'.format(best_episode['test_acc']))
        return

    print('Initial training done \n')
    logging.info('Initial training done \n')

    i = 1
    model.restore_model(sess)
    # Let 75% of the k points be used as support and rest as query when adapting
    episode_support = math.floor(0.75 * params['k'])
    episode_query = params['k'] - episode_support

    transfer_best_episode = {'episode': -1, 'train_acc': -1, 'test_acc': -1}
    es_acc = 0.0

    for support_batch, query_batch, query_labels_batch in generate_training_episode(x_train2, y_train2, params['classes_per_episode'], episode_support, episode_query, params['training_episodes'], batch_size=params['query_batch_size']):
        feed_dict = {
            model.query: query_batch,
            model.label: query_labels_batch,
            model.is_train: True,
            model.learning_rate: params['learning_rate']
        }

        if params['dataset'] == 'tiny_imagenet':
            prototypes = model.compute_batch_prototypes(sess, support_batch, params['classes_per_episode'])
            feed_dict[model.p] = prototypes
        else:
            feed_dict[model.support] = support_batch
        
        sess.run(model.optimize, feed_dict=feed_dict)

        if i % 200 == 1:
            train_perf, train_std = proto_episodic_performance(sess, model, x_train2, y_train2, params['classes_per_episode'], episode_support, episode_query, params['query_batch_size'], params['evaluation_episodes'])
            train_perf[0] = float(train_perf[0])
            train_perf[1] = float(train_perf[1])
            print('train [{}] train cost: {} train accuracy: {}'.format(i, train_perf[1], train_perf[0]))
            logging.info('train [{}] train cost: {} train accuracy: {}'.format(i, train_perf[1], train_perf[0]))

            if train_perf[0] > transfer_best_episode['train_acc']:
                transfer_best_episode['episode'] = i
                transfer_best_episode['train_acc'] = train_perf[0]
                test_cost, test_acc = proto_performance(sess, model, x_train2, y_train2, x_test2, y_test2, batch_size=params['query_batch_size'])
                transfer_best_episode['test_acc'] = float(test_acc)

        if i % params['patience'] == 0:
            acc_diff = transfer_best_episode['train_acc'] - es_acc
            if acc_diff < params['percentage_es'] * es_acc:
                print('Early Stopping Episode: {}\n'.format(i))
                logging.info('Early Stopping Episode: {}\n'.format(i))
                break
            es_acc = transfer_best_episode['train_acc']

        i += 1

    print('Transfer training done \n')
    print('test accuracy: {}'.format(transfer_best_episode['test_acc']))
    logging.info('Transfer training done \n')
    logging.info('test accuracy: {}'.format(transfer_best_episode['test_acc']))

def get_model(params):
    model, data = None, None
    if params['command'] == 'hist_loss':
        if params['dataset'] == 'mnist':
            model = MNISTHistModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'], 
                params['k'], params['n'], params['t2_test'])
        elif params['dataset'] == 'isolet':
            model = IsoletHistModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotHistModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetHistModel(params)
            data = TinyImageNet(params['data_path']).kntl_data_form(350, params['n'], params['k'], params['n'])
    elif params['command'] == 'proto':
        if params['dataset'] == 'mnist':
            model = MNISTProtoModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'], 
                params['k'], params['n'], params['t2_test'])
        elif params['dataset'] == 'isolet':
            model = IsoletProtoModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotProtoModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetProtoModel(params)
            data = TinyImageNet(params['data_path']).kntl_data_form(350, params['n'], params['k'], params['n'])
    elif params['command'] == 'weight_transfer':
        if params['dataset'] == 'mnist':
            model = MNISTWeightTransferModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'],params['k'], params['n'], params['t2_test'])
	    
	   
        elif params['dataset'] == 'isolet':
            model = IsoletWeightTransferModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotWeightTransferModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetWeightTransferModel(params)
            data = TinyImageNet(params['data_path']).kntl_data_form(350, params['n'], params['k'], params['n'])
	    
    elif params['command'] == 'baseline':
        if params['dataset'] == 'mnist':
            model = MNISTBaselineModel(params)
            data = MNIST(params['data_path']).kntl_data_form(params['t1_train'], params['t1_valid'], 
                params['k'], params['n'], params['t2_test'])
        elif params['dataset'] == 'isolet':
            model = IsoletBaselineModel(params)
            data = Isolet(params['data_path']).kntl_data_form(250, params['n'], params['k'], params['n'])
        elif params['dataset'] == 'omniglot':
            model = OmniglotBaselineModel(params)
            data = Omniglot(params['data_path']).kntl_data_form(params['n'], params['k'], params['n'])
        else:
            model = TinyImageNetBaselineModel(params)
            data = TinyImageNet(params['data_path']).kntl_data_form(350, params['n'], params['k'], params['n'])
    else:
        print('Unknown model type')
        logging.debug('Unknown model type')
        quit()
    #pdb.set_trace()
    return model, data

def run(params):
    params = vars(params)
#     {'dataset': 'mnist', 'data_path': '/home/abhishek/Desktop/Research_Papers/comp_6470/adapted_deep_embeddings/datasets/mnist', 't1_train': 8000, 't1_valid': 3000, 'k': 10, 'n': 5, 't2_test': 10000, 'epochs': 500, 'batch_size': 2048, 'learning_rate': 0.005, 'patience': 20, 'percentage_es': 0.01, 'random_seed': 1234, 'replications': 1, 'gpu': '0', 'controller': '/cpu:0', 'save_dir': '/home/abhishek/Desktop/Research_Papers/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_10_5/weight_transfer', 'log_file': '/home/abhishek/Desktop/Research_Papers/comp_6470/adapted_deep_embeddings/trained_models/mnist/mnist_10_5/weight_transfer/log.txt', 'command': 'weight_transfer'}

    logging.info(params)

    random.seed(params['random_seed'])
    initialization_seq = random.sample(range(50000), params['replications']) 
    ##initialization_seq = 28883

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    #pdb.set_trace()
    for rep in range(params['replications']):
        tf.reset_default_graph()  #This does siemthing like clearing the default graph stack and i have no idea what does that mean 
        with tf.Session(config=config) as sess:
            #tf.set_random_seed(initialization_seq[rep])
            np.random.seed(initialization_seq[rep])
            
            model, data = get_model(params)
            #model : weight_transfer_model.MNISTWeightTransferModel
         
            assert not model is None
            assert not data is None
            
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            params['init'] = init
            model.create_saver()
            #self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

            sess.run(init)
            
            rep_path = os.path.join(params['save_dir'], 'replication{}'.format(rep + 1))
            os.mkdir(rep_path)
            model.config['save_dir_by_rep'] = rep_path

            logging.debug('running training/testing')
            #pdb.set_trace()
            if params['command'] == 'baseline':
                train_classification(sess, model, data, params, weight_transfer=False)
            elif params['command'] == 'weight_transfer':
                train_classification(sess, model, data, params, weight_transfer=True)
            elif params['command'] == 'hist_loss':
                train_histogram_loss(sess, model, data, params)
            elif params['command'] == 'proto':
                train_proto_nets(sess, model, data, params)
            else:
                print('Unknown model type')
                logging.debug('Unknown model type')
                quit()
            #pdb.set_trace()
            print("Inside run(params): in run_model.py")
            

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import pickle

import math
import os
import random
import sys
import time
from typing import List, Any

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import json
from metrics import compute_metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
# import UPRR_sinkhorn_parallel as CM
import URCC
from click_model import ClickModel
from sklearn.metrics import roc_auc_score, log_loss




# rank list size should be read from data
# tf.app.flags.DEFINE_string("data_dir", "../sinkhorn4rerank/data/Yahoo/", "Data directory")
# tf.app.flags.DEFINE_string("train_dir", "./model/Yahoo_train/", "Training directory")
# tf.app.flags.DEFINE_string("click_dir", "./model/Yahoo_train/click/", "click model directory(for ranker)")
tf.app.flags.DEFINE_string("data_dir", "../sinkhorn4rerank/data/MSLR10K/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model/MSLR10K_train/", "Training directory")
tf.app.flags.DEFINE_string("click_dir", "./model/MSLR10K_train/click/urge_dnn_learning_rate=5e-4_batch_64", "click model directory(for ranker)")
tf.app.flags.DEFINE_string("decode_dir", "./model/MSLR_train/ranker/urge_dnn_learning_rate=5e-4_batch_64", "Directory for decode.")
tf.app.flags.DEFINE_string("hparams", "", "Hyper-parameters for models.")

tf.app.flags.DEFINE_string("train_stage", 'ranker',
                           "traing stage.")

# tf.app.flags.DEFINE_string("init_ranker", 'SVM',
#                            "initial ranker")
# tf.app.flags.DEFINE_string("init_ranker", 'lambdaMART',
#                            "initial ranker")
tf.app.flags.DEFINE_string("init_ranker", 'dnn',
                           "initial ranker")

tf.app.flags.DEFINE_integer("batch_size", 512,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("np_random_seed", 1385,
                            "random seed for numpy")
tf.app.flags.DEFINE_integer("tf_random_seed", 20933,
                            "random seed for tensorflow")
tf.app.flags.DEFINE_integer("emb_size", 10,
                            "Embedding to use during training.")
tf.app.flags.DEFINE_integer("train_list_cutoff", 10,
                            "The number of documents to consider in each list during training.")
tf.app.flags.DEFINE_integer("max_train_iteration", 0,
                            "Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding data.")
tf.app.flags.DEFINE_boolean("decode_train", False,
                            "Set to True for decoding training data.")
tf.app.flags.DEFINE_boolean("decode_valid", False,
                            "Set to True for decoding valid data.")
# To be discarded.
tf.app.flags.DEFINE_boolean("feed_previous", False,
                            "Set to True for feed previous internal output for training.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Set to True for test program.")

FLAGS = tf.app.flags.FLAGS

tf.set_random_seed(FLAGS.tf_random_seed)
np.random.seed(FLAGS.np_random_seed)


def create_model(session, data_set, forward_only, ckpt = None):
    """Create model and initialize or load parameters in session."""
    # print('create model', data_set.user_field_M, data_set.item_field_M)

    model = URCC.URCC(data_set.rank_list_size, data_set.embed_size, FLAGS.batch_size, FLAGS.hparams,
                    forward_only, train_stage=FLAGS.train_stage)


    print(ckpt)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    if not ckpt:
        if FLAGS.train_stage == 'ranker':
            ckpt = tf.train.get_checkpoint_state(FLAGS.click_dir)
            print('reloading partial parameters from %s' % ckpt.model_checkpoint_path)
            click_variables_to_restore = [v for v in tf.global_variables() if
                                          v.name.split('/')[0] == 'ClickRelNet' or v.name.split('/')[
                                              0] == 'ClickExamNet' or v.name.split('/')[0] == 'ClickNet' or v.name.split('/')[0] == 'GAT']
            saver = tf.train.Saver(click_variables_to_restore)
            saver.restore(session, ckpt.model_checkpoint_path)
        elif FLAGS.train_stage == 'click':
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + 'click/')
    if not forward_only:
        model.global_gradient_update()
        tf.summary.scalar('Loss', tf.reduce_mean(model.loss))
        # tf.summary.scalar('Gradient Norm', model.norm)
        tf.summary.scalar('Learning Rate', model.learning_rate)
        tf.summary.scalar('Final Loss', tf.reduce_mean(model.loss))
    model.summary = tf.summary.merge_all()

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    initialize_op = tf.variables_initializer(uninitialized_vars)
    session.run(initialize_op)

    return model


def train():
    # Prepare data.
    print("Reading data in %s" % FLAGS.data_dir)
    if FLAGS.init_ranker == 'SVM':
        ranker_name = 'ranker'
    elif FLAGS.init_ranker == 'lambdaMART':
        ranker_name = 'ranker_mart'
    elif FLAGS.init_ranker == 'dnn':
        ranker_name = 'ranker_dnn'
    else:
        print('initial ranker does not exist')
        exit()

    train_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff, ranker_name=ranker_name)
    valid_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff, ranker_name=ranker_name)
    # rand_valid_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff, ranker=['randRanker1', 'randRanker2'])
    print("Rank list size %d" % train_set.rank_list_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating model...")
        model = create_model(sess, train_set, False)

        if FLAGS.train_stage == 'ranker':
            check_point_dir = FLAGS.train_dir + FLAGS.train_stage + '/urcc_' + FLAGS.init_ranker + '_' + str(
                FLAGS.hparams) + '_batch_' + str(FLAGS.batch_size)
        if FLAGS.train_stage == 'click':
            check_point_dir = FLAGS.train_dir + FLAGS.train_stage + '/urcc_' + FLAGS.init_ranker + '_' + str(
                FLAGS.hparams)+ '_batch_' + str(FLAGS.batch_size)
        print(check_point_dir)

        # Create tensorboard summarizations.
        train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_log',
                                             sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.train_dir + '/valid_log')
        if not os.path.exists(check_point_dir):
            print('mkdir', check_point_dir)
            os.makedirs(check_point_dir)
        log_file = open(check_point_dir + '/output.log', 'a')

        if FLAGS.train_stage == 'click':
            # Training of bias model
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            best_bias_auc = 0
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                input_feed = model.get_batch_for_click(train_set)
                step_loss, summary, auc, acc, click, log_loss = model.click_step(sess, input_feed, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                train_writer.add_summary(summary, current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:

                    # Print statistics for the previous epoch.
                    # loss = math.exp(loss) if loss < 300 else float('inf')
                    print("global step %d learning rate %.4f step-time %.2f loss "
                          "%.2f auc %.4f acc %.4f log_loss %.4f" % (tf.convert_to_tensor(model.global_step).eval(),
                                                                   tf.convert_to_tensor(model.learning_rate).eval(),
                                                                   step_time, loss, auc, acc, log_loss))
                    print("global step %d learning rate %.4f step-time %.2f loss "
                          "%.2f auc %.4f acc %.4f log_loss %.4f" % (tf.convert_to_tensor(model.global_step).eval(),
                                                                   tf.convert_to_tensor(model.learning_rate).eval(),
                                                                   step_time, loss,
                                                                   auc, acc, log_loss), file=log_file, flush=True)


                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    # Validate model
                    it, batch_num = 0, 0
                    bias_loss, bias_acc, bias_auc, bias_log_loss = 0, 0, 0, 0
                    while batch_num < valid_set.user_num / FLAGS.batch_size:
                        input_feed1 = model.get_batch_for_click_by_index(valid_set, it)
                        v_loss, summary, auc, acc, click, step_log_loss = model.click_step(sess, input_feed1, True)
                        bias_loss += v_loss
                        bias_log_loss += step_log_loss
                        bias_acc += acc
                        bias_auc += auc
                        it += FLAGS.batch_size
                        batch_num += 1
                    valid_writer.add_summary(summary, current_step)
                    print("bias eval: loss %.4f auc %.4f acc %.4f logloss %.4f" % (
                    bias_loss / batch_num, bias_auc / batch_num, bias_acc/batch_num, bias_log_loss/batch_num))
                    print("bias eval: loss %.4f auc %.4f acc %.4f logloss %.4f" % (
                    bias_loss / batch_num, bias_auc / batch_num, bias_acc/batch_num, bias_log_loss/batch_num), file=log_file, flush=True)

                    if best_bias_auc <bias_auc / batch_num:
                        best_bias_auc = bias_auc/batch_num
                        checkpoint_path = check_point_dir + "/model.ckpt"
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                    if loss == float('inf'):
                        break

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

                    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                        break

        # Training of the ranker
        if FLAGS.train_stage == 'ranker':

            # Training of bias model
            step_time, loss = 0.0, 0.0
            current_step = 0
            best_click = 0
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                input_feed, labels, clicks = model.get_batch_for_ranker(train_set)
                labels, scores, step_loss, summary, _ = model.ranker_step(sess, input_feed, labels, clicks, forward_only=False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if (current_step < 40) or (current_step % FLAGS.steps_per_checkpoint == 0):
                    # Print statistics for the previous epoch.

                    print("global step %d learning rate %.4f step-time %.2f loss "
                          "%.8f" % (tf.convert_to_tensor(model.global_step).eval(),
                                    tf.convert_to_tensor(model.learning_rate).eval(), step_time, loss))
                    print('on train set')

                    # Validate model
                    it = 0
                    count_batch = 0.0
                    all_labels, all_scores, all_rank, all_deltas = [], [], [], []
                    while it < valid_set.user_num:
                        input_feed, labels, clicks = model.get_batch_for_ranker_by_index(valid_set, it)
                        labels, scores = model.ranker_step(sess, input_feed, labels, clicks, forward_only=True)


                        it += 1
                        count_batch += 1.0
                        all_labels.extend(labels)
                        all_scores.extend(scores)

                    print('on test set')
                    print(current_step)
                    print(current_step, file=log_file, flush=True)
                    click_prob = compute_metrics(all_labels, all_scores, valid_set, None, log_file)
                    # ctr.write(str(click_prob) + ', ')

                    if best_click < click_prob:
                        best_click = click_prob
                        checkpoint_path = check_point_dir + "/model.ckpt"
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    if loss == float('inf') or loss == float("nan"):
                        break

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

                    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                        break


def decode(model_path):
    if FLAGS.init_ranker == 'SVM':
        ranker_name = 'ranker'
    elif FLAGS.init_ranker == 'lambdaMART':
        ranker_name = 'ranker_mart'
    elif FLAGS.init_ranker == 'dnn':
        ranker_name = 'ranker_dnn'
    else:
        print('initial ranker does not exist')
        exit()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        # Load test data.
        print("Reading data in %s" % FLAGS.data_dir)
        test_set = None
        if FLAGS.decode_train:
            test_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff, ranker_name=ranker_name)
        elif FLAGS.decode_valid:
            test_set = data_utils.read_data(FLAGS.data_dir, 'valid', FLAGS.train_list_cutoff, ranker_name=ranker_name)
        else:
            test_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff, ranker_name=ranker_name)

        # Create model and load parameters.
        print(model_path)
        model = create_model(sess, test_set, False, ckpt=tf.train.get_checkpoint_state(model_path))


        # model = create_model(sess, test_set, False)
        model.batch_size = 1  # We decode one sentence at a time.
        if FLAGS.train_stage == 'ranker':
            for i in range(1):
                all_labels, all_scores = [], []
                for it in tqdm(range(int(test_set.user_num))):
                    input_feed, labels, clicks = model.get_batch_for_ranker_by_index(test_set, it)
                    labels, scores = model.ranker_step(sess, input_feed, labels, clicks, forward_only=True)
                    all_labels.extend(labels)
                    all_scores.extend(scores)

                compute_metrics(all_labels, all_scores, test_set, None, None)

        elif FLAGS.train_stage == 'click':
            it, batch_num = 0, 0
            bias_loss, bias_acc, bias_auc, bias_log_loss = 0, 0, 0, 0
            while batch_num < test_set.user_num / FLAGS.batch_size:
                input_feed1 = model.get_batch_for_click_by_index(test_set, it)
                v_loss, summary, auc, acc, click, step_log_loss = model.click_step(sess, input_feed1, True)
                bias_loss += v_loss
                bias_log_loss += step_log_loss
                bias_acc += acc
                bias_auc += auc
                it += FLAGS.batch_size
                batch_num += 1
            print("bias eval: loss %.4f auc %.4f acc %.4f logloss %.4f" % (
                bias_loss / batch_num, bias_auc / batch_num, bias_acc / batch_num, bias_log_loss / batch_num))
    return


def main(_):
    if FLAGS.decode:
        decode(FLAGS.decode_dir)
    else:
        train()


if __name__ == "__main__":
    tf.app.run()

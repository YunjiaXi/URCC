from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from sklearn.metrics import roc_auc_score, log_loss
from tensorflow.python.ops import variable_scope

def sigmoid_prob(logits):
    return tf.sigmoid(logits - tf.reduce_mean(logits, -1, keep_dims=True))


def network(input_data, layer_sizes, scope, is_training, reuse=False, assign_vars=None):
    with variable_scope.variable_scope(scope, reuse=reuse):
        output_data = input_data
        output_sizes = layer_sizes
        current_size = input_data.get_shape().as_list()[-1]
        var_list=[]
        for i in range(len(output_sizes)):
            if assign_vars is None:
                expand_W = variable_scope.get_variable("expand_W_%d" % i, [current_size, output_sizes[i]],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            else:
                expand_W = variable_scope.get_variable("expand_W_%d" % i, initializer=assign_vars[2*i+0])

            var_list.append(expand_W)
            if assign_vars is None:
                expand_b = variable_scope.get_variable("expand_b_%d" % i, [output_sizes[i]])
            else:
                expand_b = variable_scope.get_variable("expand_b_%d" % i, initializer=assign_vars[2*i+1])
            var_list.append(expand_b)
            output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
            # output_data = tf.layers.batch_normalization(output_data, training=is_training)
            output_data = tf.nn.relu(output_data)
            current_size = output_sizes[i]
        return output_data, var_list


class URCC(object):

    def __init__(self, rank_list_size, embed_size, batch_size, hparam_str, forward_only=False, train_stage='click'):
        """Create the model.
		Args:
			rank_list_size: size of the ranklist.
			batch_size: the size of the batches used during training;
						the model construction is not independent of batch_size, so it cannot be
						changed after initialization.
			embed_size: the size of the input feature vectors.
			forward_only: if set, we do not construct the backward pass in the model.
		"""

        self.hparams = tf.contrib.training.HParams(
            learning_rate=1e-5,  # Learning rate.
            learning_rate_decay_factor=0.96,  # Learning rate decays by this much.
            max_gradient_norm=0.0,  # Clip gradients to this norm.
            l2_loss=0.0,  # Set strength for L2 regularization.
            update_target_ranker_interval=1,
            click_hidden_layer_sizes=[1024, 1024, 512, 64], # MSLR
            rnn_hidden_layer_sizes=[512, 256, 64], # MSLR
            ranker_hidden_layer_sizes=[1024, 512, 256, 64],  # MSLR
            gcn_hidden_layer_sizes=[64, 64, 64, 64],
            rnn_hidden_size=64,
            pair_each_query=10,
        )
        print(hparam_str)
        self.hparams.parse(hparam_str)
        self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.hparams.learning_rate_decay_factor)
        self.start_index = 0
        self.count = 1
        self.rank_list_size = rank_list_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.train_stage = train_stage
        self.bias_model_step = tf.Variable(0, trainable=False)
        self.click_model_step = tf.Variable(0, trainable=False)
        self.ranker_model_step = tf.Variable(0, trainable=False)

        self.forward_only = forward_only
        self.data_step = 0
        self.data_permutation = None
        self.item_field_M = self.embed_size + self.hparams.gcn_hidden_layer_sizes[-1]
        # self.item_field_M = self.embed_size
        self.update_target_ranker_interval = self.hparams.update_target_ranker_interval

        #####not use now
        self.update_interval = 200
        self.exam_vs_rel = 20


        # Feeds for inputs.
        self.item_features = tf.placeholder(tf.float32, shape=[None, self.rank_list_size, self.embed_size], name='item_features')
        self.mask = tf.placeholder(tf.float32, shape=[None, self.rank_list_size], name='mask')
        self.list_mask = tf.placeholder(tf.float32, shape=[None, self.rank_list_size, self.rank_list_size], name="list_mask")
        self.pos = tf.placeholder(tf.int32, shape=[None, self.rank_list_size], name='pos')
        self.target_clicks = tf.placeholder(tf.float32, shape=[None, self.rank_list_size], name='target_clicks')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.attn_drop = tf.placeholder(tf.float32, name="attn_drop")
        self.ffd_drop = tf.placeholder(tf.float32, name="ffd_drop")
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.pos_neg_ratio = tf.placeholder(tf.float32, name='pos_neg_ratio')
        self.length = tf.placeholder(tf.int32, shape=[None], name='length')
        if train_stage == 'ranker':
            self.pos_item_feature = tf.placeholder(tf.float32, shape=[None, self.item_field_M],
                                                   name="pos_item_feature")
            self.neg_item_feature = tf.placeholder(tf.float32, shape=[None, self.item_field_M],
                                                   name="neg_item_feature")
            self.deltaR = tf.placeholder(tf.float32, shape=[None], name="delta_revenue")
            self.pair_label = tf.placeholder(tf.float32, shape=[None], name="pair_label")

        self.embeddings = self.gcn_attention_network(self.item_features, scope="GAT")
        self.click, self.click_loss, self.click_acc = self.BiRNNClickNet()
        if self.train_stage == 'ranker':
            self.pos_score = self.RankNet(self.pos_item_feature, scope='ranker')
            self.neg_score = self.RankNet(self.neg_item_feature, scope='ranker')

        if not forward_only:
            # Gradients and SGD update operation for training the model.
            if self.train_stage == 'click':
                self.loss = self.click_loss
                self.global_step = self.click_model_step
            if self.train_stage == 'ranker':
                self.loss = tf.reduce_mean(
                    self.deltaR * self.pair_label * tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=(self.pos_score - self.neg_score),
                        labels=tf.ones_like(self.pair_label)))
                self.global_step = self.ranker_model_step

            self.optimizer_func = tf.train.AdamOptimizer

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        print(' init finish')

    def copy_model_parameters(self, scope1, scope2):
        """
         Copies the model parameters of one estimator to another.
    #
    #     Args:
    #       sess: Tensorflow session instance
    #       ranker1: Estimator to copy the paramters from
    #       ranker2: Estimator to copy the parameters to
    #     """
        r1_params = [t for t in tf.trainable_variables() if t.name.startswith(scope1)]
        r1_params = sorted(r1_params, key=lambda v: v.name)
        r2_params = [t for t in tf.trainable_variables() if t.name.startswith(scope2)]
        r2_params = sorted(r2_params, key=lambda v: v.name)

        self.copy_ops = []
        for r1_v, r2_v in zip(r1_params, r2_params):
            op = r2_v.assign(r1_v)
            self.copy_ops.append(op)

    def gcn_attention_network(self, input_data, scope, reuse=False):
        with variable_scope.variable_scope(scope, reuse=reuse):
            embeddings = input_data
            output_sizes = self.hparams.gcn_hidden_layer_sizes
            current_size = input_data.get_shape().as_list()[-1]
            embeddings_mask = tf.expand_dims(self.mask, -1)
            for i in range(len(output_sizes)):
                gcn_W = variable_scope.get_variable("gat_W_%d" % i, [current_size, output_sizes[i]],
                                                    initializer=tf.contrib.layers.xavier_initializer())

                gcn_b = variable_scope.get_variable("gat_b_%d" % i, [output_sizes[i]])
                w_A = variable_scope.get_variable("w_A_%d" % i, [current_size*2, 1],
                                                  initializer=tf.contrib.layers.xavier_initializer())
                b_A = variable_scope.get_variable("w_b_%d" % i, [1])
                node_num = self.rank_list_size

                a = tf.reshape(tf.tile(embeddings, [1, node_num, 1]), [-1, node_num, node_num, current_size])
                b = tf.reshape(tf.tile(embeddings, [1, 1, node_num]), [-1, node_num, node_num, current_size])
                c = tf.reshape(tf.concat([a, b], 3), [-1, current_size*2])
                value = tf.reshape(tf.matmul(c, w_A) + b_A, [-1, node_num, node_num])
                value = value * (tf.ones((node_num, node_num)) - tf.eye(node_num))
                value = tf.maximum(0.01 * value, value)
                adj_matrix = tf.nn.softmax(value, 1)

                embeddings = tf.matmul(adj_matrix, embeddings)
                embeddings = tf.reshape(embeddings, [-1, current_size])
                embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, gcn_W) + gcn_b)
                embeddings = tf.nn.dropout(embeddings, keep_prob=self.keep_prob)
                embeddings = tf.reshape(embeddings, [-1, self.rank_list_size, output_sizes[i]])
                embeddings = embeddings * tf.tile(embeddings_mask, [1, 1, output_sizes[i]])
                current_size = output_sizes[i]

        return embeddings


    def global_gradient_update(self):
        print('bulid gradient')
        self.l2_loss = tf.Variable(0.0, trainable=False)
        params = tf.trainable_variables()
        # print(params)
        if self.hparams.l2_loss > 0:
        	for p in params:
        		self.l2_loss = self.hparams.l2_loss * tf.nn.l2_loss(p)
        		self.loss += self.l2_loss
        opt = self.optimizer_func(self.hparams.learning_rate)
        # opt = tf.train.GradientDescentOptimizer(self.ranker_learning_rate)
        self.gradients = tf.gradients(self.loss, params)
        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients, self.hparams.max_gradient_norm)
            # self.clipped_gradients = []
            # for x in self.gradients:
            #     self.clipped_gradients.append(x if x is None else tf.clip_by_value(x, -self.hparams.max_gradient_norm, self.hparams.max_gradient_norm))
            self.updates =  opt.apply_gradients(zip(self.clipped_gradients, params), global_step=self.global_step)
        else:
            self.norm = tf.global_norm(self.gradients)
            self.updates = opt.apply_gradients(zip(self.gradients, params), global_step=self.global_step)
        # self.updates = opt.minimize(self.loss, global_step=self.global_step)

    def BiRNNClickNet(self, scope=None):
        with variable_scope.variable_scope(scope or "ClickNet"):
            item_seq = tf.unstack(tf.concat(axis=2, values=[self.embeddings, self.item_features]), self.rank_list_size, 1)
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hparams.rnn_hidden_size, forget_bias=1.0)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hparams.rnn_hidden_size, forget_bias=1.0)
            #
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, item_seq, dtype='float32')
            # outputs, _, = tf.nn.dynamic_rnn(lstm_fw_cell, item_seq, dtype ='float32')
            outputs = tf.stack(outputs, axis=1)
            input = tf.reshape(outputs, (-1, self.rank_list_size, self.hparams.rnn_hidden_size * 2))
            pos_embedding = tf.reshape(tf.one_hot(tf.reshape(self.pos, [-1]), depth=self.rank_list_size),
                                       (-1, self.rank_list_size, self.rank_list_size))
            input = tf.concat(axis=2, values=[input, self.item_features])
            # input = tf.concat(axis=2, values=[input, pos_embedding, self.item_features])
            input = tf.layers.batch_normalization(inputs=input, name='bn1')
            for i, v in enumerate(self.hparams.click_hidden_layer_sizes):
                fc = tf.layers.dense(input, v, activation=tf.nn.relu, name='fc' + str(i), kernel_initializer=tf.contrib.layers.xavier_initializer())
                dp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
                input = dp
            fc_final = tf.layers.dense(input, 10, activation=tf.nn.sigmoid, name='fc_final')
            fc_final = tf.reduce_sum(fc_final * pos_embedding, axis=-1)
            score = tf.reshape(fc_final, [-1, self.rank_list_size])

            # sequence mask
            y_prob = score * self.mask
            label = self.target_clicks

            click_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast((y_prob > 0.5), dtype=tf.int32), tf.cast(label, dtype=tf.int32)),
                        dtype=tf.float32))
            click_loss = tf.losses.log_loss(label, y_prob)
        return y_prob, click_loss, click_acc

    def RankNet(self, item_feature, scope):

        hidden, _ = network(item_feature, self.hparams.ranker_hidden_layer_sizes, scope, is_training=self.is_training,
                            reuse=tf.AUTO_REUSE)
        with variable_scope.variable_scope(scope, reuse=tf.AUTO_REUSE):
            ranker_last_W = variable_scope.get_variable("rankerRel_last_W",
                                                            [self.hparams.ranker_hidden_layer_sizes[-1], 1],
                                                            initializer=tf.contrib.layers.xavier_initializer())
            ranker_last_b = variable_scope.get_variable("rankerRel_last_b", [1])
        output_score = tf.squeeze(tf.nn.bias_add(tf.matmul(hidden, ranker_last_W), ranker_last_b))
        return output_score


    def click_step(self, session, input_feed, forward_only):
        # Output feed: depends on whether we do a backward step or not.
        input_feed[self.keep_prob.name] = 1.0 if forward_only else 0.8
        input_feed[self.is_training.name] = not forward_only
        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.click_loss,  # Loss for this batch.
                           self.summary,  # Summarize statistics.
                           self.click_acc,
                           self.l2_loss,
                           self.click
                           ]
        else:
            output_feed = [self.click_loss,  # Loss for this batch.
                           self.summary,  # Summarize statistics.
                           self.click_acc,
                           self.click
                           ]

        outputs = session.run(output_feed, input_feed)

        auc_score = roc_auc_score(input_feed[self.target_clicks.name].reshape(-1), outputs[-1].reshape(-1))
        logloss = log_loss(input_feed[self.target_clicks.name].reshape(-1), outputs[-1].reshape(-1))
        if not forward_only:
            return outputs[1], outputs[2], auc_score, outputs[3], outputs[-1], logloss  # loss, no outputs, summary.
        else:
            return outputs[0], outputs[1], auc_score, outputs[2], outputs[-1], logloss  # loss, outputs, summary.


    def ranker_step(self, session, input_feed, labels, clicks, forward_only):
        input_feed[self.keep_prob.name] = 1.0
        embeddings = session.run(self.embeddings, input_feed)
        input_size = self.hparams.gcn_hidden_layer_sizes[-1] + self.embed_size
        ranker_item_feature = np.reshape(np.concatenate((embeddings, input_feed[self.item_features.name]), axis=2), [-1, input_size])
        input_feed[self.pos_item_feature.name] = ranker_item_feature
        flatten_scores = np.reshape(session.run(self.pos_score, feed_dict={
            self.pos_item_feature.name:ranker_item_feature
        }), [-1])
        scores = np.reshape(flatten_scores, [-1, self.rank_list_size]).tolist()

        if not forward_only:
            current_poss = []
            origin_poss = []
            rand_pairs = []
            list_rand_pairs = []
            new_item_feature = []
            rank_size = self.rank_list_size
            for i, sublabels in enumerate(labels):
                in_query_scores = scores[i]
                sorted_score = sorted(in_query_scores, reverse=True)
                map = {v: i for i, v in enumerate(sorted_score)}
                current_poss.extend([map[v] for v in in_query_scores])
                origin_poss.extend(np.arange(rank_size))
                clicked_items = np.nonzero(clicks[i])[0]
                list_item_feature = input_feed[self.item_features.name][i].tolist()
                for ii in range(self.hparams.pair_each_query):
                    if clicked_items.shape[0]:
                        random_pair = np.zeros([2], dtype=np.int)
                        random_pair[0] = np.random.choice(clicked_items)
                        random_pair[1] = np.random.choice(rank_size)
                    else:
                        random_pair = np.random.choice(rank_size, 2)
                    new_list = list_item_feature.copy()
                    new_list[random_pair[0]], new_list[random_pair[1]] = new_list[random_pair[1]], new_list[random_pair[0]]
                    new_item_feature.append(new_list)
                    rand_pairs.append(i * rank_size + random_pair)
                    list_rand_pairs.append(random_pair)

            rand_pairs = np.array(rand_pairs)
            list_rand_pairs = np.array(list_rand_pairs)

            input_feed[self.pos.name] = np.reshape(np.array(origin_poss), [-1, rank_size])
            old_click = session.run(self.click, feed_dict=input_feed)
            it_pos = np.tile(input_feed[self.pos.name], [1, self.hparams.pair_each_query])
            it_mask = np.tile(input_feed[self.mask.name], [1, self.hparams.pair_each_query])
            input_feed[self.pos.name] = np.reshape(it_pos, [-1, rank_size])
            input_feed[self.mask.name] = np.reshape(it_mask, [-1, rank_size])
            input_feed[self.item_features.name] = np.array(new_item_feature)
            new_click = session.run(self.click, feed_dict=input_feed)

            # new debias
            tile_labels = np.reshape(np.tile(np.array(clicks), [1, self.hparams.pair_each_query]), [-1, self.rank_list_size])
            tile_old_click = np.reshape(np.tile(old_click, [1, self.hparams.pair_each_query]), [-1, self.rank_list_size])
            idx = np.arange(0, new_click.shape[0])
            new_click[idx, list_rand_pairs[idx, 1]], new_click[idx, list_rand_pairs[idx, 0]] = new_click[idx, list_rand_pairs[idx, 0]], new_click[idx, list_rand_pairs[idx, 1]]
            delta_revenue = np.sum((new_click - tile_old_click) / (tile_old_click + 1e-17) * tile_labels, axis=1, keepdims=False)

            pair_label = ((flatten_scores[rand_pairs[:, 0]] < flatten_scores[rand_pairs[:, 1]]).astype(float) - 0.5) * 2

            _, loss = session.run([self.updates,  # Update Op that does SGD.
                                            self.loss,  # Loss for this batch.
                                    ], feed_dict={
                self.pos_item_feature.name: ranker_item_feature[rand_pairs[:, 0]],
                self.neg_item_feature.name: ranker_item_feature[rand_pairs[:, 1]],
                self.deltaR.name: delta_revenue,
                self.pair_label.name: pair_label})

            return labels, scores, loss,_ , delta_revenue
        else:
            scores = []
            start = 0
            for sublabels in labels:
                seqlen = len(sublabels)
                scores.append(flatten_scores[start:start + seqlen].tolist())
                start = start + seqlen

            return labels, scores




    def get_batch_for_click(self, data_set, pos=-1):
        # print('data processing for click')
        query_num = data_set.user_num
        rand_idx = np.random.randint(query_num, size=self.batch_size)
        return self.prepare_data_for_click(data_set, rand_idx, pos)

    def get_batch_for_click_by_index(self, data_set, index, pos=-1):
        end_idx = min(data_set.user_num, index + self.batch_size)
        idx = np.array(range(index, end_idx))
        return self.prepare_data_for_click(data_set, idx, pos)

    def get_batch_for_ranker(self, data_set):
        # print('Begin data loading...')
        query_num = data_set.user_num
        rand_idx = np.random.randint(query_num, size=self.batch_size)
        return self.prepare_data_for_ranker(data_set, rand_idx)

    def get_batch_for_ranker_by_index(self, data_set, i):
        # print('Begin data loading...')
        idx = [i]
        return self.prepare_data_for_ranker(data_set, idx)

    def prepare_data_for_ranker(self, data_set, idx):
        labels = data_set.relevance_labels[idx].reshape(-1, self.rank_list_size)
        clicks = data_set.clicks[idx]
        input_feed = {}
        input_feed[self.item_features.name] = data_set.features[idx]
        input_feed[self.pos.name] = data_set.pos[idx]
        input_feed[self.target_clicks.name] = clicks.astype(float)
        input_feed[self.mask.name] = data_set.mask[idx].astype(float)
        input_feed[self.list_mask.name] = data_set.list_mask[idx].astype(float)
        input_feed[self.length.name] = data_set.len_list[idx]
        input_feed[self.pos_neg_ratio.name] = data_set.pos_neg_ratio
        return input_feed, labels, clicks

    def prepare_data_for_click(self, data_set, idx, pos):
        # Create input feed map
        input_feed = {}
        input_feed[self.item_features.name] = data_set.features[idx]
        input_feed[self.pos.name] = data_set.pos[idx] if pos == -1 else pos
        input_feed[self.target_clicks.name] = data_set.clicks[idx].astype(float)
        input_feed[self.mask.name] = data_set.mask[idx].astype(float)
        input_feed[self.pos_neg_ratio.name] = data_set.pos_neg_ratio
        return input_feed




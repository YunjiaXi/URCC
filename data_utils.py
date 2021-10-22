# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import json
import random
import os
import sys


class Raw_data:
    def __init__(self, data_path=None, set_name=None, rank_cut=100000, ranker_name='ranker',
                 data_dir_name='Yahoo', reverse_input=True, inp_type='default'):
        self.data_path = data_path
        self.set_name = set_name
        self.feature_path = data_path + 'filter/'
        settings = json.load(open(self.data_path + 'settings.json'))
        self.embed_size = settings['embed_size']
        self.eta = settings["eta"]
        self.rank_list_size = rank_cut if rank_cut < settings['rank_cutoff'] else settings['rank_cutoff']  # k_max

        ## documnet features
        self.orig_features = []  # d_max*emb_size
        feature_fin = open(self.feature_path + set_name + '.txt')
        for line in feature_fin:
            arr = line.strip().split(' ')
            self.orig_features.append([0.0 for _ in range(self.embed_size)])
            for x in arr[2:]:
                arr2 = x.split(':')
                self.orig_features[-1][int(arr2[0]) - 1] = float(arr2[1])
        feature_fin.close()
        self.orig_features.append([0 for _ in range(self.embed_size)])
        self.orig_item_num = len(self.orig_features)
        self.item_field_M = len(self.orig_features[-1])
        self.orig_features = np.array(self.orig_features)
        print('the total number of original documents is ' + str(self.orig_item_num), self.item_field_M)

        ##q_max
        self.qids = []
        # flatten_qids = []
        self.dids = []
        # flatten_dids = []
        # flatten_poss = []
        self.clicks = []
        # flatten_clicks = []
        self.len_list = []
        self.relevance_labels = []
        flatten_rls = []
        self.propensity = []
        # flatten_prop = []
        self.esti_prop = []
        self.bids = []
        self.pos = []
        self.flatten_bids = []
        self.features = []
        self.mask = []
        self.list_mask = []
        self.init_list= []
        self.initial_scores = []
        self.gold_list = []
        self.weight_list = []

        if data_path is None:
            print('error')
            return
        self.idx = []

        init_list_fin = open(self.data_path + ranker_name + '/' +set_name + '/' + set_name + '.init_list', 'r')
        init_list = init_list_fin.readlines()
        for i, line in enumerate(init_list):
            arr = line.strip().split(' ')
            self.qids.append(i)
            l = len(arr[1:][:self.rank_list_size])
            self.len_list.append(l)
            idx = np.arange(l)
            if inp_type == 'default':
                self.idx.append(idx)
            elif inp_type == 'reverse':
                self.idx.append(idx[::-1])
            else:
                np.random.shuffle(idx)
                self.idx.append(idx)
            self.dids.append(np.array([int(x) for x in arr[1:][:self.rank_list_size]])[self.idx[i]])
            if reverse_input:
                self.init_list.append(list(reversed(self.dids[-1])) + [-1] * (self.rank_list_size - len(self.dids[-1])))
            else:
                self.init_list.append([-1] * (self.rank_list_size - len(self.dids[-1])) + self.dids[-1])
            self.initial_scores.append((np.arange(l, 0, -1)/3).tolist() + [-10.0] * (self.rank_list_size - l))

            list_features = self.orig_features[self.dids[-1]]
            self.features.append(np.pad(list_features, ((0, self.rank_list_size-l), (0, 0)), 'constant'))
            self.mask.append(np.pad(np.ones(l), (0, self.rank_list_size-l), 'constant'))
            self.list_mask.append(np.pad(np.ones([l, l]), ((0, self.rank_list_size-l), (0, self.rank_list_size-l)), 'constant'))
        init_list_fin.close()

        bid_fin = open(self.data_path + ranker_name + '/' +set_name + '/' + set_name + '.bid', 'r')
        for i, line in enumerate(bid_fin):
            arr = line.strip().split(' ')[1:][:self.rank_list_size]
            self.bids.append(np.pad(np.array([float(x) for x in arr])[self.idx[i]], (0, self.rank_list_size-len(arr)), 'constant'))
            self.flatten_bids.extend(self.bids[-1])
        bid_fin.close()

        gold_weight_fin = open(self.data_path + ranker_name + '/' + set_name + '/' + set_name + '.weights', 'r')
        for i, line in enumerate(gold_weight_fin):
            arr = np.array([float(x) for x in line.strip().split(' ')[1:][:self.rank_list_size]])[self.idx[i]].tolist()
            self.relevance_labels.append(np.pad(np.array(arr), (0, self.rank_list_size-len(arr)), 'constant'))
            flatten_rls.extend(arr)
        gold_weight_fin.close()

        click_fin = open(self.data_path + ranker_name + '/' + set_name + '/' + set_name + '.click', 'r')
        for i, line in enumerate(click_fin):
            arr = [int(x) for x in line.strip().split(' ')[1:]]
            self.clicks.append(np.pad(np.array(arr)[self.idx[i]], (0, self.rank_list_size - len(arr)), 'constant'))

        click_fin.close()
        # print(ranker_name, 'clicks ', len(flatten_clicks))

        self.user_num = len(self.len_list)
        self.item_num = len(flatten_rls)

        self.pos = np.tile(np.arange(self.rank_list_size).reshape((1, self.rank_list_size)), (self.user_num, 1))
        self.features = np.array(self.features)
        self.mask = np.array(self.mask)
        self.list_mask = np.array(self.list_mask)
        self.bids = np.array(self.bids)
        self.relevance_labels = np.array(self.relevance_labels)
        self.clicks = np.array(self.clicks)
        self.len_list = np.array(self.len_list)


        print('the total number of quries is ' + str(self.user_num) + ' \nthe total number of items is ' + str(self.item_num))
        self.pos_neg_ratio = np.sum(flatten_rls)/(len(flatten_rls) - np.sum(flatten_rls))
        self.click_pos_vs_neg = np.sum(self.clicks)/(self.clicks.shape[0] * self.rank_list_size - np.sum(self.clicks))
        print('rel pos_neg_ratio is {}'.format(self.pos_neg_ratio))
        print("click pos vs neg is {}".format(self.click_pos_vs_neg))



def read_data(data_path, set_name, rank_cut=100000, ranker_name='ranker', data_dir_name='Yahoo', inp_type='default'):
    data = Raw_data(data_path, set_name, rank_cut, ranker_name, data_dir_name, inp_type=inp_type)
    return data



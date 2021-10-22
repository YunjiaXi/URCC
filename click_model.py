import  sys
import os
import math
import types
import numpy as np
import json
import matplotlib.pyplot as plt


class ClickModel:
    def __init__(self, data_path=None, ranker='ranker', set_name='train', eta=0.7, rank_cut=1000):
        self.data_path = data_path
        self.eta = eta
        self.rank_list_size = rank_cut
        self.qids = []
        self.clicks = []
        self.relevance = []
        self.examination = []
        self.query_context_feature = []
        self.doc_context_feature = []
        self.set_name = set_name
        self.feature_path = data_path + 'filter/'
        self.ranker = ranker
        if data_path is None:
            self.embed_size = 0
            return

        # load setting and context_feature setting
        settings = json.load(open(data_path + 'settings.json'))
        self.embed_size = settings['embed_size']
        self.rank_list_size = rank_cut if rank_cut < settings['rank_cutoff'] else settings['rank_cutoff']
        self.eta = settings["eta"]
        self.pos_decay = 1.0 / np.power(np.arange(1, self.rank_list_size+1), self.eta)

        features = []  # d_max*emb_size
        feature_fin = open(self.feature_path + set_name + '.txt', 'r')
        for line in feature_fin:
            arr = line.strip().split(' ')
            features.append([0.0 for _ in range(self.embed_size)])
            for x in arr[2:]:
                arr2 = x.split(':')
                features[-1][int(arr2[0]) - 1] = float(arr2[1])
        feature_fin.close()
        self.item_num = len(features)
        self.item_field_M = len(features[-1])
        self.features = np.array(features)
        print('the total number of documents is ' + str(self.item_num), self.item_field_M)

        self.dids = []
        for ranker_name in [self.ranker]:
            init_list_fin = open(self.data_path + ranker_name + '/' +set_name + '/' + set_name + '.init_list', 'r')
            init_list = init_list_fin.readlines()
            for i, line in enumerate(init_list):
                arr = line.strip().split(' ')
                self.qids.append(i)
                self.dids.append([int(x) for x in arr[1:][:self.rank_list_size]])
            init_list_fin.close()

            gold_weight_fin = open(self.data_path + ranker_name + '/' + set_name + '/' + set_name + '.weights', 'r')
            for line in gold_weight_fin:
                self.relevance.append([float(x) for x in line.strip().split(' ')[1:][:self.rank_list_size]])
            gold_weight_fin.close()
            print('click model rel num', len(self.relevance))
    def generate_clicks(self):
        for i in range(len(self.qids)):
            clicks, click_probs, exam_probs = self.sample_clicks(self.qids[i])
            # self.examination.append(exam_probs)
            self.clicks.append(clicks)
        self.save()


    def sample_clicks(self, query_id, doc_permutation=None):
        # permutation p[k] means item p[k](placed in p[k] at the original list) is placed pos k
        if doc_permutation is None:
            doc_permutation = np.arange(len(self.dids[query_id]))
        # print(query_id)
        # print(self.dids[query_id])
        doc_features = self.features[np.array(self.dids[query_id])]
        click_probs = []
        exam_probs = []
        clicks = []
        prev_click_item = -1
        rels = self.relevance[query_id]

        for k, cur_item in enumerate(doc_permutation):
            rel = rels[cur_item]
            exam_prob = self.pos_decay[k]
            if prev_click_item >= 0:
                similar_prob = self.calcul_similarity(doc_features[prev_click_item], doc_features[cur_item])
                exam_prob *= similar_prob
            exam_probs.append(exam_prob)
            click_prob = rel * exam_prob
            click_probs.append(click_prob)
            click = 1 if np.random.rand() < click_prob else 0
            if click:
                prev_click_item = cur_item
            clicks.append(click)
        return clicks, click_probs, exam_probs

    def calcul_similarity(self, feature1, feature2):
        sumData = np.sum(feature1*feature2)
        denom = np.linalg.norm(feature1) + np.linalg.norm(feature2)
        return 0.5 + 0.5 * (sumData / denom)

    def save(self):
        print('Saving ' + self.set_name + ' clicks & examinations...', end=' ')
        click_fout = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.click', 'w')
        # examination_fout = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.examination', 'w')
        nums = 0
        bid_fout = open(self.data_path + self.ranker + '/' + self.set_name + '/' + self.set_name + '.bid', 'w')
        for i in range(len(self.relevance)):
            click_line = str(self.qids[i]) + ' ' + ' '.join(str(t) for t in self.clicks[i]) + '\n'
            # exam_line = str(self.qids[i]) + ' ' + ' '.join(str(t) for t in self.examination[i]) + '\n'
            bids = (np.random.rand(len(self.relevance[i])) * 9 + 1).tolist()
            nums += len(bids)
            bid_line = str(self.qids[i]) + ' ' + ' '.join(str(t) for t in bids) + '\n'
            click_fout.write(click_line)
            # examination_fout.write(exam_line)
            bid_fout.write(bid_line)
        click_fout.close()
        # examination_fout.close()
        bid_fout.close()
        print('Done', nums)


def main():
    DATA_PATH = 'data/Yahoo/'
    rel_type = 'DLA'
    # ETA = 0.5 if len(sys.argv) < 4 else float(sys.argv[3])
    ETA = 1.5
    EPSILION = 0.1
    # model = ClickModel(data_path=DATA_PATH, eta=ETA)

    # model.sample_clicks(20)

    # generate all clicks
    for ranker_name in ['ranker']:
        for name in ['train', 'valid', 'test']:
            model = ClickModel(data_path=DATA_PATH, ranker=ranker_name, set_name=name, eta=ETA)
            model.generate_clicks()
            # model.save()


if __name__ == '__main__':
    main()

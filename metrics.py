import sys
import numpy as np
import json
import pickle


def get_rerank_List(ls):
    return sorted(range(len(ls)), key=lambda k: ls[k], reverse=True)


class ListMetrics:
    def __init__(self, pred_scores, rel_list, dataset):
        # self.rel_divide = dataset.rel_split
        self.rel_level = 2
        self.pred, self.gold = [], []
        self.query_num = len(pred_scores)
        self.click = []
        self._click, self._click_prob = [], []
        self.pos_decay = 1.0 / np.power(np.arange(1, dataset.rank_list_size + 1), dataset.eta)
        self._map = []
        self._ndcg = [[] for i in range(self.query_num)]
        self.labels = rel_list

        for i in range(self.query_num):
            scores = pred_scores[i]
            rels = rel_list[i]
            rerank_pred = get_rerank_List(scores)
            rerank_rel = get_rerank_List(rels)
            self.pred.append(rerank_pred)
            self.gold.append(rerank_rel)

            AP_value, AP_count = 0, 0
            for _i, _f, _g in zip(range(1, len(rels) + 1), rerank_pred, rerank_rel):
                if rels[_f] >= 1:
                    AP_count += 1
                    AP_value += AP_count / _i
            MAP = float(AP_value) / AP_count if AP_count != 0 else 0.
            self._map.append(MAP)

            doc_features = dataset.features[i]
            click_probs, clicks = [], []
            prev_click_item= -1
            for k, cur_item in enumerate(rerank_pred):
                rel = rels[cur_item]
                click_prob = rel
                if rel:
                    exam_prob = self.pos_decay[k]
                    if prev_click_item >= 0:
                        similar_prob = self.calcul_similarity(doc_features[prev_click_item], doc_features[cur_item])
                        exam_prob *= similar_prob
                    click_prob *= exam_prob
                click_probs.append(click_prob)
                click = 1 if np.random.rand() < click_prob else 0
                if click:
                    prev_click_item = cur_item
                clicks.append(click)
            self._click_prob.append(np.sum(click_probs))
            self._click.append(np.sum(clicks))

        self.did_num = self.query_num * dataset.rank_list_size
        self.click_per_query = sum(self._click) / self.query_num
        self.click_prob_per_doc = sum(self._click_prob) / self.did_num



    def calcul_similarity(self, feature1, feature2):
        sumData = np.sum(feature1*feature2)
        denom = np.linalg.norm(feature1) + np.linalg.norm(feature2)
        return 0.5 + 0.5 * (sumData / denom)

    def NDCG(self, k):
        res = 0
        for i in range(self.query_num):
            label = self.labels[i]
            pre_scope = self.pred[i][:k]
            rel_scope = self.gold[i][:k]
            iDCG, DCG = 0, 0
            for _i, _f, _g in zip(range(1, k + 1), pre_scope, rel_scope):
                DCG += (pow(2, label[_f]) - 1) / (np.log2(_i + 1))
                iDCG += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))
            NDCG = float(DCG) / iDCG if iDCG != 0 else 0.
            res += NDCG
            self._ndcg[i].append(NDCG)
        return res / self.query_num

    def EER(self, k):
        res = 0
        for i in range(self.query_num):
            n = min(len(self.pred_did[i]), k)
            Ri = [(2 ** x - 1) / 2 ** self.rel_level for x in self.pred_rel[i]][:n]
            RR = []
            for j in range(n):
                RRi = 1
                for t in range(j):
                    RRi *= (1 - Ri[t])
                RRi *= Ri[j]
                RR.append(RRi)
            for j in range(n):
                res += RR[j] / (j + 1)
        return res / self.query_num



    def save(self, path):
        data = {'MAP': self._map,
                'NDCG': self._ndcg,
                'click_per_query': self._click,
                'click_prob': self._click_prob}
        output = open(path, 'wb')
        pickle.dump(data, output)
        output.close()


def compute_metrics(labels, scores, dataset, save_file=None, log=None):
    metrics = ListMetrics(scores, labels, dataset)
    Map = sum(metrics._map) / metrics.query_num
    print('MAP: ', Map)
    if log:
        print("MAP %.6f" % (Map), file=log, flush=True)
    for i in [1, 3, 5, 10]:
        NDCG = metrics.NDCG(i)
        # ERR = metrics.EER(i)
        print('NDCG@' + str(i) + ': ', NDCG)
        # print('EER@' + str(i) + ': ', ERR)
        if log:
            print("NDCG@ %d %.6f  " % (i, NDCG,), file=log, flush=True)
    print()

    print('click_per_query: ', metrics.click_per_query, '\nclick_prob_per_doc: ', metrics.click_prob_per_doc)
    if log:
        print("click_per_query %.6f    click_per_doc %.6f \n\n" % (metrics.click_per_query,  metrics.click_prob_per_doc), file=log, flush=True)
    print()
    if save_file is not None:
        metrics.save(save_file)
        print('save successfully to {}!'.format(save_file))
    return metrics.click_prob_per_doc


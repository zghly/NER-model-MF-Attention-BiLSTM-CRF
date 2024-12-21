# -*- coding:utf-8 -*-
'''
@Author: zgh
@Date: 2024-12-23
'''

import copy
import pickle
import sys
import yaml
import torch
import torch.optim as optim
from data_manager import DataManager
from model import BiLSTMCRF
import numpy as np
from sklearn.metrics import classification_report
from utils import f1_score, get_tags, format_result
from cnradical import Radical, RunOption
from pypinyin import lazy_pinyin, Style
import datetime


class ChineseNER(object):

    def __init__(self, entry="train", loc='params.pkl'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备
        self.load_config()
        self.__init_model(entry, loc)

    def __init_model(self, entry, loc):
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                # character dictionary
                "vocab": self.train_manager.vocab,
                # radical dictionary
                "bushouzidian": self.train_manager.bushouzidian,
                # pinyin dictionary
                "pinyinzidian": self.train_manager.pinyinzidian,
                # label dictionary
                "tag_map": self.train_manager.tag_map,
            }
            self.save_params(data)
            dev_manager = DataManager(batch_size=5, data_type="dev")
            self.dev_batch = dev_manager.iteration()

            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            ).to(self.device)

        elif entry == "predict":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")
            self.bushouzidian = data_map.get("bushouzidian")
            self.pinyinzidian = data_map.get("pinyinzidian")

            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            ).to(self.device)
            self.restore_model(entry, loc)

    def load_config(self):
        self.embedding_size = 100
        self.hidden_size = 128
        self.batch_size = 12
        # Network Model File Path
        self.model_path = 'models/'
        #All named entities label
        self.tags = ["BN", "DL", "DIS", "NDIS", "EI", "SC", "RA", "ST", "MA"]
        # dropout概率
        self.dropout = 0.3


    def restore_model(self, entry, loc):
        if entry == 'train':
            mod = self.model_path + "params.pkl"
            try:
                self.model.load_state_dict(torch.load(mod))
                print("model restore success!")
            except Exception as error:
                print("model restore faild! {}".format(error))
        elif entry == 'predict':
            mod = self.model_path + loc
            try:
                self.model.load_state_dict(torch.load(mod))
                print("model restore success!")
            except Exception as error:
                print("model restore faild! {}".format(error))

    def save_params(self, data):
        with open("models/data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open("models/data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        # Define the initial learning rate
        initial_learning_rate = 0.001
        # Define the hyperparameters for the Adam optimizer
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        optimizer = optim.Adam(self.model.parameters(), lr=initial_learning_rate, betas=(beta1, beta2), eps=epsilon)
        results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        for epoch in range(10):
            index = 0
            baocun_acc = []
            baocun_loss = []
            for batch in self.train_manager.get_batch():
                index += 1
                self.model.zero_grad()
                sentences, bushou_vec, pinyin_vec, weizhi_vec, trans_weizhi_vec, tags, length = zip(*batch)
                # Conversion to tensor
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(self.device)
                bushou_vec_tensor = torch.tensor(bushou_vec, dtype=torch.long).to(self.device)
                pinyin_vec_tensor = torch.tensor(pinyin_vec, dtype=torch.long).to(self.device)
                trans_weizhi_vec_tensor = torch.tensor(trans_weizhi_vec, dtype=torch.long).to(self.device)
                weizhi_vec_tensor = torch.tensor(weizhi_vec, dtype=torch.long).to(self.device)
                tags_tensor = torch.tensor(tags, dtype=torch.long).to(self.device)
                length_tensor = torch.tensor(length, dtype=torch.long).to(self.device)

                loss = self.model.neg_log_likelihood(sentences_tensor, bushou_vec_tensor, pinyin_vec_tensor,
                                                     weizhi_vec_tensor,
                                                     trans_weizhi_vec_tensor, tags_tensor, length_tensor)
                progress = ("█" * int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                    epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                )
                )
                pinjun_acc = self.evaluate()
                baocun_acc.append(pinjun_acc)
                baocun_loss.append(loss.cpu().tolist()[0])

                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), self.model_path + 'params.pkl')

            epoach_acc = sum(baocun_acc) / len(baocun_acc)
            epoach_loss = sum(baocun_loss) / len(baocun_loss)
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write("第{}个epoach训练集损失".format(epoch) + str(epoach_loss) + '\r')
                f.write("Validation dataset correctness (验证集正确率):" + str(epoach_acc) + '\r')
                f.write(' ' + '\r')


    def evaluate(self):
        sentences, bushou_vec, pinyin_vec, weizhi_vec, trans_weizhi_vec, labels, length = zip(
            *self.dev_batch.__next__())
        sen_len = []
        for i in range(len(sentences)):
            try:
                sen_len.append(sentences[i].index(1))
            except:
                sen_len.append(360)
        sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(self.device)
        bushou_vec_tensor = torch.tensor(bushou_vec, dtype=torch.long).to(self.device)
        pinyin_vec_tensor = torch.tensor(pinyin_vec, dtype=torch.long).to(self.device)
        weizhi_vec_tensor = torch.tensor(weizhi_vec, dtype=torch.long).to(self.device)
        trans_weizhi_vec_tensor = torch.tensor(trans_weizhi_vec, dtype=torch.long).to(self.device)

        _, paths = self.model(sentences_tensor, bushou_vec_tensor, pinyin_vec_tensor, weizhi_vec_tensor,
                              trans_weizhi_vec_tensor)

        print("\teval")
        labels = list(labels)
        paths = list(paths)
        for j in range(len(sen_len)):
            labels[j] = labels[j][:sen_len[j]]
            paths[j] = paths[j][:sen_len[j]]
        acc_pre = []
        for i in range(len(labels)):
            acc = 0
            for j in range(len(labels[i])):
                if labels[i][j] == paths[i][j]:
                    acc = acc + 1
            acc_pre.append((acc / len(labels[i])) * 100)
        print(acc_pre, sum(acc_pre) / len(acc_pre))
        return sum(acc_pre) / len(acc_pre)


    def getPositionEncoding(self, tra_seq, tra_dim, n=10000):
        PE = np.zeros(shape=(tra_seq, tra_dim))
        for pos in range(tra_seq):
            for i in range(int(tra_dim / 2)):
                denominator = np.power(n, 2 * i / tra_dim)
                PE[pos, 2 * i] = np.sin(pos / denominator)
                PE[pos, 2 * i + 1] = np.cos(pos / denominator)

        return PE

    def predict(self, input_str=""):
        if not input_str:
            input_str = input("请输入文本: ")
            # 根据字典将输入文本转换为向量
        all_line = list(input_str)
        bushou = []
        pinyin = []
        weizhi = []
        zimubiao = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12,
                    "m": 13,
                    "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24,
                    "y": 25, "z": 26, }

        for word in all_line:
            radical = Radical(RunOption.Radical)
            if radical.trans_ch(word) != None:
                radical_out = radical.trans_ch(word)
                style = Style.TONE3
                pinyin_1 = list(lazy_pinyin(word, style=style)[0])
                pinyinbianma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                try:
                    pinyinbianma.append(int(pinyin_1[-1]))
                    bianma = [zimubiao[i] for i in pinyin_1[:-1]]
                    for i in bianma:
                        pinyinbianma[int(i) - 1] = 1
                except Exception:
                    pinyinbianma.append(5)
                    bianma = [zimubiao[i] for i in pinyin_1]
                    for i in bianma:
                        pinyinbianma[int(i) - 1] = 1
            elif word in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                radical_out = 'num'
                pinyinbianma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif word in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                          'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
                radical_out = 'zimu'
                pinyinbianma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                radical_out = 'fuhao'
                pinyinbianma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            pinyin.append(pinyinbianma)
            bushou.append(self.bushouzidian.get(radical_out, 0))

        fanxiang = []
        for ixc in range(len(all_line)):
            if all_line[ixc][0] == "，" or all_line[ixc][0] == "；" or all_line[ixc][0] == "。":
                fanxiang.append(ixc)
        all_line1 = []
        for ixc in range(len(fanxiang)):
            if ixc == 0:
                all_line1.append(all_line[:fanxiang[ixc]])
            elif ixc == len(fanxiang):
                all_line1.append(all_line[fanxiang[ixc]:])
            else:
                all_line1.append(all_line[fanxiang[ixc - 1] + 1:fanxiang[ixc]])

        for izc in range(len(all_line1)):
            for j in range(len(all_line1[izc])):
                weizhi.append([j + 1, len(all_line1[izc]) - j])
            weizhi.append([0, 0])

        input_vec = [self.vocab.get(i, 0) for i in input_str]
        sentences = torch.tensor(input_vec).view(1, -1).to(self.device)
        bushou_vec = torch.tensor(bushou).view(1, -1).to(self.device)
        pinyin_vec = torch.tensor(pinyin).view(1, -1, 27).to(self.device)
        weizhi_vec = torch.tensor(weizhi).view(1, -1, 2).to(self.device)
        tra_weizhi = torch.tensor(self.getPositionEncoding(len(input_vec), 27, n=10000)).to(self.device)
        tra_weizhi_vec = torch.tensor(tra_weizhi).view(1, -1, 27).to(self.device)
        sorces, paths = self.model(sentences, bushou_vec, pinyin_vec, weizhi_vec, tra_weizhi_vec)
        res = {v: k for k, v in self.vocab.items()}
        fin_sen = [res[i] for i in input_vec]
        res2 = {v: k for k, v in self.tag_map.items()}
        fin_pre = [res2[i] for i in paths[0]]
        return fin_sen, fin_pre


def pre_find_fun(org_data, pre_list, ner_name):
    ner_len = len(ner_name)
    loc_id = []
    for i in range(len(pre_list)):
        if len(pre_list[i]) > 2:
            if pre_list[i][-ner_len:] == ner_name:
                loc_id.append(i)

    pre_result = []
    if len(loc_id) != 0:
        jieduan = []
        for i in range(len(loc_id)):
            if i > 0:
                if loc_id[i] - 1 != loc_id[i - 1]:
                    jieduan.append(i)
        if len(jieduan) == 0:
            shibie = [loc_id]
        if len(jieduan) == 1:
            shibie = [loc_id[:jieduan[0]], loc_id[jieduan[0]:]]
        if len(jieduan) == 2:
            shibie = [loc_id[:jieduan[0]], loc_id[jieduan[0]:jieduan[1]], loc_id[jieduan[1]:]]
        if len(jieduan) > 2:
            shibie = []
            for ll in range(len(jieduan)):
                if ll == 0:
                    ss = loc_id[:jieduan[ll]]
                    shibie.append(ss)
                elif ll == len(jieduan) - 1:
                    ss1 = loc_id[jieduan[ll - 1]:jieduan[ll]]
                    shibie.append(ss1)
                    ss = loc_id[jieduan[ll]:]
                    shibie.append(ss)
                else:
                    ss = loc_id[jieduan[ll - 1]:jieduan[ll]]
                    shibie.append(ss)

        for pp in range(len(shibie)):
            if len(shibie[pp]) == 1:
                shibie[pp] = shibie[pp]
            else:
                shibie[pp] = [shibie[pp][0], shibie[pp][-1] + 1]

        pre_result = []
        for ww in range(len(shibie)):
            if len(shibie[ww]) == 1:
                www = org_data[shibie[ww][0]]
                pre_result.append(www)
            else:
                www = org_data[shibie[ww][0]:shibie[ww][1]]
                pre_result.append(www)
    else:
        shibie = []
    if ner_name == '-NDIS':
        return pre_result, shibie
    else:
        return pre_result


if __name__ == "__main__":

    cn = ChineseNER("train")
    cn.train()
    # pre='A桥桥面铺装层良好；伸缩缝存在被砂石堵塞、防水橡胶条破损渗水等病害；护栏混凝土底座局部开裂。桥面系技术状况评定得分为92.75分。A桥上部结构技术状况良好，未发现混凝土破损、剥落、露筋等病害。技术状况评定得分为100.00分。
    # 桥台有渗水痕迹；部分支座有不均匀压缩变形。下部结构技术状况得分为97.49分。A桥技术状况得分为97.78分，根据桥梁完好状况评估标准，A桥为A级。'
    # cn = ChineseNER("predict")
    # org_sentences, pre_result = cn.predict(input_str=pre)
    # bn = pre_find_fun(pre, pre_result, '-BN')
    # dl = pre_find_fun(pre, pre_result, '-DL')
    # dis = pre_find_fun(pre, pre_result, '-DIS')
    # ndis,loc_ndis = pre_find_fun(pre, pre_result, '-NDIS')
    # ei = pre_find_fun(pre, pre_result, '-EI')
    # sc = pre_find_fun(pre, pre_result, '-SC')
    # ra = pre_find_fun(pre, pre_result, '-RA')
    # st = pre_find_fun(pre, pre_result, '-ST')
    # ma = pre_find_fun(pre, pre_result, '-MA')
    # print('bridge name entity (桥梁名称):', bn)
    # print('defect location entity (病害位置):', dl)
    # print('defect entity (病害):', dis)
    # print('no defect entity (无病害):', ndis)
    # print('defect indicator entity (评价指标):', ei)
    # print('score entity (技术状况得分):', sc)
    # print('rank entity (技术状况等级):', ra)
    # print('standardise entity (规范):', st)
    # print('Maintenance entity (维修措施):', ma)

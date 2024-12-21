# -*- coding:utf-8 -*-
'''
@Author: zgh
@Date: 2024-12-23
'''

import copy
import pickle as cPickle
from cnradical import Radical, RunOption
from pypinyin import lazy_pinyin,  Style
import torch
import numpy as np

class DataManager():
    def __init__(self, max_length=220, batch_size=20, data_type='train', tags=[]):
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.max_length = 360
        self.data_type = data_type
        self.data = []
        self.batch_data = []
        #Constructing the word dictionary
        self.vocab = {"unk": 0,"buchong": 1}
        #Constructing the radical dictionary
        self.bushouzidian={'num':0,"buchong":1,'zimu':2,'fuhao':3}
        #Constructing the pinyin dictionary
        self.zimubiao={"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,
             "n":14,"o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,"v":22,"w":23,"x":24,"y":25,"z":26,}
        self.pinyinzidian={"unk": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           "buchong": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        # Constructing the tag dictionary
        self.tag_map = {"O":0, "START":1, "STOP":2, 'dem':3}

        if data_type == "train":
            assert tags, Exception("请指定需要训练的tag类型，如[\"ORG\", \"PER\"]")
            self.generate_tags(tags)
            #Training set file path
            self.data_path = "data/substructure/train_data.txt"
        elif data_type == "dev":
            self.data_path = "data/substructure/val_data.txt"
            self.load_data_map()
        elif data_type == "test":
            self.data_path = "data/test"
            self.load_data_map()

        self.load_data()
        self.prepare_batch()

    def generate_tags(self, tags):
        #Using the BIOE labelling method
        self.tags = []
        for tag in tags:
            for prefix in ["B-", "I-"]:
                self.tags.append(prefix + tag)
        self.tags.append("O")

    def load_data_map(self):
        with open("models/data.pkl", "rb") as f:
            self.data_map = cPickle.load(f)
            self.vocab = self.data_map.get("vocab", {})
            self.bushouzidian = self.data_map.get("bushouzidian", {})
            self.pinyinzidian = self.data_map.get("pinyinzidian", {})
            self.tag_map = self.data_map.get("tag_map", {})
            self.tags = self.data_map.keys()

    #positional encoding in transformer
    def getPositionEncoding(self,tra_seq, tra_dim, n=10000):
        PE = np.zeros(shape=(tra_seq, tra_dim))
        for pos in range(tra_seq):
            for i in range(int(tra_dim / 2)):
                denominator = np.power(n, 2 * i / tra_dim)
                PE[pos, 2 * i] = np.sin(pos / denominator)
                PE[pos, 2 * i + 1] = np.cos(pos / denominator)
        return PE

    def load_data(self):
        # load data
        # covert to one-hot
        sentence = []
        target = []
        bushouxiangliang=[]
        pinyinxiangliang=[]
        weizhixiangliang=[]
        all_line=[]
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                line = line[:-1]
                #Sentence ends with "end"
                if line == "end":
                    fanxiang = []
                    for ixc in range(len(all_line)):
                        if all_line[ixc][0] == "，" or all_line[ixc][0] =="；" or all_line[ixc][0] =="。":
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
                            weizhixiangliang.append([j+1, len(all_line1[izc]) - j])
                        weizhixiangliang.append([0, 0])

                    weizhi_trans = self.getPositionEncoding(len(sentence), 27, n=10000)
                    self.data.append([sentence,bushouxiangliang, pinyinxiangliang,weizhixiangliang, weizhi_trans,target])

                    all_line=[]
                    sentence = []
                    target = []
                    bushouxiangliang = []
                    pinyinxiangliang = []
                    weizhixiangliang = []
                    all_trans_weizhi = []
                    continue

                all_line.append(line)
                try:
                    word, tag = line.split(" ")
                except Exception:
                    continue
                #radical vector construction
                radical = Radical(RunOption.Radical)  # 获取偏旁
                if radical.trans_ch(word) != None:
                    radical_out=radical.trans_ch(word)
                    style = Style.TONE3
                    pinyin = list(lazy_pinyin(word, style=style)[0])
                    pinyinbianma = [0 for i in range(26)]
                    try:
                        pinyinbianma.append(int(pinyin[-1]))
                        bianma = [self.zimubiao[i] for i in pinyin[:-1]]
                        for i in bianma:
                            pinyinbianma[int(i) - 1] = 1
                    except Exception:
                        pinyinbianma.append(5)
                        bianma = [self.zimubiao[i] for i in pinyin]
                        for i in bianma:
                            pinyinbianma[int(i) - 1] = 1
                elif word in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    radical_out='num'
                    pinyinbianma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif word in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                                 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
                    radical_out='zimu'
                    pinyinbianma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                else:
                    radical_out='fuhao'
                    pinyinbianma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                #pinyin vector
                if word not in self.pinyinzidian and self.data_type == "train":
                    self.pinyinzidian[word] = pinyinbianma
                if radical_out not in self.bushouzidian and self.data_type == "train":
                    self.bushouzidian[radical_out] = max(self.bushouzidian.values()) + 1

                #character vector
                if word not in self.vocab and self.data_type == "train":
                    self.vocab[word] = max(self.vocab.values()) + 1 

                #label tag
                if tag not in self.tag_map and self.data_type == "train" and tag in self.tags:
                    self.tag_map[tag] = len(self.tag_map.keys())

                sentence.append(self.vocab.get(word, 0))
                bushouxiangliang.append((self.bushouzidian.get(radical_out,0)))
                target.append(self.tag_map.get(tag, 0))
                pinyinxiangliang.append(pinyinbianma)

        self.input_size = len(self.vocab.values())
        print("Number of sentences in the training set (训练集句子个数){} data: {}".format(self.data_type ,len(self.data)))
        print("Number of characters in the data set (数据集字符个数)vocab size: {}".format(self.input_size))
        print("Label Dictionary Size (标签字典大小)，unique tag: {}".format(len(self.tag_map.values())))
        print("-"*50)
    
    def convert_tag(self, data):
        # add E-XXX for tags
        # add O-XXX for tags
        _, tags = data
        converted_tags = []
        for _, tag in enumerate(tags[:-1]):
            if tag not in self.tag_map and self.data_type == "train":
                self.tag_map[tag] = len(self.tag_map.keys())
            converted_tags.append(self.tag_map.get(tag, 0))
        converted_tags.append(0)
        data[1] = converted_tags
        assert len(converted_tags) == len(tags), "convert error, the list dosen't match!"
        return data

    def prepare_batch(self):
        '''
        Process data according to batch_size
        '''
        index = 0
        while True:
            if index+self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)
    
    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = self.max_length
        #Perform a zero-completion operation on a sentence whose length is less than max_length.
        for i in c_data:
            i.append(len(i[0]))
            if len(i[0])<=max_length:
                i[4] = i[4].tolist()
                i[0] = i[0] + (max_length-len(i[0])) * [1]
                #radical vector
                i[1] = i[1] + (max_length-len(i[1])) * [1]
                #pinyin vector
                for ii in range(max_length-len(i[2])):
                    i[2].append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                #relative position vector
                for ii in range(max_length-len(i[3])):
                    i[3].append([0,0])
                for ii in range(max_length - len(i[4])):
                    i[4].append([0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                i[5] = i[5] + (max_length-len(i[5])) * [0]

            else:
                i[4] = i[4].tolist()
                i[0] = i[0][:max_length]
                i[1] = i[1][:max_length]
                i[2] = i[2][:max_length]
                i[3] = i[3][:max_length]
                i[4] = i[4][:max_length]
                i[5] = i[5][:max_length]
        return c_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data

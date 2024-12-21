# -*- coding:utf-8 -*-
'''
@Author: zgh
@Date: 2024-12-23
'''
import copy
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn

START_TAG = "START"
STOP_TAG = "STOP"

def log_sum_exp(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value,  dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # if mask is not None:
    #     scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class BiLSTMCRF(nn.Module):

    def __init__(
            self,
            # vocab字符标签字典，包括实体开头中间结尾，句子开始，句子结尾，非实体，一个类别对应一个数字
            tag_map={},
            #batch大小
            batch_size=20,
            #vocab训练集字符字典，一个字符对应一个数字，vocab_size代表字典大小
            vocab_size=20,
            #隐层维度
            hidden_dim=128,
            #dropout概率
            dropout=0.3,
            #词向量维度
            embedding_dim=100
        ):
        super(BiLSTMCRF, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        #标签类别，即网络最终输出维度
        self.tag_size = len(tag_map)
        self.tag_map = tag_map
        #torch.randn(self.tag_size, self.tag_size)生成大小为(self.tag_size, self.tag_size)的张量
        #torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。
        # 即在定义网络时这个tensor就是一个可以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        #将随机tensor self.transitions的self.tag_map[START_TAG]行，self.tag_map[STOP_TAG]列赋值
        self.transitions.data[:, self.tag_map[START_TAG]] = -1000.
        self.transitions.data[self.tag_map[STOP_TAG], :] = -1000.
        #根据vocab_size大小形成字符的独热编码后输入Embedding层得到embedding_dim维度的词向量
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.word_embeddings_2 = nn.Embedding(vocab_size, self.embedding_dim)
        #定义BILSTM层：输入维度self.embedding_dim, 输出维度self.hidden_dim // 2,
        #网络层数num_layers=1, 是否使用双向LSTM，bidirectional=True, 网络输出格式batch_first=True, dropout概率，dropout=self.dropout
        # self.bert = BertModel.from_pretrained('./bert-base-chinese')
        self.lstm = nn.LSTM(256, self.hidden_dim // 2,
                        num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        #生成两个大小为(2, self.batch_size, self.hidden_dim // 2)的随机张量
        self.hidden = self.init_hidden()

        d_model=256
        self.h = 4
        assert d_model % self.h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // self.h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout_multi=0.2
        self.dropout_multi_attition = nn.Dropout(p=self.dropout_multi)

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2).to('cuda:0'),
                torch.randn(2, self.batch_size, self.hidden_dim // 2).to('cuda:0'))

    def __get_lstm_features(self, sentence,bushou,pinyin,weizhi,trans_weizhi):
        #输入句子，输出BILSTM+线性层结果
        self.hidden = self.init_hidden()
        #句子长度
        length = sentence.shape[1]
        #重新定义矩阵形状
        embeddings = self.word_embeddings(sentence).view(self.batch_size, length, self.embedding_dim)
        embeddings_2 = self.word_embeddings_2(bushou).view(self.batch_size, length, self.embedding_dim)
        fin_embeddings = torch.cat((embeddings, embeddings_2, pinyin, trans_weizhi, weizhi), dim=2)

        #多头注意力机制
        query=fin_embeddings
        key=fin_embeddings
        value=fin_embeddings
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout_multi_attition)
        # 3) "Concat" using x view and apply x final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        sub_embeddings = self.linears[-1](x)

        lstm_out, self.hidden = self.lstm(sub_embeddings, self.hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        logits = self.hidden2tag(lstm_out)
        return logits
    
    def real_path_score_(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i], tags[i+1]] + feat[tags[i + 1]]
        score = score + self.transitions[tags[-1], self.tag_map[STOP_TAG]]
        return score

    def real_path_score(self, logits, label):
        '''
        caculate real path score  
        :params logits -> [len_sent * tag_size]，BILSTM+线性层结果
        :params label  -> [1 * len_sent]，实际标签

        Score = Emission_Score + Transition_Score  
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])  
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])  
        '''
        score = torch.zeros(1).to('cuda:0')
        label = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long).to('cuda:0'), label])
        for index, logit in enumerate(logits):
            emission_score = logit[label[index + 1]]
            transition_score = self.transitions[label[index], label[index + 1]]
            score += emission_score + transition_score
        score += self.transitions[label[-1], self.tag_map[STOP_TAG]]
        return score

    def total_score(self, logits, label):
        """
        caculate total score
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]
        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        obs = []
        previous = torch.full((1, self.tag_size), 0).to('cuda:0')
        for index in range(len(logits)): 
            previous = previous.expand(self.tag_size, self.tag_size).t()
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp(scores)
        previous = previous + self.transitions[:, self.tag_map[STOP_TAG]]
        # caculate total_scores
        total_scores = log_sum_exp(previous.t())[0]
        return total_scores

    def neg_log_likelihood(self, sentences, bushou,pinyin,weizhi, trans_weizhi, tags, length):
        self.batch_size = sentences.size(0)
        #得到BILSTM+线性层输出
        logits = self.__get_lstm_features(sentences,bushou,pinyin,weizhi,trans_weizhi)
        real_path_score = torch.zeros(1).to('cuda:0')
        total_score = torch.zeros(1).to('cuda:0')
        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng]
            tag = tag[:leng]
            real_path_score += self.real_path_score(logit, tag)
            total_score += self.total_score(logit, tag)
        return total_score - real_path_score

    def forward(self, sentences,bushou,pinyin,weizhi,trans_weizhi, lengths=None):
        """
        params sentences to predict
        params lengths represent the ture length of sentence, the default is sentences.size(-1)
        """
        sentences = torch.tensor(sentences, dtype=torch.long)
        bushou = torch.tensor(bushou, dtype=torch.long)
        pinyin = torch.tensor(pinyin, dtype=torch.long)
        weizhi = torch.tensor(weizhi, dtype=torch.long)
        trans_weizhi = torch.tensor(trans_weizhi, dtype=torch.long)
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)
        logits = self.__get_lstm_features(sentences,bushou,pinyin,weizhi,trans_weizhi)
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]
            score, path = self.__viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths
    
    def __viterbi_decode(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size()).to('cuda:0')
        backpointers = torch.zeros(logits.size(), dtype=torch.long).to('cuda:0')
        
        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.cpu().numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi

    def __viterbi_decode_v1(self, logits):
        init_prob = 1.0
        trans_prob = self.transitions.t()
        prev_prob = init_prob
        path = []
        for index, logit in enumerate(logits):
            if index == 0:
                obs_prob = logit * prev_prob
                prev_prob = obs_prob
                prev_score, max_path = torch.max(prev_prob, -1)
                path.append(max_path.cpu().tolist())
                continue
            obs_prob = (prev_prob * trans_prob).t() * logit
            max_prob, _ = torch.max(obs_prob, 1)
            _, final_max_index = torch.max(max_prob, -1)
            prev_prob = obs_prob[final_max_index]
            prev_score, max_path = torch.max(prev_prob, -1)
            path.append(max_path.cpu().tolist())
        return prev_score.cpu().tolist(), path

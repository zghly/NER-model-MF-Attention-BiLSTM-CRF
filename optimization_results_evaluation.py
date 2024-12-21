# -*- coding:utf-8 -*-
'''
@Author: zgh
@Date: 2024-12-23
'''

from syntactic_analysis import syntactic_completion
import re
from seqeval.metrics import f1_score

def biaoqiantihuan(shiti, juzi, juzi_biaoqian, biaoqian):
    for m in re.finditer(shiti, juzi):
        len_lable = abs(m.start() - m.end())
        if juzi_biaoqian[m.start():m.end()] == list("O" * len_lable):
            juzi_biaoqian[m.start():m.end()] = biaoqian
        else:
            changdu = 0
            for yuansu in juzi_biaoqian[m.start():m.end()]:
                if yuansu != "O":
                    changdu = changdu + 1
            if changdu < len_lable:
                    juzi_biaoqian[m.start():m.end()] = biaoqian
    return juzi_biaoqian

def sebtebce_label(org, bn, dl, dis, ndis, ei, sc, ra, st, ma):
    org_sen = org[0]
    org_lab = org[1]
    org_pre = ['O' for i in range(len(org_sen))]
    # Annotation of sentences based on entity recognition results
    for i in bn:
        bq='BN'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in dl:
        bq='DL'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in dis:
        bq='DIS'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in ndis:
        bq='NDIS'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in ei:
        bq='EI'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in sc:
        bq='SC'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in ra:
        bq='RA'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in st:
        bq='ST'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)
    for i in ma:
        bq='MA'
        biaoqian = ['I-'+bq] * len(i)
        biaoqian[0] = 'B-'+bq
        org_pre = biaoqiantihuan(i, org_sen, org_pre, biaoqian)

    return [org_sen, org_lab, org_pre]

val_file = "data/substructure/test_data.txt"

with open(val_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

#Get the list of all sentences in the test set and the corresponding tag list
#all_val_sentence = [[Sentence, tag list], [], [], []...]
all_val_sentence = []
sentence = ''
label = []
# 打印出列表中的每一行
for line in lines:
    xx=line.strip()
    xx=xx.split(' ')
    if len(xx)==2:
        sentence=sentence+xx[0]
        label.append(xx[1])
    elif len(xx)==1:
        all_val_sentence.append([sentence,label])
        sentence = ''
        label = []


all_youhua=[]
all_sen=''
entity_lable=[]
entity_identification_results=[]
entity_identification_results_after_correction=[]

for i in range(len(all_val_sentence)):
    youhua, youhuaqian, youhuahou, fin_youhuahou, dis_clf_ner = syntactic_completion(all_val_sentence[i][0], 'params.pkl')
    print('Entity identification results before correction',youhuaqian)
    print('Entity identification results after correction',youhuahou)

    ss1 = sebtebce_label(all_val_sentence[i], youhuaqian[0],youhuaqian[1], youhuaqian[2],youhuaqian[3],
                         youhuaqian[4],youhuaqian[5], youhuaqian[6],youhuaqian[7], youhuaqian[8])
    ss3 = sebtebce_label(all_val_sentence[i], fin_youhuahou[0], fin_youhuahou[1], fin_youhuahou[2], fin_youhuahou[3], fin_youhuahou[4],
                     fin_youhuahou[5], fin_youhuahou[6], fin_youhuahou[7], fin_youhuahou[8])

    all_youhua.append(youhua)
    all_sen=all_sen+ss1[0]
    entity_lable=entity_lable+ss1[1]
    entity_identification_results=entity_identification_results+ss1[2]
    entity_identification_results_after_correction=entity_identification_results_after_correction+ss3[2]

print('F1 score before correction',round(f1_score([entity_lable], [entity_identification_results]),3))
print('F1 score after correction',round(f1_score([entity_lable], [entity_identification_results_after_correction]),3))

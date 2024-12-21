# -*- coding:utf-8 -*-
'''
@Author: zgh
@Date: 2024-12-23
'''

import copy
import numpy as np
import random
import math
# from spacy.tokens import Doc, DocBin
'''
NER Data Enhancement, contains:
1. similar entities are exchanged with each other
2. Comma position breaks sentences, followed by random exchange some phrase position
3. Comma position breaks sentences, followed by random delete some phrases
4. Comma position breaks sentences, followed by random insert some phrases
'''

all_txt=[]
with open('data/substructure/lable_text.txt', 'r', encoding='utf-8') as f:
    for ann in f.readlines():
        ann = ann.strip('\n')
        ann = ann.replace(' ', '')
        all_txt.append(ann)

# Constructing an entity dictionary based on training set annotated data and prior knowledge
text_data=[]
bridge_name=[]
defect_location=[]
defect=[]
no_defect=[]
defect_indicator=[]
score=[]
rank=[]
standardise=[]
maintenance=[]
data_len=10

for i in range(len(all_txt)):
    #text data
    if (i+1)%data_len==1:
        text_data.append(all_txt[i])
    #bridge name entity
    elif (i+1)%data_len==2:
        xx2=all_txt[i][5:].split('，')
        for i in xx2:
            if i not in bridge_name and i!='xx':
                bridge_name.append(i)
    #defect location entity
    elif (i+1)%data_len==3:
        if len(all_txt[i]) > 7:
            xx4=list(all_txt[i][7:].split('，'))
            for i in xx4:
                if i not in defect_location and i!='xx':
                    defect_location.append(i)
    #defect entity
    elif (i + 1) % data_len == 4:
        if len(all_txt[i]) > 5:
            xx5 = list(all_txt[i][5:].split('，'))
            for i in xx5:
                if i not in defect and i!='xx':
                    defect.append(i)
    #no defect entity
    elif (i + 1) % data_len == 5:
        if len(all_txt[i]) > 6:
            xx6 = list(all_txt[i][6:].split('，'))
            for i in xx6:
                if i not in no_defect and i!='xx':
                    no_defect.append(i)
    #defect indicator entity
    elif (i + 1) % data_len == 6:
        if len(all_txt[i]) > 7:
            xx7 = list(all_txt[i][7:].split('，'))
            for i in xx7:
                if i not in defect_indicator and i!='xx':
                    defect_indicator.append(i)
    #score entity
    elif (i + 1) % data_len == 7:
        if len(all_txt[i])>9:
            xx8 = list(all_txt[i][9:].split('，'))
            for i in xx8:
                if i not in score and i!='xx':
                    score.append(i)
    #rank entity
    elif (i + 1) % data_len == 8:
        if len(all_txt[i])>9:
            xx9 = list(all_txt[i][9:].split('，'))
            for i in xx9:
                if i not in rank and i!='xx':
                    rank.append(i)
    #standardise entity
    elif (i + 1) % data_len == 9:
        if len(all_txt[i])>5:
            xx10= list(all_txt[i][5:].split('，'))
            # print(xx9)
            for i in xx10:
                if i not in standardise and i!='xx':
                    standardise.append(i)
    # maintenance entity
    elif (i + 1) % data_len == 10:
        if len(all_txt[i])>7:
            xx11 = list(all_txt[i][7:].split('，'))
            # print(xx9)
            for i in xx11:
                if i not in maintenance and i!='xx':
                    maintenance.append(i)

def find_occurrences(text, pattern):
    occurrences = []
    start = 0
    while True:
        index = text.find(pattern, start)
        if index == -1:
            break
        occurrences.append(index)
        start = index + 1
    return occurrences

def str_loc(lines,shiti_list):
    # Input sentence with entity dictionary
    #Output the position of the entities contained in the sentence
    dingwei=[]
    xxx2=[]
    for i in shiti_list:
        if i in lines:
            xxx2.append(i)
    for i in xxx2:
        loc1=find_occurrences(lines, i)
        for j in range(len(loc1)):
            loc2=loc1[j]+len(i)
            dingwei.append([loc1[j],loc2])
    xxx3 = copy.deepcopy(dingwei)
    xxx4 = copy.deepcopy(dingwei)
    for i in range(len(xxx4)):
        for j in range(len(xxx3)):
            if (xxx4[i][0] > xxx3[j][0] and xxx4[i][1] < xxx3[j][1]) or (xxx4[i][0] == xxx3[j][0] and xxx4[i][1] < xxx3[j][1])\
                    or (xxx4[i][0] > xxx3[j][0] and xxx4[i][1] == xxx3[j][1]):
                if xxx4[i] in dingwei:
                    dingwei.remove(xxx4[i])
    return dingwei

def shititihuan_geshi(llines):
    # Input the sentence llines, locate all entities contained in the sentence according to the entity dictionary
    #Output format is [‘The object of this modelling is the third link of the Dadongmen Interchange Bridge, which is a continuous girder bridge,’, [[7, 16, ‘BN’], [21, 25, ‘BT’], [103, 112, ‘BT’], [159, 164, ‘BT’]]]]
    shiti_loc_liebiao=[]
    xx1=str_loc(llines,bridge_name)
    for i in xx1:
        shiti_loc_liebiao.append([i[0],i[1],"BN"])
    xx2=str_loc(llines,defect_location)
    for i in xx2:
        shiti_loc_liebiao.append([i[0],i[1],"DL"])
    xx3 = str_loc(llines, defect)
    for i in xx3:
        shiti_loc_liebiao.append([i[0], i[1], "DIS"])
    xx4=str_loc(llines,no_defect)
    for i in xx4:
        shiti_loc_liebiao.append([i[0],i[1],"NDIS"])
    xx5=str_loc(llines,defect_indicator)
    for i in xx5:
        shiti_loc_liebiao.append([i[0],i[1],"EI"])
    xx6=str_loc(llines,score)
    for i in xx6:
        shiti_loc_liebiao.append([i[0],i[1],"SC"])
    xx7=str_loc(llines,rank)
    for i in xx7:
        shiti_loc_liebiao.append([i[0],i[1],"RA"])
    xx8=str_loc(llines,standardise)
    for i in xx8:
        shiti_loc_liebiao.append([i[0],i[1],"ST"])
    xx9 = str_loc(llines, maintenance)
    for i in xx9:
        shiti_loc_liebiao.append([i[0],i[1],"MA"])
    return [llines,shiti_loc_liebiao]

def suijishitixuanze(dd_shiti):
    # Select a similar entity and replace it
    #Return value contains entities before and after replacement
    cc=[]
    for i in dd_shiti:
        if i in bridge_name:
            cc.append([i,random.sample(bridge_name, 1)])
        if i in defect_location:
            cc.append([i,random.sample(defect_location, 1)])
        if i in defect:
            cc.append([i,random.sample(defect, 1)])
        if i in no_defect:
            cc.append([i,random.sample(no_defect, 1)])
        if i in defect_indicator:
            cc.append([i,random.sample(defect_indicator, 1)])
        if i in score:
            cc.append([i,random.sample(score, 1)])
        if i in rank:
            cc.append([i,random.sample(rank, 1)])
        if i in standardise:
            cc.append([i,random.sample(standardise, 1)])
        if i in maintenance:
            cc.append([i,random.sample(maintenance, 1)])
    return cc

def shititihuan(e_data,data_len):
    # Sentence entity replacement function, input contains a list of all sentences e_data, replace the number of sentences data_len
    # Output the replaced sentences without entity labels.
    tihuan_juzi=random.sample(e_data, data_len)
    mm=[]
    for i in tihuan_juzi:
        shiti=[]
        chushi_juzi=i[0]
        chushibiaoqian=i[1]
        for j in chushibiaoqian:
            shiti.append(chushi_juzi[j[0]:j[1]])
        xx_tihuan=random.sample(shiti, 6)
        xxx_tihuan=suijishitixuanze(xx_tihuan)
        for k in xxx_tihuan:
            loc1 = chushi_juzi.find(k[0])
            loc2 = loc1 + len(k[0])
            jj1=chushi_juzi[:loc1]
            jj2=chushi_juzi[loc2:]
            chushi_juzi=jj1+k[1][0]+jj2
        mm.append(chushi_juzi)
    return mm

def duanju_suijijiaohuan(e_data,data_len):
    #random exchange some phrase position
    jiaohuan_juzi=random.sample(e_data, data_len)
    mm=[]
    for i in jiaohuan_juzi:
        ii=i[0].split('。')
        ii=[s+'。' for s in ii][:-1]
        ii_new = []
        for t in ii:
            ii1 = t.split('；')
            ii1 = [s + '；' for s in ii1]
            ii_new = ii_new + ii1
        for u in range(len(ii_new)):
            if ii_new[u][-2:] == '。；':
                ii_new[u] = ii_new[u][:-1]
        if len(ii_new)>3:
            new_ii = copy.deepcopy(ii_new)
            for k in range(3):
                ii_s=copy.deepcopy(new_ii)
                sunxu=random.sample(ii_s, 2)
                for j in range(len(ii_s)):
                    if ii_s[j]==sunxu[0]:
                        new_ii[j]=sunxu[1]
                    if ii_s[j]==sunxu[1]:
                        new_ii[j]=sunxu[0]
            new_juzi=''
            for m in new_ii:
                new_juzi=new_juzi+m
            mm.append(new_juzi)
    return mm

def duanju_suijishanchu(e_data,data_len):
    # random delect some phrases
    shanchu_juzi=random.sample(e_data, data_len)
    mm=[]
    for i in shanchu_juzi:
        ii=i[0].split('。')
        ii=[s+'。' for s in ii][:-1]
        ii_new = []
        for t in ii:
            ii1 = t.split('；')
            ii1 = [s + '；' for s in ii1]
            ii_new = ii_new + ii1
        for u in range(len(ii_new)):
            if ii_new[u][-2:] == '。；':
                ii_new[u] = ii_new[u][:-1]
        if len(ii_new)>5:
            new_ii=random.sample(ii_new, 3)
            for k in new_ii:
                ii_new.remove(k)
            new_juzi=''
            for m in ii_new:
                new_juzi=new_juzi+m
            mm.append(new_juzi)
    return mm

def remove_duplicates(lst):
    new_lst = []
    for item in lst:
        if item not in new_lst:
            new_lst.append(item)
    return new_lst

def duanju_charu(e_data,data_len):
    # random insert some phrases
    charu_juzi=[]
    mm=[]
    for i in e_data:
        ii=i[0].split('。')
        ii=[s+'。' for s in ii][:-1]
        ii_new = []
        for t in ii:
            ii1 = t.split('；')
            ii1 = [s + '；' for s in ii1]
            ii_new = ii_new + ii1
        for u in range(len(ii_new)):
            if ii_new[u][-2:] == '。；':
                ii_new[u] = ii_new[u][:-1]
        for k in ii_new:
                charu_juzi.append(k)
    duanjuzi=random.sample(e_data, data_len)
    for i in duanjuzi:
        ii=i[0].split('，')
        ii=[s+'，' for s in ii][:-1]
        ii_new_1 = []
        for t in ii:
            ii1 = t.split('；')
            ii1 = [s + '；' for s in ii1]
            ii_new_1 = ii_new_1 + ii1
        for u in range(len(ii_new_1)):
            if ii_new_1[u][-2:] == '。；':
                ii_new_1[u] = ii_new_1[u][:-1]
        charu_juzi_1 = random.sample(charu_juzi, 2)
        charuweizhi=np.random.randint(1, len(ii_new_1), [2])
        new_ii=copy.deepcopy(ii_new_1)
        for ss in range(len(charuweizhi)):
            if charu_juzi_1[ss] not in new_ii:
                new_ii=new_ii[:charuweizhi[ss]] + [charu_juzi_1[ss]] + new_ii[charuweizhi[ss]:]
        new_ii=remove_duplicates(new_ii)
        new_juzi=''
        for m in new_ii:
            new_juzi=new_juzi+m
        mm.append(new_juzi)
    return mm

random.shuffle(text_data)

train_data=[]
val_data=[]
test_data=[]
train_num=1
val_num=1
test_num=1
for i in range(train_num):
    ss=shititihuan_geshi(text_data[i])
    train_data.append(ss)
for i in range(val_num):
    ss=shititihuan_geshi(text_data[train_num+i])
    val_data.append(ss)
for i in range(test_num):
    ss=shititihuan_geshi(text_data[train_num+val_num+i])
    test_data.append(ss)

#exchange similar entities
zengqiang1=shititihuan(train_data,(len(train_data)//2))
#random exchange some phrase position
zengqiang2=duanju_suijijiaohuan(train_data,(len(train_data)//2))
#random delect some phrases
zengqiang3=duanju_suijishanchu(train_data,(len(train_data)//2))
#random insert some phrases
zengqiang4=duanju_charu(train_data,(len(train_data)//2))
print("Initial sentence number",len(train_data))
print("sentence number with exchange similar entities",(len(train_data)//2))
print("sentence number with random exchange some phrase position",(len(train_data)//2))
print("sentence number with random delect some phrases",(len(train_data)//2))
print("sentence number with random insert some phrases",(len(train_data)//2))

char_num=0
entity_num=0
for i in range(len(train_data)):
    char_num=len(train_data[i][0])+char_num
    entity_num=len(train_data[i][1])+entity_num
print("Initial training set characters number",char_num)
print("Initial training set entity number",entity_num)

zengqiang=zengqiang1+zengqiang2+zengqiang3+zengqiang4

for i in zengqiang:
    ss=shititihuan_geshi(i)
    train_data.append(ss)

char_num=0
entity_num=0
for i in range(len(train_data)):
    char_num=len(train_data[i][0])+char_num
    entity_num=len(train_data[i][1])+entity_num
print("Initial training set characters number after data augmentation",char_num)
print("Initial training set entity number after data augmentation",entity_num)

def biaoqianshengcheng(biaoqianweizhi):
    llen=biaoqianweizhi[1]-biaoqianweizhi[0]
    bbq=[]
    for i in range(llen):
        bbq.append('I-'+biaoqianweizhi[2])
    bbq[0]='B-'+biaoqianweizhi[2]
    return bbq

def biaoqian(e_data):
    xx1 = []
    for i in range(len(e_data[0])):
        xx1.append('O')
    for i in e_data[1]:
        xx2=biaoqianshengcheng(i)
        xx1[i[0]:i[1]]=xx2
    xs1=list(e_data[0])
    return [xs1,xx1]

train_data_lable=[]
val_data_lable=[]
test_data_lable=[]
for i in train_data:
    train_data_lable.append(biaoqian(i))
for i in val_data:
    val_data_lable.append(biaoqian(i))
for i in test_data:
    test_data_lable.append(biaoqian(i))

print("sentence number in train data:",len(train_data_lable))
with open('data/substructure/train_data.txt', 'w',encoding='utf-8') as f:
    for i in range(len(train_data_lable)):
        for j in range(len(train_data_lable[i][0])):
            f.write(train_data_lable[i][0][j]+ ' ' + train_data_lable[i][1][j]+'\r')
        f.write('end' + '\r')
print("sentence number in val data:",len(val_data_lable))
with open('data/substructure/val_data.txt', 'w',encoding='utf-8') as f:
    for i in range(len(val_data_lable)):
        for j in range(len(val_data_lable[i][0])):
            f.write(val_data_lable[i][0][j]+ ' ' + val_data_lable[i][1][j]+'\r')
        f.write('end' + '\r')
print("sentence number in test data:",len(test_data_lable))
with open('data/substructure/test_data.txt', 'w',encoding='utf-8') as f:
    for i in range(len(test_data_lable)):
        for j in range(len(test_data_lable[i][0])):
            f.write(test_data_lable[i][0][j]+ ' ' + test_data_lable[i][1][j]+'\r')
        f.write('end' + '\r')


# -*- coding:utf-8 -*-
'''
@Author: zgh
@Date: 2024-12-23
'''

from cnradical import Radical, RunOption
from pypinyin import lazy_pinyin,  Style
import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from joblib import dump, load
from ltp import LTP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 初始化LTP模型
ltp= LTP('./small')
def filter_words(sentence, words_to_remove):
    #Split sentences using LTP and remove punctuation and specific words
    segmented = ltp.pipeline([sentence], tasks = ["cws"], return_dict = False)
    words = segmented[0][0]
    words = [word for word in words if word not in words_to_remove and word.isalnum() and len(word)>=2]
    words = list(set(words))
    return words

#读取标注数据中所有实体
all_txt=[]
with open('data/substructure/lable_text.txt', 'r', encoding='utf-8') as f:
    for ann in f.readlines():
        ann = ann.strip('\n')
        ann = ann.replace(' ', '')
        all_txt.append(ann)

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
no_entity_word=[]

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

texts = []
labels = []
bridge_name_label=['桥梁名称' for i in bridge_name]
defect_location_label=['病害位置' for i in defect_location]
defect_label=['病害' for i in defect]
no_defect_label=['无病害' for i in no_defect]
defect_indicator_label=['评价指标' for i in defect_indicator]
score_label=['技术状况得分' for i in score]
rank_label=['技术状况等级' for i in rank]
standardise_label=['规范' for i in standardise]
maintenance_label=['维修措施' for i in maintenance]
texts = texts+bridge_name+defect_location+defect+no_defect+defect_indicator+score+rank+standardise+maintenance
labels = labels+bridge_name_label+defect_location_label+defect_label+no_defect_label+defect_indicator_label+score_label+rank_label+standardise_label+maintenance_label
ltp.add_words(words=texts)

for i in text_data:
    filtered_result = filter_words(i, texts)
    no_entity_word=no_entity_word+filtered_result

no_entity_word_lable=['其他' for i in no_entity_word]
texts=texts+no_entity_word
labels=labels+no_entity_word_lable

def load_params():
    with open("models/data.pkl", "rb") as fopen:
        data_map = pickle.load(fopen)
    return data_map
def getPositionEncoding(tra_seq, tra_dim, n=10000):
    PE = np.zeros(shape=(tra_seq, tra_dim))
    for pos in range(tra_seq):
        for i in range(int(tra_dim / 2)):
            denominator = np.power(n, 2 * i / tra_dim)
            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)
    return PE
data_map = load_params()
input_size = data_map.get("input_size")
tag_map = data_map.get("tag_map")
vocab = data_map.get("vocab")
bushouzidian = data_map.get("bushouzidian")
pinyinzidian = data_map.get("pinyinzidian")
def phrase_encoding(input_str):
    if len(input_str)<10:
        input_str=input_str+'x'*(10-len(input_str))
    elif len(input_str)>10:
        input_str=input_str[:10]
    all_line = list(input_str)
    bushou = []
    pinyin = []
    weizhi = []
    zimubiao = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12,
                "m": 13,
                "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24,
                "y": 25, "z": 26, }

    for word in all_line:
        radical = Radical(RunOption.Radical)  # 获取偏旁
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
        bushou.append(bushouzidian.get(radical_out, 0))

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

    input_vec = [vocab.get(i, 0) for i in input_str]
    sentences = torch.tensor(input_vec).view(1, -1)
    bushou_vec = torch.tensor(bushou).view(1, -1)
    pinyin_vec = torch.tensor(pinyin).view(1, -1, 27)
    weizhi_vec = torch.tensor(weizhi).view(1, -1, 2)
    tra_weizhi = getPositionEncoding(len(input_vec), 27, n=10000)
    tra_weizhi_vec = torch.tensor(tra_weizhi).view(1, -1, 27)
    svn_bianma=[]
    for i in range(len(input_vec)):
        svn_bianma.append([input_vec[i]]+[bushou[i]]+pinyin[i]+list(tra_weizhi[i]))
    svn_bianma = [item for sublist in svn_bianma for item in sublist]
    return svn_bianma

# 创建字典将训练集实体编码结果保存到字典文件my_dict.pickle中
my_dict = {}
# texts_encoding=[]
for i in range(len(texts)):
# for i in range(10):
    print(i,'/',len(texts))
    # print(len(phrase_encoding(texts[i])))
    # texts_encoding.append(phrase_encoding(texts[i]))
    my_dict[texts[i]] = np.array(phrase_encoding(texts[i]))
# print(my_dict)
# 写入pickle文件
with open('my_dict.pickle', 'wb') as pickle_file:
    pickle.dump(my_dict, pickle_file)

with open('my_dict.pickle', 'rb') as pickle_file:
    loaded_dict = pickle.load(pickle_file)

texts_encoding=[]
for i in range(len(texts)):
    texts_encoding.append(loaded_dict[texts[i]])
texts_encoding=np.array(texts_encoding)

# Dimensionality reduction operations using PCA from 560 to 128 dimensions
pca = PCA(n_components=128)
X_reduced = pca.fit_transform(texts_encoding)

print('Divide the entity classifier training set and test set')
X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels, test_size=0.3, random_state=42)

# loaded models to calculate accuracy
loaded_svm = load('./clf_model/svm_voting.joblib')
new_predictions = loaded_svm.predict(X_test)
loaded_svm_acc = accuracy_score(y_test, new_predictions)
print(f'Support vector machine prediction correctness (支持向量机预测正确率): {loaded_svm_acc:.4f}')
loaded_gbt = load('./clf_model/gbt_voting.joblib')
new_predictions = loaded_gbt.predict(X_test)
loaded_gbt_acc = accuracy_score(y_test, new_predictions)
print(f'Gradient boosting tree prediction correctness (梯度提升树预测正确率): {loaded_gbt_acc:.4f}')
loaded_knn = load('./clf_model/knn_voting.joblib')
new_predictions = loaded_knn.predict(X_test)
loaded_knn_acc = accuracy_score(y_test, new_predictions)
print(f'KNN prediction correctness (KNN预测正确率): {loaded_knn_acc:.4f}')
loaded_rf = load('./clf_model/rf_voting.joblib')
new_predictions = loaded_rf.predict(X_test)
loaded_rf_acc = accuracy_score(y_test, new_predictions)
print(f'Random forest prediction correctness (随机森林预测正确率): {loaded_rf_acc:.4f}')
all_acc=loaded_svm_acc+loaded_rf_acc+loaded_gbt_acc+loaded_knn_acc
all_w=[loaded_svm_acc/all_acc, loaded_gbt_acc/all_acc, loaded_knn_acc/all_acc, loaded_rf_acc/all_acc]

def weighted_sum(dicts, weights):
    if len(dicts) != len(weights):
        raise ValueError("字典列表和权重列表的长度必须相同")
    all_keys = set(key for d in dicts for key in d)
    weighted_sum_dict = {key: 0 for key in all_keys}
    for d, weight in zip(dicts, weights):
        for key in all_keys:
            weighted_sum_dict[key] += d.get(key, 0) * weight
    if not weighted_sum_dict:
        return None
    max_key = max(weighted_sum_dict, key=weighted_sum_dict.get)
    return max_key, weighted_sum_dict[max_key]

svm_clf = load('./clf_model/svm_voting.joblib')
gbt_clf = load('./clf_model/gbt_voting.joblib')
knn_clf = load('./clf_model/knn_voting.joblib')
rf_clf = load('./clf_model/rf_voting.joblib')

#Voting Classifier Correctness Calculation
def voting_classfier_acc(svm_clf,gbt_clf,knn_clf,rf_clf,yuce_list):
    all_fin_res=[]
    for i in range(len(yuce_list)):
        new_vector_reduced = [yuce_list[i]]
        svm_classes = svm_clf.classes_
        gbt_classes = gbt_clf.classes_
        knn_classes = knn_clf.classes_
        rf_classes = rf_clf.classes_
        svm_proba = svm_clf.predict_proba(new_vector_reduced)
        gbt_proba = gbt_clf.predict_proba(new_vector_reduced)
        knn_proba = knn_clf.predict_proba(new_vector_reduced)
        rf_proba = rf_clf.predict_proba(new_vector_reduced)
        svm_proba_dict = dict(zip(svm_classes, svm_proba[0]))
        gbt_proba_dict = dict(zip(gbt_classes, gbt_proba[0]))
        knn_proba_dict = dict(zip(knn_classes, knn_proba[0]))
        rf_proba_dict = dict(zip(rf_classes, rf_proba[0]))
        fin_res=weighted_sum([svm_proba_dict, gbt_proba_dict, knn_proba_dict, rf_proba_dict], all_w)
        all_fin_res.append(fin_res)
    return all_fin_res

acc_num=0
acc_data = voting_classfier_acc(svm_clf,gbt_clf,knn_clf,rf_clf,X_test)
for i in range(len(X_test)):
    if acc_data[i][0]==y_test[i]:
        acc_num=acc_num+1

print(f'Voting classifier prediction correctness (投票分类器预测正确率): {acc_num/len(acc_data):.4f}')

def zw_predict(svm_clf,gbt_clf,knn_clf,rf_clf,yuce_list):
    all_fin_res=[]
    all_clf_pro=[]
    for i in range(len(yuce_list)):
        new_vector = np.array(phrase_encoding(yuce_list[i]))
        new_vector_reduced = pca.transform([new_vector])  # 对新向量进行降维
        # 获取类别标签
        svm_classes = svm_clf.classes_
        gbt_classes = gbt_clf.classes_
        knn_classes = knn_clf.classes_
        rf_classes = rf_clf.classes_
        # 对新数据点进行预测并输出概率
        svm_proba = svm_clf.predict_proba(new_vector_reduced)
        gbt_proba = gbt_clf.predict_proba(new_vector_reduced)
        knn_proba = knn_clf.predict_proba(new_vector_reduced)
        rf_proba = rf_clf.predict_proba(new_vector_reduced)
        # 将类别和概率对应起来
        svm_proba_dict = dict(zip(svm_classes, svm_proba[0]))
        gbt_proba_dict = dict(zip(gbt_classes, gbt_proba[0]))
        knn_proba_dict = dict(zip(knn_classes, knn_proba[0]))
        rf_proba_dict = dict(zip(rf_classes, rf_proba[0]))
        fin_res, clf_pro=weighted_sum([svm_proba_dict, gbt_proba_dict, knn_proba_dict, rf_proba_dict], all_w)
        all_clf_pro.append(clf_pro)
        all_fin_res.append(fin_res)
    return all_fin_res,all_clf_pro

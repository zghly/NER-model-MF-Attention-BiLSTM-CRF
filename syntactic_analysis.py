# -*- coding:utf-8 -*-
'''
@Author: zgh
@Date: 2024-12-23
'''

import copy
from main import ChineseNER,pre_find_fun
import re
from ltp import LTP
from joblib import dump, load
from updata import zw_predict
ltp = LTP('./small') # 默认加载 LTP/Small 模型

def str_loc(lines,shiti_list):
    # Input sentence with entity dictionary
    # Output the position of the entities contained in the sentence
    dingwei=[]
    xxx2=[]
    for i in shiti_list:
        if i in lines:
            xxx2.append(i)
    for i in xxx2:
        loc1=[substr.start() for substr in re.finditer(i, lines)]
        for j in range(len(loc1)):
            loc2=loc1[j]+len(i)
            dingwei.append([i,loc1[j],loc2])
    #Deletion for containing relational entities
    xxx3 = copy.deepcopy(dingwei)
    xxx4 = copy.deepcopy(dingwei)
    for i in range(len(xxx4)):
        for j in range(len(xxx3)):
            if (xxx4[i][1] > xxx3[j][1] and xxx4[i][2] < xxx3[j][2]):
                if xxx4[i] in dingwei:
                    dingwei.remove(xxx4[i])
    return dingwei

def quchong(dingwei):
    xxx3 = copy.deepcopy(dingwei)
    xxx4 = copy.deepcopy(dingwei)
    for i in range(len(xxx4)):
        for j in range(len(xxx3)):
            if (xxx4[i][1] >= xxx3[j][1] and xxx4[i][2] < xxx3[j][2]) or (xxx4[i][1] > xxx3[j][1] and xxx4[i][2] <= xxx3[j][2]):
                if xxx4[i] in dingwei:
                    dingwei.remove(xxx4[i])
    return dingwei

def dis_clf_find(ner_list,list_lable):
# function is used to find the entity list label and voting classifier prediction results inconsistent entity
# Output bner_list is the list after removing the inconsistent entities,
# false_clf contains the inconsistent entities [Polling Classifier Prediction Category, Entity, Prediction Probability].
    svm_clf = load('./clf_model/svm_voting.joblib')
    gbt_clf = load('./clf_model/gbt_voting.joblib')
    knn_clf = load('./clf_model/knn_voting.joblib')
    rf_clf = load('./clf_model/rf_voting.joblib')
    ner_list_clf, ss = zw_predict(svm_clf,gbt_clf,knn_clf,rf_clf,ner_list)
    false_clf=[]
    for i in range(len(ner_list_clf)):
        #predical probability threshold
        if ner_list_clf[i] != list_lable and ss[i]>0.65:
            false_clf.append([ner_list_clf[i],ner_list[i],ss[i]])
    bner_list = copy.deepcopy(ner_list)
    for i in false_clf:
        if i[1] in ner_list:
            bner_list.remove(i[1])
    return bner_list,false_clf

#Optimisation of named entity recognition results based on syntactic dependency analysis and voting classifier results to reduce missed entities
def syntactic_completion(pre_sentence,modle_loc):
    cn = ChineseNER("predict",modle_loc)
    org_sentences, pre_result = cn.predict(input_str=pre_sentence)
    pre_result1 = copy.deepcopy(pre_result)

    #Get the prediction results for all tags of the sentence
    bn = pre_find_fun(pre_sentence, pre_result1, '-BN')
    dl = pre_find_fun(pre_sentence, pre_result1, '-DL')
    dis = pre_find_fun(pre_sentence, pre_result1, '-DIS')
    ndis, loc_ndis = pre_find_fun(pre_sentence, pre_result1, '-NDIS')
    ei = pre_find_fun(pre_sentence, pre_result1, '-EI')
    sc = pre_find_fun(pre_sentence, pre_result1, '-SC')
    ra = pre_find_fun(pre_sentence, pre_result1, '-RA')
    st = pre_find_fun(pre_sentence, pre_result1, '-ST')
    ma = pre_find_fun(pre_sentence, pre_result1, '-MA')
    bn = [item for item in bn if len(item) != 1]
    dl = [item for item in dl if len(item) != 1]
    dis = [item for item in dis if len(item) != 1]
    ndis = [item for item in ndis if len(item) != 1]
    ei = [item for item in ei if len(item) != 1]
    sc = [item for item in sc if len(item) != 1]
    ra = [item for item in ra if len(item) != 1]
    st = [item for item in st if len(item) != 1]
    ma = [item for item in ma if len(item) != 1]

    ltp.add_words(words=bn)
    ltp.add_words(words=dl)
    ltp.add_words(words=dis)
    ltp.add_words(words=ndis)
    ltp.add_words(words=ei)
    ltp.add_words(words=sc)
    ltp.add_words(words=ra)
    ltp.add_words(words=st)
    ltp.add_words(words=ma)

    ii_1 = pre_sentence.split('。')
    ii_1 = [s + '。' for s in ii_1][:-1]
    ii_2 = []
    for t in ii_1:
        ii1 = t.split('；')
        ii1 = [s + '；' for s in ii1]
        ii_2 = ii_2 + ii1
    for u in range(len(ii_2)):
        if ii_2[u][-2:] == '。；':
            ii_2[u] = ii_2[u][:-1]
    add_ner=[]
    for i in range(len(ii_2)):
        tra_bn = quchong(str_loc(ii_2[i], bn))
        tra_dl = quchong(str_loc(ii_2[i], dl))
        tra_dis = quchong(str_loc(ii_2[i], dis))
        tra_ndis = quchong(str_loc(ii_2[i], ndis))
        tra_ei = quchong(str_loc(ii_2[i], ei))
        tra_sc = quchong(str_loc(ii_2[i], sc))
        tra_ra = quchong(str_loc(ii_2[i], ra))
        tra_st = quchong(str_loc(ii_2[i], st))
        tra_ma = quchong(str_loc(ii_2[i], ma))

        #Dependent syntactic analysis using the ltp tool to get the subject and predicate in the sentence
        try:
            result = ltp.pipeline([ii_2[i]], tasks=["cws", "dep"])
            zhuyu = []
            for j in range(len(result.dep[0]['label'])):
                if (result.dep[0]['label'][j] == 'SBV'):
                    zhuyu.append(result.cws[0][j])
            weiyu = []
            for j in range(len(result.dep[0]['label'])):
                if (result.dep[0]['label'][j] == 'VOB' or result.dep[0]['label'][j] == 'COO'):
                    weiyu.append(result.cws[0][j])
        except Exception as e:
            zhuyu = []
            weiyu = []

        tra_bn_quchu = [ijk[0] for ijk in tra_bn]
        tra_dl_quchu = [ijk[0] for ijk in tra_dl]
        tra_dis_quchu = [ijk[0] for ijk in tra_dis]
        tra_ndis_quchu = [ijk[0] for ijk in tra_ndis]
        tra_ei_quchu = [ijk[0] for ijk in tra_ei]
        tra_sc_quchu = [ijk[0] for ijk in tra_sc]
        tra_ra_quchu = [ijk[0] for ijk in tra_ra]
        tra_st_quchu = [ijk[0] for ijk in tra_st]
        tra_ma_quchu = [ijk[0] for ijk in tra_ma]

        zhuyu = [item for item in zhuyu if item not in tra_bn_quchu and not any(item in s for s in tra_bn_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_dl_quchu and not any(item in s for s in tra_dl_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_dis_quchu and not any(item in s for s in tra_dis_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_ndis_quchu and not any(item in s for s in tra_ndis_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_ei_quchu and not any(item in s for s in tra_ei_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_sc_quchu and not any(item in s for s in tra_sc_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_ra_quchu and not any(item in s for s in tra_ra_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_st_quchu and not any(item in s for s in tra_st_quchu)]
        zhuyu = [item for item in zhuyu if item not in tra_ma_quchu and not any(item in s for s in tra_ma_quchu)]

        weiyu = [item for item in weiyu if item not in tra_bn_quchu and not any(item in s for s in tra_bn_quchu)]
        weiyu = [item for item in weiyu if item not in tra_dl_quchu and not any(item in s for s in tra_dl_quchu)]
        weiyu = [item for item in weiyu if item not in tra_dis_quchu and not any(item in s for s in tra_dis_quchu)]
        weiyu = [item for item in weiyu if item not in tra_ndis_quchu and not any(item in s for s in tra_ndis_quchu)]
        weiyu = [item for item in weiyu if item not in tra_ei_quchu and not any(item in s for s in tra_ei_quchu)]
        weiyu = [item for item in weiyu if item not in tra_sc_quchu and not any(item in s for s in tra_sc_quchu)]
        weiyu = [item for item in weiyu if item not in tra_ra_quchu and not any(item in s for s in tra_ra_quchu)]
        weiyu = [item for item in weiyu if item not in tra_st_quchu and not any(item in s for s in tra_st_quchu)]
        weiyu = [item for item in weiyu if item not in tra_ma_quchu and not any(item in s for s in tra_ma_quchu)]

        add_ner=add_ner+zhuyu
        add_ner=add_ner+weiyu

    bn_sy = copy.deepcopy(bn)
    dl_sy = copy.deepcopy(dl)
    dis_sy = copy.deepcopy(dis)
    ndis_sy = copy.deepcopy(ndis)
    ei_sy = copy.deepcopy(ei)
    sc_sy = copy.deepcopy(sc)
    ra_sy = copy.deepcopy(ra)
    st_sy = copy.deepcopy(st)
    ma_sy = copy.deepcopy(ma)

    svm_clf = load('./clf_model/svm_voting.joblib')
    gbt_clf = load('./clf_model/gbt_voting.joblib')
    knn_clf = load('./clf_model/knn_voting.joblib')
    rf_clf = load('./clf_model/rf_voting.joblib')
    add_ner_class, yucegailv = zw_predict(svm_clf,gbt_clf,knn_clf,rf_clf,add_ner)

    for i in range(len(add_ner_class)):
        if add_ner_class[i] == '桥梁名称':
            bn_sy.append(add_ner[i])
        elif add_ner_class[i] == '病害位置':
            dl_sy.append(add_ner[i])
        elif add_ner_class[i] == '病害':
            dis_sy.append(add_ner[i])
        elif add_ner_class[i] == '无病害':
            ndis_sy.append(add_ner[i])
        elif add_ner_class[i] == '评价指标':
            ei_sy.append(add_ner[i])
        elif add_ner_class[i] == '技术状况得分':
            sc_sy.append(add_ner[i])
        elif add_ner_class[i] == '技术状况等级':
            ra_sy.append(add_ner[i])
        elif add_ner_class[i] == '规范':
            st_sy.append(add_ner[i])
        elif add_ner_class[i] == '维修措施':
            ma_sy.append(add_ner[i])

    #Optimisation of entity recognition results based on voting classifiers to reduce misclassified entities
    bn_clf = copy.deepcopy(bn_sy)
    dl_clf = copy.deepcopy(dl_sy)
    dis_clf = copy.deepcopy(dis_sy)
    ndis_clf = copy.deepcopy(ndis_sy)
    ei_clf = copy.deepcopy(ei_sy)
    sc_clf = copy.deepcopy(sc_sy)
    ra_clf = copy.deepcopy(ra_sy)
    st_clf = copy.deepcopy(st_sy)
    ma_clf = copy.deepcopy(ma_sy)
    #Find entities whose entity identification results differ from the voting classifier results
    err_clf=[]
    bn_clf_1, err_clf_bn = dis_clf_find(bn_clf, '桥梁名称')
    err_clf = err_clf + err_clf_bn
    dl_clf_1, err_clf_dl = dis_clf_find(dl_clf, '病害位置')
    err_clf = err_clf + err_clf_dl
    dis_clf_1, err_clf_dis = dis_clf_find(dis_clf, '病害')
    err_clf = err_clf + err_clf_dis
    ndis_clf_1, err_clf_ndis = dis_clf_find(ndis_clf, '无病害')
    err_clf = err_clf + err_clf_ndis
    ei_clf_1, err_clf_ei = dis_clf_find(ei_clf, '评价指标')
    err_clf = err_clf + err_clf_ei
    sc_clf_1, err_clf_sc = dis_clf_find(sc_clf, '技术状况得分')
    err_clf = err_clf + err_clf_sc
    ra_clf_1, err_clf_ra = dis_clf_find(ra_clf, '技术状况等级')
    err_clf = err_clf + err_clf_ra
    st_clf_1, err_clf_st = dis_clf_find(st_clf, '规范')
    err_clf = err_clf + err_clf_st
    ma_clf_1, err_clf_ma = dis_clf_find(ma_clf, '维修措施')
    err_clf = err_clf + err_clf_ma

    for i in range(len(err_clf)):
        if err_clf[i][0]=='桥梁名称':
            bn_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='病害位置':
            dl_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='病害':
            dis_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='无病害':
            ndis_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='评价指标':
            ei_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='技术状况得分':
            sc_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='技术状况等级':
            ra_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='规范':
            st_clf_1.append(err_clf[i][1])
        if err_clf[i][0]=='维修措施':
            ma_clf_1.append(err_clf[i][1])

    # Entity identification results before and after the correction
    xx_1 = [bn, dl, dis, ndis, ei, sc, ra, st, ma]
    xx_2 = [bn_sy, dl_sy, dis_sy, ndis_sy, ei_sy, sc_sy, ra_sy, st_sy, ma_sy]
    xx_3 = [bn_clf_1, dl_clf_1,dis_clf_1,ndis_clf_1,ei_clf_1,sc_clf_1,ra_clf_1,st_clf_1,ma_clf_1]
    return  [add_ner,add_ner_class],xx_1, xx_2, xx_3, err_clf
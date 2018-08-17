# coding=utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017-2087 TIC-Recruit Project
# @Author  : Chao Ma <machao13@baidu.com>
# @Time    : 2018/4/24 22:02
# @File    : iterator_in_deepth.py
# @Software: PyCharm
"""
__file__
    iterator_in_deepth.py

__description__

    todo
__author__

    Chao Ma <machao13@baidu.com>

__date__

    2018/4/24 22:02
"""
import log
import jieba
import gensim
import numpy as np
from gensim.corpora import Dictionary

log.init_log(log_fold="./log/", log_name="polylda")

STOP_WORDS_ADDR_ZN = './reference/stopwords'
STOP_WORDS_ADDR_EN = './reference/stopwords_en'

STOPWORD_SET_ZN = set([line.strip().decode('utf-8') for line in open(STOP_WORDS_ADDR_ZN, 'rb')])
STOPWORD_SET_EN = set([line.strip().decode('utf-8') for line in open(STOP_WORDS_ADDR_EN, 'rb')])

STOPWORD = STOPWORD_SET_ZN | STOPWORD_SET_EN

def tokenize(line, stop_words):
    segs = jieba.cut(line, cut_all=False)
    final = []
    for seg in segs:
        if (len(seg) >= 2) and (seg not in stop_words):
            final.append(seg.lower())  #little trick but important, lower很重要
    return final


Corpus_1 = ["擅长机器学习，数据挖掘。熟悉nlp。", "有阿里巴巴实习经历。"]
Corpus_2 = ["擅长机器学习，数据挖掘。熟悉nlp。", "有阿里巴巴实习经历。"]

Corpus_1 = [tokenize(document, STOPWORD) for document in Corpus_1]
Corpus_2 = [tokenize(document, STOPWORD) for document in Corpus_2]

dct1 = Dictionary(Corpus_1)
dct2 = Dictionary(Corpus_2)

Corpus_1 = [dct1.doc2bow(document) for document in Corpus_1]
Corpus_2 = [dct2.doc2bow(document) for document in Corpus_2]

Corpus_matrix_1 = gensim.matutils.corpus2dense(Corpus_1, len(dct1)).T.astype(np.int).tolist()
Corpus_matrix_2 = gensim.matutils.corpus2dense(Corpus_2, len(dct2)).T.astype(np.int).tolist()


if __name__=="__main__":
    from polylda import PolyLDA
    polylda = PolyLDA(n_topics=100,n_iter=100, languages=2)
    a = np.random.randint(0, 100, 900).reshape(30, 30)
    b = polylda.fit_transform([Corpus_matrix_1,Corpus_matrix_2])

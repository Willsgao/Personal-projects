# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import warnings
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import hmmlearn.hmm as hl

warnings.filterwarnings(
    'ignore', category=DeprecationWarning)
np.seterr(all='ignore')


def search_speeches(directory, speeches):
    # 将不同系统路径格式统一化
    directory = os.path.normpath(directory)
    if not os.path.isdir(directory):
        raise IOError("路径" + directory + '不存在')
    # 获取文件夹中的子目录
    for entry in os.listdir(directory):
        # 获取路径的分类文件夹名作为分类标签
        label = directory[directory.rfind(
            os.path.sep) + 1:]
        # 拼接新的路径path
        path = os.path.join(directory, entry)
        # 如果新路径还是文件夹则递归查询
        if os.path.isdir(path):
            search_speeches(path, speeches)
        # 如果是'.wav'结尾的文件则进一步处理
        elif os.path.isfile(path) and \
                path.endswith('.wav'):
            if label not in speeches:
                speeches[label] = []
            speeches[label].append(path)

# 训练模型
train_speeches = {}
speeches = search_speeches(
    'speeches/training', train_speeches)

train_x, train_y = [], []
for label, filenames in train_speeches.items():
    mfccs = np.array([])
    for filename in filenames:
        sample_rate, sigs = wf.read(filename)
        mfcc = sf.mfcc(sigs, sample_rate)
        if len(mfccs) == 0:
            mfccs = mfcc
        else:
            mfccs = np.append(mfccs, mfcc, axis=0)
    train_x.append(mfccs)
    train_y.append(label)

models = {}
for mfccs, label in zip(train_x, train_y):
    model = hl.GaussianHMM(
        n_components=4, covariance_type='diag',
        n_iter=1000)
    models[label] = model.fit(mfccs)

# 测试模型
test_speeches = {}
speeches = search_speeches(
    'speeches/testing', test_speeches)

test_x, test_y = [], []
for label, filenames in test_speeches.items():
    mfccs = np.array([])
    for filename in filenames:
        sample_rate, sigs = wf.read(filename)
        mfcc = sf.mfcc(sigs, sample_rate)
        if len(mfccs) == 0:
            mfccs = mfcc
        else:
            mfccs = np.append(mfccs, mfcc, axis=0)
    test_x.append(mfccs)
    test_y.append(label)

pred_test_y = []
for mfccs in test_x:
    best_score, best_label = None, None
    for label, model in models.items():
        score = model.score(mfccs)
        if (best_score is None) or (
                best_score < score):
            best_score, best_label = score, label
    pred_test_y.append(best_label)

print(test_y)
print(pred_test_y)

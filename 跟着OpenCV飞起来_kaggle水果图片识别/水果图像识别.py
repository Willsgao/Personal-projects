# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import warnings
import numpy as np
import cv2 as cv
import hmmlearn.hmm as hl
warnings.filterwarnings(
    'ignore', category=DeprecationWarning)
np.seterr(all='ignore')


def search_objects(directory):
    directory = os.path.normpath(directory)
    if not os.path.isdir(directory):
        raise IOError(directory + '不是文件夹')
    objects = {}
    for curdir, _, files in os.walk(directory):
        for jpeg in [file for file in files
                     if file.endswith('.jpg')]:
            path = os.path.join(curdir, jpeg)
            label = jpeg.split('_')[0]
            # label = path.split(os.path.sep)[-2]
            if label not in objects:
                objects[label] = []
            objects[label].append(path)
    return objects


# 收集图片数据集的标签和灰度值
def label_desc02(objects, flags=None):
    data_x, data_y, data_z = [], [], []
    for label, files in objects.items():
        descs = np.array([])
        for file in files:
            image = cv.imread(file)
            # 训练阶段可不收集图片，不用给定flags值
            if flags:
                data_z.append([])
                data_z[-1].append(image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            f = 600 / min(h, w)
            gray = cv.resize(gray, None, fx=f, fy=f)
            star = cv.xfeatures2d.StarDetector_create()
            keypoints = star.detect(gray)
            sift = cv.xfeatures2d.SIFT_create()
            _, desc = sift.compute(gray, keypoints)
            if len(descs) == 0:
                descs = desc
                # print(descs)
            else:
                descs = np.append(descs, desc, axis=0)
                # print(descs)
            if flags:
                data_x.append(descs)
                data_y.append(label)
        if not flags:
            data_x.append(descs)
            data_y.append(label)
    return data_x, data_y, data_z


# 训练模型函数，hmm模型
def model_train(data_x, data_y):
    models = {}
    for descs, label in zip(data_x, data_y):
        model = hl.GaussianHMM(
            n_components=4, covariance_type='diag',
            n_iter=1000)
        models[label] = model.fit(descs)
    return models


# 定义利用图像识别进行模型预测的函数
def model_pred(test_x, models):
    pred_test_y = []
    for descs in test_x:
        # for desc in descs:
        best_score, best_label = None, None
        for label, model in models.items():
            score = model.score(descs)
            if not best_score or (
                    best_score < score):
                best_score, best_label = score, label
        pred_test_y.append(best_label)
    return pred_test_y


# 可视化识别的图片
def show_pics(test_y, pred_test_y, test_z):
    i = 0
    for label, pred_label, images in zip(
            test_y, pred_test_y, test_z):
        for image in images:
            i += 1
            style = '{} - {} {} {}'.format(
                i, label,
                '==' if label == pred_label
                else '!=', pred_label)
            cv.imshow(style, image)


if __name__ == '__main__':
    train_path = 'train'
    test_path = 'test'
    train_files = search_objects(train_path)
    test_files = search_objects(test_path)
    train_x, train_y, _ = label_desc02(train_files)
    test_x, test_y, test_z = label_desc02(test_files, 1)

    models = model_train(train_x, train_y)
    pred_test_y = model_pred(test_x, models)
    print(test_y)
    print(pred_test_y)
    # print(test_files)
    show_pics(test_y, pred_test_y, test_z)
    cv.waitKey()

from fastai.vision import *
from fastai.metrics import error_rate
import warnings
import torch
import os

warnings.filterwarnings('ignore')


def predict():
    learn = load_learner(path='./data/models/')

    img = open_image('imgs/img2inference.jpg')

    pred_class, pred_idx, outputs = learn.predict(img)
    res_name = ''
    for name, _class in learn.data.c2i.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if _class == pred_class.data.item():
            print(name, round(torch.max(outputs).item() * 100, 4), '%')
            res_name = name
    return res_name, str(round(torch.max(outputs).item() * 100, 4))

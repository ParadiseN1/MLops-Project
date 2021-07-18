from fastai.vision import *
from fastai.metrics import error_rate
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    learn = load_learner(path='/Users/sv/.fastai/data/oxford-iiit-pet/images/models/')

    img = open_image('imgs/Beagle_harrier.jpeg')

    pred_class,pred_idx,outputs = learn.predict(img)

    for name, _class in learn.data.c2i.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if _class == pred_class.data.item():
            print(name)

"""
Creates more training examples
by rotating existing pictures
"""

import pandas as pd
import imageio
import sys
from PIL import Image
from joblib import Parallel, delayed
from scipy.ndimage import rotate
from glob import glob

X_train = pd.read_pickle('./pickle/traindata')
y_train = pd.read_pickle('./pickle/trainlabels')

pos_ind = X_train[y_train == 1].index

def get_new_y(y, multiplier):
    temp_df = X_train[y_train == 1]
    temp_df = y_train.reindex(temp_df.index + (multiplier * 1000000))
    temp_df.loc[:] = 1
    return y.append(temp_df)

def get_new_X(X, multiplier):
    temp_df = X_train[y_train == 1]
    temp_df = temp_df.set_index(temp_df.index + (multiplier * 1000000))
    return X.append(temp_df)

def augment_df(degrees):
    if degrees > 269:
        i = range(0, 3)
    elif degrees > 179:
        i = range(0, 2)
    elif degrees > 89:
        i = range(0, 1)
    else:
        return

    X_train_aug = X_train
    y_train_aug = y_train

    for j in i:
        X_train_aug = get_new_X(X_train_aug, j + 1)
        y_train_aug = get_new_y(y_train_aug, j + 1)

    X_train_aug.to_pickle('./nodules/pickle/traindata')
    y_train_aug.to_pickle('./nodules/pickle/trainlabels')


def augment(idx, degrees):
    img_path = './nodules/train/nodule_' + str(idx) + '.jpg'
    img_path90 = './nodules/train/nodule_' + str(idx + 1000000) + '.jpg'
    img_path180 = './nodules/train/nodule_' + str(idx + 2000000) + '.jpg'
    img_path270 = './nodules/train/nodule_' + str(idx + 3000000) + '.jpg'

    base = imageio.imread(img_path)

    if (glob(img_path90) == [] and degrees > 89):
        rot90 = rotate(base, 90, reshape = False)
        Image.fromarray(rot90).convert('L').save(img_path90)
    
    if (glob(img_path180) == [] and degrees > 179):
        rot180 = rotate(base, 180, reshape = False)
        Image.fromarray(rot180).convert('L').save(img_path180)

    if (glob(img_path270) == [] and degrees > 269):
        rot270 = rotate(base, 180, reshape = False)
        Image.fromarray(rot180).convert('L').save(img_path270)


def main():
    if len(sys.argv) < 2:
        raise ValueError('<rot> argument not found. Specify "90", "180" or "270"')
    else:
        rot = sys.argv[1]
        if rot not in ['90', '180', '270']:
            raise ValueError('Invalid <rot> argument. Specify "90", "180" or "270"')

        rot = int(rot)

        Parallel(n_jobs = 6)(delayed(augment)(idx, rot) for idx in pos_ind)
        augment_df(rot)

if __name__ == "__main__":
    main()
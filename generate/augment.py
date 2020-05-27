"""
Creates more training examples
by rotating existing pictures
"""

import pandas as pd
import imageio
from PIL import Image
from joblib import Parallel, delayed
from scipy.ndimage import rotate
from glob import glob

X_train = pd.read_pickle('./pickle/traindata')
y_train = pd.read_pickle('./pickle/trainlabels')

pos_ind = X_train[y_train == 1].index

def augment(idx):
    img_path = './nodules/train/nodule_' + str(idx) + '.jpg'
    img_path90 = './nodules/train/nodule_' + str(idx + 1000000) + '.jpg'
    img_path180 = './nodules/train/nodule_' + str(idx + 2000000) + '.jpg'
    img_path270 = './nodules/train/nodule_' + str(idx + 3000000) + '.jpg'

    base = imageio.imread(img_path)

    if glob(img_path90) == []:
        print('rot90')
        rot90 = rotate(base, 90, reshape = False)
        Image.fromarray(rot90).convert('L').save(img_path90)
    
    if glob(img_path180) == []:
        print('rot180')
        rot180 = rotate(base, 180, reshape = False)
        Image.fromarray(rot180).convert('L').save(img_path180)

    if glob(img_path270) == []:
        print('rot270')
        rot270 = rotate(base, 180, reshape = False)
        Image.fromarray(rot180).convert('L').save(img_path270)

Parallel(n_jobs = 6)(delayed(augment)(idx) for idx in pos_ind)



# Augment X set ----------------------
# 1 pass -----------------------------
# take positives
temp_df = X_train[y_train == 1]
# reindex positives
temp_df = temp_df.set_index(temp_df.index + 1000000)
# add new positives
X_train_aug = X_train.append(temp_df)


# Augment y set ----------------------
# 1 pass -----------------------------
# take positives
temp_df = X_train[y_train == 1]
# reindex positives
temp_df = y_train.reindex(temp_df.index + 1000000)
# set all series values to 1
temp_df.loc[:] = 1
# add new positives
y_train_aug = y_train.append(temp_df)


X_train_aug.to_pickle('./nodules/pickle/traindata')
y_train_aug.to_pickle('./nodules/pickle/trainlabels')





# Augment X set ----------------------
# 1 pass -----------------------------
# take positives
temp_df = X_train[y_train == 1]
# reindex positives
temp_df = temp_df.set_index(temp_df.index + 2000000)
# add new positives
X_train_aug2 = X_train_aug.append(temp_df)


# Augment y set ----------------------
# 1 pass -----------------------------
# take positives
temp_df = X_train[y_train == 1]
# reindex positives
temp_df = y_train.reindex(temp_df.index + 2000000)
# set all series values to 1
temp_df.loc[:] = 1
# add new positives
y_train_aug2 = y_train_aug.append(temp_df)


X_train_aug2.to_pickle('./nodules/pickle/traindata')
y_train_aug2.to_pickle('./nodules/pickle/trainlabels')





# Augment X set ----------------------
# 1 pass -----------------------------
# take positives
temp_df = X_train[y_train == 1]
# reindex positives
temp_df = temp_df.set_index(temp_df.index + 3000000)
# add new positives
X_train_aug3 = X_train_aug2.append(temp_df)


# Augment y set ----------------------
# 1 pass -----------------------------
# take positives
temp_df = X_train[y_train == 1]
# reindex positives
temp_df = y_train.reindex(temp_df.index + 3000000)
# set all series values to 1
temp_df.loc[:] = 1
# add new positives
y_train_aug3 = y_train_aug2.append(temp_df)


X_train_aug3.to_pickle('./nodules/pickle/traindata')
y_train_aug3.to_pickle('./nodules/pickle/trainlabels')
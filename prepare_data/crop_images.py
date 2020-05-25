"""
Creates .jpg image sets for train, test or val
Run $ python crop_images.py <target>
<target> = 'train', 'test', 'val'
"""
import sys
import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import SimpleITK as sitk 

LUNA_PATH = '../dataset/'
CAND_PATH = LUNA_PATH + 'CSVFILES/candidates.csv'
ANNT_PATH = LUNA_PATH + 'CSVFILES/annotations.csv'
PICK_PATH = './pickle/'


def do_split():
    """
    Splits data to train, test and val sets
    """
    cand_df = pd.read_csv(CAND_PATH)

    # Get indexes of positives and negatives
    pos_ids = cand_df[cand_df['class'] == 1].index # 1351
    neg_ids = cand_df[cand_df['class'] == 0].index # 549714
    # Take negatives only ten times the size of positives (13510)
    neg_ids = np.random.choice(neg_ids, len(pos_ids) * 10, replace = False)
    # Create new df only with new negatives and positives
    cand_df = cand_df.iloc[list(pos_ids) + list(neg_ids)]

    # Take all columns for X except last. Last column is y ('class')
    X = cand_df.iloc[:,:-1]
    y = cand_df.iloc[:,-1]

    # Split to train and test, then split train to train and val
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

    X_train.to_pickle(PICK_PATH + 'traindata')
    y_train.to_pickle(PICK_PATH + 'trainlabels')
    X_test.to_pickle(PICK_PATH + 'testdata')
    y_test.to_pickle(PICK_PATH + 'testlabels')
    X_val.to_pickle(PICK_PATH + 'valdata')
    y_val.to_pickle(PICK_PATH + 'vallabels')


def main():
    if len(sys.argv) < 2:
        raise ValueError('<target> argument not found. Specify "train", "test" or "val"')
    else:
        target = sys.argv[1]
        if target not in ['train', 'test', 'val']:
            raise ValueError('Invalid <target> argument. Specify "train", "test" or "val"')

    PICKLE_FILE = PICK_PATH + target + 'data'
    OUTPUT_IMG_NAME = target + '/nodule_'

    if not os.path.isfile(PICKLE_FILE):
        do_split()

    X = pd.read_pickle(PICKLE_FILE)
    

if __name__ == "__main__":
    main()
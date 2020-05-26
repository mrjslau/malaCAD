"""
Creates .jpg image sets for train, test or val
Run $ python crop_images.py <target>
<target> = 'train', 'test', 'val'
"""
import sys
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import SimpleITK as sitk 
from joblib import Parallel, delayed

LUNA_PATH = '../dataset/'
MHD_PATH = LUNA_PATH + '*/'
CAND_PATH = LUNA_PATH + 'CSVFILES/candidates.csv'
PICK_PATH = './pickle/'

class CT(object):
    def __init__(self, filename = None, coordinates = None):
        self.filename = filename
        self.coordinates = coordinates
        self.itk_img = None
        self.img_arr = None
        
    # Read .mhd/.raw with SimpleITK
    def read_mhd(self):
        path = glob(MHD_PATH + self.filename + '.mhd')
        print(path)
        self.itk_img = sitk.ReadImage(path[0])
        self.img_arr = sitk.GetArrayFromImage(self.itk_img)
    
    # Get voxel coordinates
    def get_voxel(self):
        origin = self.get_origin()
        spacing = self.get_spacing()
        coordinates = self.get_coordinates()
        return tuple([np.absolute(coordinates[i] - origin[i]) / spacing[i] for i in range(3)])
    
    def get_coordinates(self):
        return self.coordinates
    
    def get_origin(self):
        return self.itk_img.GetOrigin()
    
    def get_spacing(self):
        return self.itk_img.GetSpacing()

    def get_subimage(self, dim):
        """
        Returns cropped image
        """
        self.read_mhd()
        x, y, z = self.get_voxel()
        dim_h = dim / 2
        nodule = self.img_arr[int(z), int(y - dim_h) : int(y + dim_h), int(x - dim_h) : int(x + dim_h)]
        return nodule

    def normalize_planes(self, npzarray):
        """
        Converts Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray

    def save_nodule(self, filename, dim):
        """
        Saves cropped nodule from a CT scan to a file
        """
        nodule = self.get_subimage(dim)
        nodule = self.normalize_planes(nodule)
        Image.fromarray(nodule * 255).convert('L').save(filename)


def create_cropped_imgs(idx, out, X_df, dim = 50):
    """
    Creates CT object and saves cropped nodules
    """
    print(idx)
    outfile = out + str(idx) + '.jpg'

    if (glob(outfile)):
        pass
    else:
        scan = CT(np.asarray(X_df.loc[idx])[0], np.asarray(X_df.loc[idx])[1:])
    
        scan.save_nodule(outfile, dim)

def do_split():
    """
    Splits data to train, test and val sets
    """
    cand_df = pd.read_csv(CAND_PATH)

    # Get indexes of positives and negatives
    pos_ids = cand_df[cand_df['class'] == 1].index # 1351 - 1350
    neg_ids = cand_df[cand_df['class'] == 0].index # 549714 - 548960
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
    OUTPUT_IMG_NAME = './nodules/' + target + '/nodule_'

    if not os.path.isfile(PICKLE_FILE):
        do_split()

    X = pd.read_pickle(PICKLE_FILE)
    Parallel(n_jobs = 6)(delayed(create_cropped_imgs)(idx, OUTPUT_IMG_NAME, X) for idx in X.index)
    

if __name__ == "__main__":
    main()
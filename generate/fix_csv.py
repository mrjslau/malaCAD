"""
Fix candidates.csv file if LUNA
set is incomplete
"""

import numpy as np
import pandas as pd
import os
import glob

LUNA_PATH = '../dataset/'
MHD_PATH = LUNA_PATH + '*/'
CAND_PATH = LUNA_PATH + 'CSVFILES/candidates.csv'

path = glob.glob(MHD_PATH + self.filename + '.mhd')

def __main__():
    cand_df = pd.read_csv(CAND_PATH)

    for s in list(cand_df['seriesuid']):
        if (glob.glob(MHD_PATH + s + '.mhd') == []):
                cand_df = cand_df[cand_df['seriesuid'] != s]

    cand_df.to_csv('new.csv', index=False)

if __name__ == "__main__":
    main()
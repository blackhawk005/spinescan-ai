import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

TRAIN_IMAGES_PATH = '/Volumes/SSD/rsna-2022-cervical-spine-fracture-detection/train_images'
TEST_IMAGES_PATH = '/Volumes/SSD/rsna-2022-cervical-spine-fracture-detection/test_images'
METADATA_PATH = '/Volumes/SSD/rsna-2022-cervical-spine-fracture-detection/metadata'

import pandas as pd
df_seg = pd.read_csv(f'{METADATA_PATH}/meta_segmentation.csv')

slice_max_seg = df_seg.groupby('StudyInstanceUID')['Slice'].max().to_dict()
df_seg['SliceRatio'] = 0
df_seg['SliceRatio'] = df_seg['Slice'] / df_seg['StudyInstanceUID'].map(slice_max_seg)

targets = ['C1','C2','C3','C4','C5','C6','C7']
print(len(df_seg))


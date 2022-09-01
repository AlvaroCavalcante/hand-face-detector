"""
This script is used to iterate over a random sampled dataset and correct it's sample strategy to adopt 
a signer independent setup, following the name of the interpreters in each set.
"""

import glob
import shutil

import pandas as pd
from tqdm import tqdm

base_path = '/home/alvaro/Downloads'

base_save_path = '/home/alvaro/Downloads/new_object_detection'

folders = ['validation-003', 'train-002', 'test-001']

sub_folders = {
    'test-001': 'test',
    'train-002': 'train',
    'validation-003': 'validation'
}

interpreter_labels = [
    'labels_test', 'labels_train', 'labels_validation']


for label in interpreter_labels:
    print('Processing label {}'.format(label))

    df = pd.read_csv(f'{base_save_path}/{label}.csv',
                     names=['video_name', 'label'])

    interpreters = list(df['video_name'].values)

    for signer in tqdm(interpreters):
        for folder in folders:
            fold = sub_folders[folder]

            path = f'{base_path}/{folder}/{fold}/{signer}_*'
            images = glob.glob(path)
            
            for img in images:
                signer_frame_id = img.split('/')[-1]
                shutil.copy(img,
                    f'{base_save_path}/{label.split("_")[-1]}/{signer_frame_id}')

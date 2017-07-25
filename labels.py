import os
import pandas as pd
import sklearn.model_selection

import paths


def get_labels(test_size=0.2, random_state=32):
    tmp_df = pd.read_csv(paths.train_csv)
    assert tmp_df['image_name'].apply(lambda x: os.path.isfile(paths.train_jpg + x + '.jpg')).all(), \
        "Some images referenced in the CSV file were not found"

    if test_size != 0:
        train, val = sklearn.model_selection.train_test_split(tmp_df, test_size=test_size, random_state=random_state)

    else:
        print('Skipping validation split!')
        _, val = sklearn.model_selection.train_test_split(tmp_df, test_size=0.2, random_state=random_state)
        train = tmp_df

    return train, val


def get_labels_df():
    tmp_df = pd.read_csv(paths.train_csv)
    assert tmp_df['image_name'].apply(lambda x: os.path.isfile(paths.train_jpg + x + '.jpg')).all(), \
        "Some images referenced in the CSV file were not found"

    return tmp_df


def get_labels_stratified(test_size=0.2, random_state=32):
    tmp_df = pd.read_csv(paths.train_csv)
    assert tmp_df['image_name'].apply(lambda x: os.path.isfile(paths.train_jpg + x + '.jpg')).all(), \
        "Some images referenced in the CSV file were not found"

    if test_size != 0:
        train, val = sklearn.model_selection.train_test_split(tmp_df, test_size=test_size, random_state=random_state)

    else:
        print('Skipping validation split!')
        _, val = sklearn.model_selection.train_test_split(tmp_df, test_size=0.2, random_state=random_state)
        train = tmp_df

    return train, val
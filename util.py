import pandas as pd

import paths


def find_epoch_val(model_name):
    print(model_name)
    log = pd.read_csv(paths.logs + model_name)

    best_epoch = 0
    best_val_loss = float('inf')
    for row in log.iterrows():
        index, cols = row
        epoch = cols['epoch']
        loss = cols['loss']
        val_loss = cols['val_loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

    return best_epoch
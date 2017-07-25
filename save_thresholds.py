import numpy as np
import sklearn.model_selection
import multiprocessing
from itertools import repeat, chain

import paths
import labels
from datasets import mlb
from find_best_threshold import optimise_f2_thresholds_fast

labels_df = labels.get_labels_df()
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=1)
split = list(kf.split(labels_df))

models = [
    'nn_semisupervised_densenet_121',
]


def do(model_fold):
    model, i = model_fold
    net = np.load(paths.predictions + '{}-split_{}.npz'.format(model, i))
    train_idx, val_idx = split[i]
    train_predictions = net['train']
    train_true = mlb.transform(labels_df.ix[train_idx]['tags'].str.split()).astype(np.float32)

    thresholds = []
    for train in train_predictions:
        threshold = optimise_f2_thresholds_fast(train_true, train, verbose=True)
        thresholds.append(threshold)

    thresholds = np.stack(thresholds, axis=1)
    np.save(paths.thresholds + '{}-split_{}'.format(model, i), thresholds)
    print('Saved {}-split_{}'.format(model, i))


def flatmap(f, items):
    return chain.from_iterable(map(f, items))

models = flatmap(lambda m: repeat(m, 5), models)


tbd = zip(models, range(5))

p = multiprocessing.Pool(10)
for i in enumerate(p.imap(do, tbd)):
    print(i)
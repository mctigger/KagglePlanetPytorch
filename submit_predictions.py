import os
import numpy as np

from scipy.stats.mstats import gmean
import sklearn.model_selection

import paths
import labels
from datasets import mlb
import find_best_threshold

np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)


def submit_cv_ensemble(ensemble, output_file):
    thresholds = []
    tests = []
    for net, val, train in ensemble:
        y_val = mlb.transform(train['tags'].str.split()).astype(np.float32)
        threshold = find_best_threshold.optimise_f2_thresholds_fast(y_val, net['train'])
        thresholds.append(threshold)
        test = net['test']
        tests.append(test)

    threshold_avg = np.average(np.stack(thresholds, axis=0), axis=0)
    test_avg = np.average(np.stack(tests, axis=0), axis=0)

    test_images = list(map(lambda path: path[:-len('.jpg')], os.listdir(paths.test_jpg)))
    test_avg[test_avg > threshold_avg] = 1
    test_avg[test_avg <= threshold_avg] = 0

    predictions = mlb.inverse_transform(test_avg)
    test_results = zip(predictions, test_images)

    with open(paths.submissions + output_file, 'w') as submission:
        submission.write('image_name,tags\n')
        for tags, target in test_results:
            output = target + ',' + ' '.join(tags)
            submission.write("%s\n" % output)

        print('Submission ready!')


def load_cv_folds(model_name):
    models = []
    for i in range(5):
        net = np.load(paths.predictions + model_name + '-split_{}.npz'.format(i))
        net = {
            'train': np.average(net['train'], axis=0),
            'val': np.average(net['val'], axis=0),
            'test': np.average(net['test'], axis=0)
        }
        models.append(net)

    labels_df = labels.get_labels_df()
    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=1)
    split = kf.split(labels_df)
    folds = []
    for i, ((train_idx, val_idx), net) in enumerate(zip(split, models)):
        val = labels_df.ix[val_idx]
        train = labels_df.ix[train_idx]
        folds.append((net, val, train))
        print(i)

    return folds

# load_cv_folds takes the model name
folds = load_cv_folds('nn_finetune_resnet_50')
# you can chose a name yourself here, but I name my ensembled model_fold_1+2+3+4+5
submit_cv_ensemble(folds, 'nn_finetune_resnet_50_fold_1+2+3+4+5')
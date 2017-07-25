import os
import sys

import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import paths
import labels
from datasets import mlb
from find_best_threshold import fbeta, optimise_f2_thresholds_fast, optimise_f2_thresholds

models = [
    'nn_semisupervised_densenet_121',
]

for model_name in models:
    try:
        random_state = 1
        labels_df = labels.get_labels_df()
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        split = list(kf.split(labels_df))


        x_train_all = []
        x_val_all = []
        x_test_all = []
        labels_train_all = []
        labels_val_all = []
        thresholds_all = []

        for i, (labels_train, labels_val) in enumerate(split):
            net = np.load(paths.predictions + '{}-split_{}.npz'.format(model_name, i))
            thresholds = np.load(paths.thresholds + '{}-split_{}.npy'.format(model_name, i))

            thresholds_all.append(np.average(thresholds, axis=1))
            train = net['train']
            val = net['val']
            test = net['test']

            x_train_all.append(train)
            x_val_all.append(val)
            x_test_all.append(test)

            labels_train = mlb.transform(labels_df.ix[labels_train]['tags'].str.split()).astype(np.float32)
            labels_val = mlb.transform(labels_df.ix[labels_val]['tags'].str.split()).astype(np.float32)

            labels_train_all.append(labels_train)
            labels_val_all.append(labels_val)

        train = x_train_all = np.concatenate(x_train_all, axis=1)
        val = x_val_all = np.concatenate(x_val_all, axis=1)
        # Test gets stacked over folds instead of concat
        test = x_test_all = np.stack(x_test_all, axis=0)

        # Thresholds get averaged as comparison
        thresholds_average = np.average(np.stack(thresholds_all, axis=0), axis=0)

        # Labels get concatenated for right order
        labels_train = labels_train_all = np.concatenate(labels_train_all, axis=0)
        labels_val = labels_val_all = np.concatenate(labels_val_all, axis=0)


        p_reference_average = []
        p_val = []
        p_val_hard = []
        p_test = []
        for i in range(17):
            x_train = train[:, :, i]
            x_train = np.rollaxis(x_train, 1)
            y_train = labels_train[:, i]

            x_val = val[:, :, i]
            x_val = np.rollaxis(x_val, 1)
            y_val = labels_val[:, i]

            x_test = test[:, :, :, i]
            x_test = np.rollaxis(x_test, 2, 1)

            x_val_avg = np.average(np.copy(x_val), axis=1)
            x_val_avg[x_val_avg >= thresholds_average[i]] = 1
            x_val_avg[x_val_avg < thresholds_average[i]] = 0

            n_estimators = 1000
            max_evals = 15

            def objective(space):
                estimator = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=int(space['max_depth']),
                    min_child_weight=int(space['min_child_weight']),
                    gamma=space['gamma'],
                    subsample=space['subsample'],
                    colsample_bytree=space['colsample_bytree']
                )

                estimator.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_train, y_train), (x_val, y_val)],
                    early_stopping_rounds=30,
                    verbose=False,
                    eval_metric='error'
                )

                score = accuracy_score(y_val, estimator.predict(x_val))

                return {'loss': 1 - score, 'status': STATUS_OK}

            space = {
                'max_depth': hp.quniform("max_depth", 3, 20, 1),
                'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                'subsample': hp.uniform('subsample', 0.7, 0.9),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.9),
                'gamma': hp.choice('gamma', [0, 0.01, 0.1, 1])
            }


            trials = Trials()
            best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )

            # Fit best estimator
            estimator = XGBClassifier(
                n_estimators=n_estimators*5,
                max_depth=int(best['max_depth']),
                min_child_weight=int(best['min_child_weight']),
                gamma=best['gamma'],
                subsample=best['subsample'],
                colsample_bytree=best['colsample_bytree'],
            )

            estimator.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
                eval_metric='error'
            )

            print('Feature', i)
            print('XGBoost', accuracy_score(y_val, estimator.predict(x_val)))
            print('Average', accuracy_score(y_val, x_val_avg))
            p_val.append(estimator.predict_proba(x_val)[:, 1])
            p_val_hard.append(estimator.predict(x_val))
            p_reference_average.append(x_val_avg)

            fold_test = []
            for f in range(5):
                fold_test.append(estimator.predict_proba(x_test[f, :, :])[:, 1])
            fold_test = np.average(np.stack(fold_test, axis=0), axis=0)
            p_test.append(fold_test)


        # Stack predictions for each of the 17 targets
        # Predictions by XGBoost
        p_val = np.stack(p_val, axis=1)
        p_test = np.stack(p_test, axis=1)
        # Predictions by XGBoost which will not be optimised for f2
        p_val_hard = np.stack(p_val_hard, axis=1)
        # Predictions from averaging
        p_reference_average = np.stack(p_reference_average, axis=1)

        print(p_val.shape)
        print(p_val_hard.shape)
        print(p_reference_average.shape)


        print('=======================')
        print('fbeta(labels_val, p_reference_average)', fbeta(labels_val, p_reference_average))
        print('fbeta(labels_val, p_val_hard)', fbeta(labels_val, p_val_hard))

        threshold = optimise_f2_thresholds(labels_val, p_val)
        p_val[p_val >= threshold] = 1
        p_val[p_val < threshold] = 0

        print('fbeta(labels_val, p_val)', fbeta(labels_val, p_val))

        p_test[p_test >= threshold] = 1
        p_test[p_test < threshold] = 0


        # WRITE PREDICTIONS
        test_images = list(map(lambda path: path[:-len('.jpg')], os.listdir(paths.test_jpg)))
        predictions = mlb.inverse_transform(p_test)
        test_results = zip(predictions, test_images)

        with open(paths.submissions + '{}_XGB_HYPEROPT'.format(model_name), 'w') as submission:
            submission.write('image_name,tags\n')
            for tags, target in test_results:
                output = target + ',' + ' '.join(tags)
                submission.write("%s\n" % output)

            print('Submission ready!')

    except:
        print("Unexpected error:", sys.exc_info()[0])
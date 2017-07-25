import json

import pandas as pd
import numpy as np

import os

import paths
from datasets import mlb, KaggleAmazonTestDataset

# (submission, weight)
submissions = [
    ('nn_semisupervised_resnet_50_XGB_HYPEROPT', 1),
    ('nn_semisupervised_resnet_50_random_v1_XGB_HYPEROPT', 1),
    ('nn_semisupervised_resnet_101_XGB_HYPEROPT', 1),
]

submission_name = 'WEIGHTED_VOTING_SUBMISSION'


def voting_submission():
    submissions_raw = []
    weight_sum = 0
    for (submission, weight) in submissions:
        df = pd.read_csv(paths.submissions + submission)
        df = df.sort_values('image_name')
        print(df[:3])
        raw = mlb.transform(df['tags'].str.split()).astype(np.float32)
        submissions_raw.append(raw*weight)
        weight_sum += weight

    stacked_submissions = np.stack(submissions_raw, 0)
    avg_submissions = np.sum(stacked_submissions, 0) / weight_sum

    avg_submissions[avg_submissions >= 0.5] = 1
    avg_submissions[avg_submissions < 0.5] = 0

    tags = mlb.inverse_transform(avg_submissions)

    test_images = sorted(list(map(lambda path: path[:-len('.jpg')], os.listdir(paths.test_jpg))))

    print(test_images[:3])

    with open(paths.ensemble_weights + submission_name, 'w') as fp:
        json.dump(submissions, fp, sort_keys=True, indent=4)

    with open(paths.submissions + submission_name, 'w') as submission:
        submission.write('image_name,tags\n')
        for labels, target in zip(tags, test_images):
            output = target + ',' + ' '.join(labels)
            submission.write("%s\n" % output)

        print('Submission ready!')

voting_submission()
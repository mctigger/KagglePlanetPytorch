import sys
import os
import itertools
import torch
import numpy as np
import pandas as pd
from pydoc import locate

from torch.autograd import Variable
import torchvision.transforms
from torch.utils.data import DataLoader

import sklearn.model_selection

import paths
import transforms
import labels
import util
from datasets import KaggleAmazonJPGDataset, KaggleAmazonTestDataset, mlb

batch_size = 128


def predict(net, loaders):
    net.cuda()
    net.eval()

    def apply_transform(loader):
        predictions_acc = []
        for batch_idx, (data, targets) in enumerate(loader):
            data = Variable(data.cuda(async=True), volatile=True)
            output = net(data)
            predictions = output.cpu().data.numpy()

            for prediction, target in zip(predictions, targets):
                predictions_acc.append(prediction)

            print('[{}/{} ({:.0f}%)]'.format(batch_idx * len(data), len(loader.dataset),
                                             100. * batch_idx / len(loader)), end='\r')
            sys.stdout.flush()

        predictions_acc = np.array(predictions_acc)

        return predictions_acc

    results = []

    for i, l in enumerate(loaders):
        print('TTA {}'.format(i))
        r = apply_transform(l)
        results.append(r)

    return np.stack(results, axis=0)


def get_loader(df, transformations):
    dset_val = KaggleAmazonJPGDataset(df, paths.train_jpg, transformations, divide=False)

    loader_val = DataLoader(dset_val,
                            batch_size=batch_size,
                            num_workers=12,
                            pin_memory=True,)
    return loader_val


def get_test_loader(test_images, transformations):
    dset_test = KaggleAmazonTestDataset(test_images, paths.test_jpg, '.jpg', transformations, divide=False)
    loader_val = DataLoader(dset_test,
                            batch_size=batch_size,
                            num_workers=12,
                            pin_memory=True)
    return loader_val


def predict_model(model, state, train, val, output_file, pre_transforms):
    model.load_state_dict(state)

    transformations = [
        [transforms.rotate_90(0)],
        [transforms.rotate_90(1)],
        [transforms.rotate_90(2)],
        [transforms.rotate_90(3)],
        [transforms.fliplr()],
        [transforms.fliplr()],
        [transforms.augment_deterministic(scale_factor=1/1.1)],
        [transforms.augment_deterministic(scale_factor=1/1.2)],
        [transforms.augment_deterministic(scale_factor=1.1)],
        [transforms.augment_deterministic(scale_factor=1.2)],
        [transforms.augment_deterministic(translation=(20, 20))],
        [transforms.augment_deterministic(translation=(-20, 20))],
        [transforms.augment_deterministic(translation=(20, -20))],
        [transforms.augment_deterministic(translation=(-20, -20))],
    ]

    transformations = list(map(lambda t: transforms.apply_chain(
        pre_transforms + t + [
            torchvision.transforms.ToTensor()
        ]
    ), transformations))

    train_loaders = map(lambda t: get_loader(train, t), transformations)
    val_loaders = map(lambda t: get_loader(val, t), transformations)

    test_images = list(map(lambda path: path[:-len('.jpg')], os.listdir(paths.test_jpg)))
    test_loaders = map(lambda t: get_test_loader(test_images, t), transformations)

    print('Predicting val...')
    val_predictions = predict(model, val_loaders)
    print(val_predictions.shape)

    print('Predicting train...')
    train_predictions = predict(model, train_loaders)
    print(train_predictions.shape)

    print('Predicting test...')
    test_predictions = predict(model, test_loaders)
    print(test_predictions.shape)


    # Save train, val and test
    np.savez(
        paths.predictions + output_file,
        train=train_predictions,
        val=val_predictions,
        test=test_predictions,
    )


def predict_kfold(model_name, pre_transforms=[]):
    model = locate(model_name + '.generate_model')()
    random_state = locate(model_name + '.random_state')
    print('Random state: {}'.format(random_state))

    labels_df = labels.get_labels_df()
    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=random_state)
    split = kf.split(labels_df)

    for i, (train_idx, val_idx) in enumerate(split):
        split_name = model_name + '-split_' + str(i)
        best_epoch = util.find_epoch_val(split_name)
        print('Using epoch {} for predictions'.format(best_epoch))
        epoch_name = split_name + '-epoch_' + str(best_epoch)
        train = labels_df.ix[train_idx]
        val = labels_df.ix[val_idx]
        state = torch.load(os.path.join(paths.models, split_name, epoch_name))

        predict_model(model, state, train, val, output_file=split_name, pre_transforms=pre_transforms)

if __name__ == "__main__":
    # Use the model name without -fold_x or -epoch_k here. Will automatically use the best epoch of every fold!
    predict_kfold('nn_semisupervised_densenet_121')


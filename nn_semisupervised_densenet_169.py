import os
import sys

from itertools import chain

import numpy as np
import pandas as pd

import torchvision.models
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from torchsample.callbacks import CSVLogger, LearningRateScheduler
from callbacks import ModelCheckpoint, SemiSupervisedUpdater

import sklearn.model_selection

import paths
import labels
import transforms
from datasets import KaggleAmazonUnsupervisedDataset, KaggleAmazonSemiSupervisedDataset, KaggleAmazonJPGDataset, mlb
from ModuleTrainer import ModuleTrainer

name = os.path.basename(sys.argv[0])[:-3]


def generate_model():
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.classifier = nn.Linear(pretrained_model.classifier.in_features, 17)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

            self.features = pretrained_model.features
            self.layer1 = pretrained_model.features._modules['denseblock1']
            self.layer2 = pretrained_model.features._modules['denseblock2']
            self.layer3 = pretrained_model.features._modules['denseblock3']
            self.layer4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.sigmoid(self.classifier(out))
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))


random_state = 1
labels_df = labels.get_labels_df()
unsupervised_dataframe = pd.read_csv(paths.submissions + 'SOTA')

kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=random_state)
kf_unsupervised = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=879)
split = supervised_split = kf.split(labels_df)
unsupervised_split = kf_unsupervised.split(unsupervised_dataframe)


def train_net(train, val, unsupervised, model, name):
    unsupervised_initialization = mlb.transform(unsupervised['tags'].str.split()).astype(np.float32)
    unsupervised_samples = unsupervised['image_name'].as_matrix()

    unsupervised_initialization = unsupervised_initialization[:len(unsupervised_initialization)//2*3]
    unsupervised_samples = unsupervised_samples[:len(unsupervised_samples)//2*3]

    transformations_train = transforms.apply_chain([
        transforms.random_fliplr(),
        transforms.random_flipud(),
        transforms.augment(),
        torchvision.transforms.ToTensor()
    ])

    transformations_val = transforms.apply_chain([
        torchvision.transforms.ToTensor()
    ])

    dset_train_unsupervised = KaggleAmazonUnsupervisedDataset(
        unsupervised_samples,
        paths.test_jpg,
        '.jpg',
        transformations_train,
        transformations_val,
        unsupervised_initialization
    )

    dset_train_supervised = KaggleAmazonJPGDataset(train, paths.train_jpg, transformations_train, divide=False)
    dset_train = KaggleAmazonSemiSupervisedDataset(dset_train_supervised, dset_train_unsupervised, None, indices=False)

    train_loader = DataLoader(dset_train,
                              batch_size=32,
                              shuffle=True,
                              num_workers=10,
                              pin_memory=True)

    dset_val = KaggleAmazonJPGDataset(val, paths.train_jpg, transformations_val, divide=False)
    val_loader = DataLoader(dset_val,
                            batch_size=32,
                            num_workers=10,
                            pin_memory=True)

    ignored_params = list(map(id, chain(
        model.classifier.parameters(),
        model.layer1.parameters(),
        model.layer2.parameters(),
        model.layer3.parameters(),
        model.layer4.parameters()
    )))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.layer1.parameters()},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0, weight_decay=0.0001)

    trainer = ModuleTrainer(model)

    def schedule(current_epoch, current_lrs, **logs):
        lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
        epochs = [0, 1, 6, 8, 12]

        for lr, epoch in zip(lrs, epochs):
            if current_epoch >= epoch:
                current_lrs[5] = lr
                if current_epoch >= 2:
                    current_lrs[4] = lr * 1
                    current_lrs[3] = lr * 1
                    current_lrs[2] = lr * 1
                    current_lrs[1] = lr * 1
                    current_lrs[0] = lr * 0.1

        return current_lrs

    trainer.set_callbacks([
        ModelCheckpoint(
            paths.models,
            name,
            save_best_only=False,
            saving_strategy=lambda epoch: True
        ),
        CSVLogger(paths.logs + name),
        LearningRateScheduler(schedule),
        SemiSupervisedUpdater(trainer, dset_train_unsupervised, start_epoch=6, momentum=0.25)
    ])

    trainer.compile(loss=nn.BCELoss(),
                    optimizer=optimizer)

    trainer.fit_loader(train_loader,
                       val_loader,
                       nb_epoch=16,
                       verbose=1,
                       cuda_device=0)

if __name__ == "__main__":
    for i, ((train_idx, val_idx), (train_idx_unsupervised, val_idx_unsupervised)) in enumerate(zip(supervised_split, unsupervised_split)):
        name = os.path.basename(sys.argv[0])[:-3] + '-split_' + str(i)
        train_net(
            labels_df.ix[train_idx],
            labels_df.ix[val_idx],
            unsupervised_dataframe.ix[train_idx_unsupervised][::4],
            generate_model(),
            name
        )

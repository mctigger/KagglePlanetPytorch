import os
import sys

from itertools import chain


import torchvision.models
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch.nn.init
from torch.utils.data import DataLoader

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import CSVLogger, LearningRateScheduler

import sklearn.model_selection

import paths
import labels
import transforms
import callbacks
from datasets import KaggleAmazonJPGDataset

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
            self.dense1 = pretrained_model.features._modules['denseblock1']
            self.dense2 = pretrained_model.features._modules['denseblock2']
            self.dense3 = pretrained_model.features._modules['denseblock3']
            self.dense4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.sigmoid(self.classifier(out))
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))


random_state = 1
labels_df = labels.get_labels_df()
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=random_state)
split = kf.split(labels_df)


def train_net(train, val, model, name):
    transformations_train = transforms.apply_chain([
        transforms.to_float,
        transforms.random_fliplr(),
        transforms.random_flipud(),
        transforms.augment_color(0.1),
        transforms.augment(),
        torchvision.transforms.ToTensor()
    ])

    transformations_val = transforms.apply_chain([
        transforms.to_float,
        torchvision.transforms.ToTensor()
    ])

    dset_train = KaggleAmazonJPGDataset(train, paths.train_jpg, transformations_train, divide=False)
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
        model.dense1.parameters(),
        model.dense2.parameters(),
        model.dense3.parameters(),
        model.dense4.parameters()
    )))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': model.dense1.parameters()},
        {'params': model.dense2.parameters()},
        {'params': model.dense3.parameters()},
        {'params': model.dense4.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0, weight_decay=0.0005)

    trainer = ModuleTrainer(model)

    # New settings
    def schedule(current_epoch, current_lrs, **logs):
        lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
        epochs = [0, 4, 15, 20, 25]

        for lr, epoch in zip(lrs, epochs):
            if current_epoch >= epoch:
                current_lrs[5] = lr
                if current_epoch >= 2:
                    current_lrs[4] = lr * 1
                    current_lrs[3] = lr * 1
                    current_lrs[2] = lr * 0.5
                    current_lrs[1] = lr * 0.2
                    current_lrs[0] = lr * 0.1

        return current_lrs


    trainer.set_callbacks([
        callbacks.ModelCheckpoint(
            paths.models,
            name,
            save_best_only=False,
            saving_strategy=lambda epoch: True
        ),
        CSVLogger('./logs/' + name),
        LearningRateScheduler(schedule)
    ])

    trainer.compile(loss=nn.BCELoss(),
                    optimizer=optimizer)

    trainer.fit_loader(train_loader,
                       val_loader,
                       nb_epoch=30,
                       verbose=1,
                       cuda_device=0)


if __name__ == "__main__":
    for i, (train_idx, val_idx) in enumerate(split):
        name = os.path.basename(sys.argv[0])[:-3] + '-split_' + str(i)
        train_net(labels_df.ix[train_idx], labels_df.ix[val_idx], generate_model(), name)
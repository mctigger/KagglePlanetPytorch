import os
import torch
from torch import np
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import skimage.io

tags = [
    'blooming',
    'selective_logging',
    'blow_down',
    'conventional_mine',
    'bare_ground',
    'artisinal_mine',
    'primary',
    'agriculture',
    'water',
    'habitation',
    'road',
    'cultivation',
    'slash_burn'
]

tags_weather = [
    'cloudy',
    'partly_cloudy',
    'haze',
    'clear'
]
mlb = MultiLabelBinarizer()
mlb = mlb.fit([tags, tags_weather])


class KaggleAmazonDataset(Dataset):
    def __init__(self, dataframe, img_path, transform):
        self.img_path = img_path
        self.transform = transform

        self.X_train = dataframe['image_name'].as_matrix()
        self.y_train = mlb.transform(dataframe['tags'].str.split()).astype(np.float32)

    def __len__(self):
        return len(self.X_train)


class KaggleAmazonJPGDataset(KaggleAmazonDataset):
    def __init__(self, dataframe, img_path, transform, divide=True):
        super(KaggleAmazonJPGDataset, self).__init__(dataframe, img_path, transform)

        self.divide = divide

    def __getitem__(self, index):
        img = skimage.io.imread(self.img_path + self.X_train[index] + '.jpg')

        if self.divide:
            img = img / 255

        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label


class KaggleAmazonTestDataset(Dataset):

    def __init__(self, test_images, img_path, img_ext, transform, divide=True):
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform
        self.test_images = test_images
        self.divide = divide

    def __getitem__(self, index):
        img = skimage.io.imread(self.img_path + self.test_images[index] + self.img_ext)

        if self.divide:
            img = img / 255

        img = self.transform(img)

        return img, self.test_images[index]

    def __len__(self):
        return len(self.test_images)


class KaggleAmazonUnsupervisedDataset(Dataset):
    def __init__(self, paths, img_path, img_ext, transform_train, transform_val, y_train):
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform_train = transform_train
        self.transform = transform_train
        self.transform_val = transform_val
        self.X_train = paths
        self.y_train = y_train

    def __getitem__(self, index):
        img = skimage.io.imread(self.img_path + self.X_train[index] + self.img_ext)

        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])

        return img, label

    def __len__(self):
        return len(self.X_train)


class KaggleAmazonSemiSupervisedDataset(Dataset):
    def __init__(self, supervised, unsupervised, transform, indices=True):
        self.supervised = supervised
        self.unsupervised = unsupervised
        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if index < len(self.supervised):
            x, y = self.supervised[index]
            i = 0

        else:
            x, y = self.unsupervised[index-len(self.supervised)]
            i = 1

        if self.transform:
            x = self.transform(x)

        if self.indices:
            return x, y, i
        else:
            return x, y

    def __len__(self):
        return len(self.supervised) + len(self.unsupervised)


if __name__ == "__main__":
    print(mlb.classes)

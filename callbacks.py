from __future__ import absolute_import
from __future__ import print_function

import os
import time
from torchsample import Callback
import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader


class ModelCheckpoint(Callback):
    """
    Model Checkpoint to save model weights during training
    """

    def __init__(self,
                 directory,
                 name,
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=True,
                 saving_strategy=lambda epoch: True,
                 mode=lambda loss_new, loss_old: loss_new < loss_old,
                 verbose=0):

        self.directory = os.path.join(directory, name)
        self.file = os.path.join(self.directory, name)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.saving_strategy = saving_strategy
        self.verbose = verbose
        self.best_loss = None

        print('')

        os.makedirs(self.directory, exist_ok=True)

        super(ModelCheckpoint, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if not self.saving_strategy(epoch):
            return

        if self.save_best_only:
            loss = logs.get(self.monitor)
            if loss is None:
                pass
            else:
                file = self.file + '-best_epoch_{}'.format(epoch)
                if self.best_loss is None:
                    self.best_loss = loss
                    if self.verbose > 0:
                        print('\nEpoch {}: Saving model for the first time to {}'.format(epoch + 1, file))
                    self.model.save_state_dict(file)

                elif self.mode(self.best_loss, loss):
                    if self.verbose > 0:
                        print('\nEpoch {}: improved from {} to {} saving model to {}'.format(epoch+1, self.best_loss, loss, file))
                    self.best_loss = loss
                    self.model.save_state_dict(file)

        else:
            file = self.file + '-epoch_{}'.format(epoch)
            if self.verbose > 0:
                print('\nEpoch %i: saving model to %s' % (epoch+1, file))
            self.model.save_state_dict(file)


class SemiSupervisedUpdater(Callback):
    def __init__(self, trainer, dataset, start_epoch=0, momentum=0, batch_size=96):
        super(SemiSupervisedUpdater, self).__init__()

        self.trainer = trainer
        self.dataset = dataset
        self.start_epoch = start_epoch
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
        self.momentum = momentum

    def predict_loader(self,
                       loader,
                       cuda_device=-1):
        prediction_list = []
        for batch_idx, batch_data in enumerate(loader):
            if not isinstance(batch_data, (tuple, list)):
                batch_data = [batch_data]
            input_batch = batch_data[0]
            if not isinstance(input_batch, (list, tuple)):
                input_batch = [input_batch]
            input_batch = [Variable(ins, volatile=True) for ins in input_batch]
            if cuda_device > -1:
                input_batch = [ins.cuda(cuda_device) for ins in input_batch]

            prediction_list.append(self.model.model(*input_batch))

        # concatenate all outputs of the same type together (when there are multiple outputs)
        if len(prediction_list) > 0 and isinstance(prediction_list[0], (tuple, list)):
            nb_out = len(prediction_list[0])
            out_list = []
            for out_i in range(nb_out):
                precdiction_out_i = [prediction[out_i] for prediction in prediction_list]
                out_list.append(th.cat(precdiction_out_i, 0))
            return out_list

        return th.cat(prediction_list, 0)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return

        self.dataset.transform = self.dataset.transform_val
        predictions = self.predict_loader(self.loader, cuda_device=0).cpu().data.numpy()

        self.dataset.transform = self.dataset.transform_train
        self.dataset.y_train = (1 - self.momentum) * self.dataset.y_train + predictions * self.momentum


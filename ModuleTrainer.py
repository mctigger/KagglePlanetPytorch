from __future__ import print_function
from __future__ import absolute_import

from torch.autograd import Variable

from torchsample.modules import ModuleTrainer

import math

# local imports
from torchsample.modules._utils import (_get_current_time, _nb_function_args)
from torchsample.callbacks import CallbackModule, History, TQDM
from torchsample.constraints import ConstraintModule
from torchsample.initializers import InitializerModule
from torchsample.metrics import MetricsModule
from torchsample.regularizers import RegularizerModule


class SemisupervisedModuleTrainer(ModuleTrainer):
    def fit_loader(self,
                   loader,
                   val_loader=None,
                   nb_epoch=100,
                   cuda_device=-1,
                   metrics=None,
                   verbose=1,
                   custom_loss=None
                   ):

        # store whether validation data was given
        if val_loader is None:
            has_validation_data = False
        else:
            has_validation_data = True

            # create regularizers
        if hasattr(self.model, 'regularizers'):
            for reg in self.model.regularizers:
                self.add_regularizer(reg)
        if self._has_regularizers:
            regularizers = RegularizerModule(self._regularizers)

        # create constraints
        if hasattr(self.model, 'constraints'):
            for constraint in self.model.constraints:
                self.add_constraint(constraint)
        if self._has_constraints:
            constraints = ConstraintModule(self._constraints)
            constraints.set_model(self.model)

        # create metrics
        if hasattr(self.model, 'metrics'):
            for metric in self.model.metrics:
                self.add_metric(metric)
        if self._has_metrics:
            metrics = MetricsModule(self._metrics)

        # create initializers
        if hasattr(self.model, 'initializers'):
            for initializer in self.model.initializers:
                self.add_initializer(initializer)
        if self._has_initializers:
            initializers = InitializerModule(self._initializers)
            initializers(self.model)

        if cuda_device > -1:
            self.model.cuda(cuda_device)

        # enter context-manager for progress bar
        with TQDM() as pbar:
            # create callbacks
            progressbar = []
            # add progress bar if necessary
            if verbose > 0:
                progressbar = [pbar]
            callbacks = CallbackModule(self._callbacks + progressbar)
            callbacks.set_model(self)

            train_begin_logs = {
                'start_time': _get_current_time(),
                'has_validation_data': has_validation_data
            }
            callbacks.on_train_begin(logs=train_begin_logs)

            # calculate total number of batches
            nb_batches = int(math.ceil(len(loader.dataset) / loader.batch_size))

            # loop through each epoch
            for epoch_idx in range(nb_epoch):
                epoch_logs = {
                    'nb_batches': nb_batches,
                    'nb_epoch': nb_epoch,
                    'has_validation_data': has_validation_data
                }
                callbacks.on_epoch_begin(epoch_idx, epoch_logs)

                for batch_idx, data, in enumerate(loader):
                    batch_logs = {'batch_idx': batch_idx}
                    callbacks.on_batch_begin(batch_idx, batch_logs)
                    data = [Variable(ins) for ins in data]
                    if cuda_device > -1:
                        data = tuple([ins.cuda(cuda_device) for ins in data])

                    batch_logs['batch_samples'] = len(data[0])

                    x, y, i = data
                    # zero grads and forward pass
                    self._optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = custom_loss(outputs, y, i, epoch_idx)

                    batch_logs['loss'] = loss.data[0]

                    # calculate custom/special batch metrics if necessary
                    if self._has_metrics:
                        metric_logs = metrics(data)
                        batch_logs.update(metric_logs)

                    # backward pass and optimizer step
                    loss.backward()
                    self._optimizer.step()

                    callbacks.on_batch_end(batch_idx, batch_logs)

                    # apply explicit constraints if necessary
                    if self._has_constraints:
                        constraints.on_batch_end(batch_idx)

                if has_validation_data:
                    val_loss = self.evaluate_loader(val_loader,
                                                    cuda_device=cuda_device)
                    if self._has_metrics:
                        val_loss, val_metric_logs = val_loss
                        epoch_logs.update(val_metric_logs)
                    epoch_logs['val_loss'] = val_loss
                    self.history.batch_metrics['val_loss'] = val_loss

                # END OF EPOCH
                epoch_logs.update(self.history.batch_metrics)
                if self._has_metrics:
                    epoch_logs.update(metric_logs)

                callbacks.on_epoch_end(epoch_idx, epoch_logs)

                # apply Epoch-level constraints if necessary
                if self._has_constraints:
                    constraints.on_epoch_end(epoch_idx)
                # reset all metric counters
                if self._has_metrics:
                    metrics.reset()
                # exit the training loop if necessary (e.g. EarlyStopping)
                if self._stop_training:
                    break

        train_logs = {
            'final_loss': self.history.losses[-1],
            'best_loss': min(self.history.losses),
            'end_time': _get_current_time()
        }
        if has_validation_data:
            train_logs['final_val_loss'] = self.history.val_losses[-1]
            train_logs['best_val_loss'] = min(self.history.val_losses)

        callbacks.on_train_end(logs=train_logs)

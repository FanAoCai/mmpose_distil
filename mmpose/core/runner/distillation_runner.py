# Copyright (c) Open-MMLab. All rights reserved.
import time
import warnings
import logging

import torch

import mmcv
from mmcv.runner.builder import RUNNERS
from mmcv.runner import EpochBasedRunner
from mmcv.parallel import is_module_wrapper
from mmcv.runner.utils import get_host_info

from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)

import pdb

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

@RUNNERS.register_module()
class DistillationRunner(EpochBasedRunner):
    """Knowledge Distillation Epoch-based Runner.

    This runner train models epoch by epoch, the epoch length is defined by the
    dataloader[0], which is the main dataloader.
    """
    def __init__(self,
                 smodel: torch.nn.Module,
                 tmodel: torch.nn.Module,
                 batch_processor: Optional[Callable] = None,
                 optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 meta: Optional[Dict] = None,
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[int] = None) -> None:
        
        super(DistillationRunner, self).__init__(
                model=smodel,
                batch_processor=batch_processor,
                optimizer=optimizer,
                work_dir=work_dir,
                logger=logger,
                meta=meta,
                max_iters=max_iters,
                max_epochs=max_epochs)
        
        if is_module_wrapper(tmodel):
            self.tmodel = tmodel.module
        else:
            self.tmodel = tmodel
        self.tmodel.eval()
        self.tmodel = self.tmodel.cuda()

    def run_iter(self, data_batch, train_mode, source, **kwargs):
        # time1=time.time()
        toutputs = self.tmodel.forward_test_distill(data_batch['img'], **kwargs)
        # time2=time.time()
        # print("time1:toutput",time2-time1)
        if train_mode:
            # used teacher outputs as extra supervised information
            data_batch = dict(list(data_batch.items()) + list({'teacher_output_heatmap': toutputs['output_heatmap']}.items()))
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
            # time3=time.time()
            # print("time2:output",time3-time2)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
            # time3=time.time()
            # print("time3:output",time3-time2)
        
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        # Since we have multiple sources, we add a suffix to log_var names,
        # so that we can differentiate them.
        if 'log_vars' in outputs:
            log_vars = outputs['log_vars']
            log_vars = {k + source: v for k, v in log_vars.items()}
            self.log_buffer.update(log_vars, outputs['num_samples'])
        
        self.outputs = outputs

    def train(self, data_loaders, **kwargs):
        self.model.train()
        #self.tmodel.eval()

        self.mode = 'train'
        self.data_loaders = data_loaders
        self.main_loader = self.data_loaders[0]
        # Add aliasing
        self.data_loader = self.main_loader
        self.aux_loaders = self.data_loaders[1:]
        self.aux_iters = [cycle(loader) for loader in self.aux_loaders]

        auxiliary_iter_times = [1] * len(self.aux_loaders)
        use_aux_per_niter = 1
        if 'train_ratio' in kwargs:
            train_ratio = kwargs.pop('train_ratio')
            use_aux_per_niter = train_ratio[0]
            auxiliary_iter_times = train_ratio[1:]

        self._max_iters = self._max_epochs * len(self.main_loader)

        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.main_loader):
            
            self._inner_iter = i
            self.call_hook('before_train_iter')
            # before_train_iter_time=time.time()
            self.run_iter(data_batch, train_mode=True, source='')
            # print(data_batch)
            # after_train_iter_time=time.time()
            # print('run_iter_time',after_train_iter_time-before_train_iter_time)
            self.call_hook('after_train_iter')
            # after_call_hook_time=time.time()
            # print('call_hook_time',after_call_hook_time-after_train_iter_time)

            if self._iter % use_aux_per_niter != 0:
                self._iter += 1
                continue
            
            for idx, n_times in enumerate(auxiliary_iter_times):
                extra_start_time=time.time()
                for _ in range(n_times):
                    data_batch = next(self.aux_iters[idx])
                    self.call_hook('before_train_iter')
                    self.run_iter(
                        data_batch, train_mode=True, source=f'/aux{idx}')
                    self.call_hook('after_train_iter')
                extra_end_time=time.time()
                print('extra_time',extra_end_time-extra_start_time)
            self._iter += 1
            
        self.call_hook('after_train_epoch')
        self._epoch += 1

    # Now that we use validate hook, not implement this func to save efforts.
    def val(self, data_loader, **kwargs):
        raise NotImplementedError

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training.
                `data_loaders[0]` is the main data_loader, which contains
                target datasets and determines the epoch length.
                `data_loaders[1:]` are auxiliary data loaders, which contain
                auxiliary web datasets.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2)] means running 2
                epochs for training iteratively. Note that val epoch is not
                supported for this runner for simplicity.
            max_epochs (int | None): The max epochs that training lasts,
                deprecated now. Default: None.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(workflow) == 1 and workflow[0][0] == 'train'
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        mode, epochs = workflow[0]
        self._max_iters = self._max_epochs * len(data_loaders[0])

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            if isinstance(mode, str):  # self.train()
                if not hasattr(self, mode):
                    raise ValueError(
                        f'runner has no method named "{mode}" to run an '
                        'epoch')
                epoch_runner = getattr(self, mode)    #获取mode的值，是train还是test
            else:
                raise TypeError(
                    f'mode in workflow must be a str, but got {mode}')

            for _ in range(epochs):
                if mode == 'train' and self.epoch >= self._max_epochs:
                    break
                epoch_runner(data_loaders, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')


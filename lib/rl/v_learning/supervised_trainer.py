# Derived from https://github.com/vita-epfl/CrowdNav/blob/master/crowd_nav/utils/trainer.py

import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils import AbstractLogger


class SupervisedTrainer:
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self._model = model
        self.device = device
        self._criterion = nn.MSELoss().to(device)
        self._memory = memory
        self._data_loader = None
        self._batch_size = batch_size
        self._optimizer = None

    def set_learning_rate(self, learning_rate, wd=0.):
        logging.info('Current learning rate: %f', learning_rate)
        # self._optimizer = optim.SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate,
                                     weight_decay=wd)

    def optimize_epoch(self, num_epochs, output_path: str, logger: AbstractLogger):
        if self._optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self._data_loader is None:
            self._data_loader = DataLoader(self._memory, self._batch_size, shuffle=True)
        average_epoch_loss = 0
        min_loss = np.inf
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self._data_loader:
                inputs, values = data
                inputs = {k: v.float().to(self.device) for k, v in inputs.items()}
                values = values.to(self.device)
                # inputs = Variable(inputs)
                # values = Variable(values)

                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._criterion(outputs, values)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()

            average_epoch_loss = epoch_loss / len(self._memory)
            logger.log("il/loss", average_epoch_loss)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)
            if average_epoch_loss < min_loss:
                min_loss = average_epoch_loss
                torch.save(self._model.state_dict(), str(output_path))

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self._optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self._data_loader is None:
            self._data_loader = DataLoader(self._memory, self._batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values = next(iter(self._data_loader))
            inputs = {k: v.float().to(self.device) for k, v in inputs.items()}
            values = values.to(self.device)
            # inputs = Variable(inputs)
            # values = Variable(values)

            self._optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = self._criterion(outputs, values)
            loss.backward()
            self._optimizer.step()
            losses += loss.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss

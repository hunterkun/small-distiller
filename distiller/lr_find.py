from matplotlib import pyplot as plt
import tqdm
from distiller.learning_rate import ExponentialLR, LinearLR
import torch

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.model_device = next(self.model.parameters()).device
        if device:
            self.device = device
        else:
            self.device=self.model_device
        
    def range_test(self, train_loader, val_loader=None, end_lr=10, num_iter=100, step_mode="exp", smooth_f=0.05, diverge_th=5):
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.model.to(self.device)
        if step_mode.lower() == 'exp':
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == 'linear':
            lr_schedule = LineraLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))
        
        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0,1]")
        
        iterator = iter(train_loader)
        for iteration in range(num_iter):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            loss = self.train_batch(inputs, targets)
            if val_loader:
                loss = self.validate(val_loader)
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[0])
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history['loss'][-1]
                if loss < self.best_loss:
                    self.best_loss = loss
                    
            # check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print('Stop early, the loss has diverged')
                break
        print("Learning rate search finished. See the graph with {LRFinder_name}.plot()")

                
    def plot(self, skip_start=10, skip_end=5):
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        
        lrs = self.history['lr']
        losses = self.history['loss']
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start: - skip_end]
            losses = losses[skip_start: - skip_end]
        fig, ax = plt.subplots(1,1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        plt.show()

    def train_batch(self, inputs, target):
        self.model.train()
        inputs, target = inputs.to(self.device), target.to(self.device)

        output = self.model(inputs)

        loss = self.criterion(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dataloader):
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                # Move data to the correct device
                inputs = inputs.to(self.device)
                labels = targets.to(self.device)

                # Forward pass and loss computation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        return running_loss / len(dataloader.dataset)
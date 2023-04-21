import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from .losses import classification_loss
import tqdm
import numpy as np
import os
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    fc_beta = 0.3
    fc_gamma = 0.75
    resize = None

    def __init__(self, ViT, savefolder):
        self.ViT = ViT

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

    def batch(self, x, y, train=False):
        X = torch.Tensor(x).to(device)
        Y = torch.Tensor(y).to(device)

        if self.resize is not None:
            X = self.resize(X)

        Ypred = self.ViT(X)

        if train:
            self.optimizer.zero_grad()

        # fc_tversky(Y, Ypred, beta=self.fc_beta, gamma=self.fc_gamma)
        loss = classification_loss(Ypred, Y)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss

    def train(self, train_data, val_data, nepochs, learning_rate=1.e-3, save_freq=5, lr_decay=None, decay_freq=5):
        self.optimizer = optim.NAdam(self.ViT.parameters(), lr=learning_rate)

        loss_ep = []

        if lr_decay is not None:
            scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            scheduler = None

        for epoch in range(1, nepochs + 1):
            if scheduler is not None:
                learning_rate = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} -- lr: {learning_rate}")
            print("-------------------------------------------------------")

            # batch loss data
            pbar = tqdm.tqdm(train_data, desc='Training: ')

            train_data.shuffle()

            # set to training mode
            self.ViT.train()

            loss_batch = []
            # loop through the training data
            for i, (input_img, target_classes) in enumerate(pbar):

                # train on this batch
                loss = self.batch(input_img, target_classes, train=True)

                # append the current batch loss
                loss_batch.append(loss.item())

                mean_loss = np.mean(loss_batch)

                pbar.set_postfix_str(f"loss: {mean_loss:.3e}")

            # update the epoch loss
            loss_ep.append(np.mean(loss_batch))

            # batch loss data
            pbar = tqdm.tqdm(val_data, desc='Validation: ')

            val_data.shuffle()

            # set to training mode
            self.ViT.eval()

            loss_batch = []
            # loop through the training data
            for i, (input_img, target_classes) in enumerate(pbar):

                # train on this batch
                loss = self.batch(input_img, target_classes, train=False)

                # append the current batch loss
                loss_batch.append(loss.item())

                mean_loss = np.mean(loss_batch)

                pbar.set_postfix_str(f"loss: {mean_loss:.3e}")

            if epoch % save_freq == 0:
                self.save(epoch)

            if scheduler is not None:
                if epoch % decay_freq == 0:
                    scheduler.step()

    def save(self, epoch):
        savefile = f'{self.savefolder}/CNN_ep_{epoch:03d}.pth'

        print(f"Saving to {savefile}")
        torch.save(self.ViT.state_dict(), savefile)

    def load_last_checkpoint(self):
        checkpoints = sorted(glob.glob(self.savefolder + "CNN_ep*.pth"))

        epochs = set([int(ch.split(
            '/')[-1].replace('CNN_ep_', '')[:-4]) for
            ch in checkpoints])

        try:
            assert len(epochs) > 0, "No checkpoints found!"

            start = max(epochs)
            self.load(f"{self.savefolder}/CNN_ep_{start:03d}.pth")
            self.start = start + 1
        except Exception as e:
            print(e)
            print("Checkpoints not loaded")

    def load(self, checkpoint):
        self.ViT.load_state_dict(torch.load(checkpoint, map_location=device))

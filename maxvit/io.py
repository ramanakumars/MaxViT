import numpy as np
import glob
import os
from astropy.io import ascii
from torchvision.transforms import Resize
from torchvision.io import read_image


class DataGenerator():
    def shuffle(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return self.ndata // self.batch_size

    def __getitem__(self, index):

        if index >= len(self):
            raise StopIteration()

        batch_indices = self.indices[index *
                                     self.batch_size:(index + 1) * self.batch_size]

        imgs, labels = self.get_from_indices(batch_indices)

        return imgs, labels


class PNGDataGenerator(DataGenerator):
    def __init__(self, png_folder, labels, batch_size, indices=None):
        self.png_folder = png_folder

        self.labels = np.load(labels)

        self.batch_size = batch_size

        self.subject_ids = np.sort(self.labels[:, 0])

        if indices is not None:
            self.indices = indices
            self.ndata = len(indices)
        else:
            self.ndata = len(self.subject_ids)
            self.indices = np.arange(self.ndata)

        print(f"Found data with {self.ndata} images")

    def get_from_indices(self, indices):
        subject_ids = self.subject_ids[indices]

        imgs = []
        labels = np.zeros(((len(subject_ids), self.labels.shape[1] - 1)))

        for j, sid in enumerate(subject_ids):
            img = read_image(self.png_folder + f"{sid}.png") / 255.
            imgs.append(img.numpy())
            labels[j, :] = self.labels[np.where(self.subject_ids == sid)[
                0], 1:].astype(float)

        return np.asarray(imgs), labels


class JPEGDataGenerator(DataGenerator):
    def __init__(self, main_folder, batch_size, indices=None):
        self.main_folder = main_folder

        self.labels = np.asarray([os.path.normpath(e).split(
            os.sep)[-1] for e in sorted(glob.glob(self.main_folder + "*/"))])
        self.imgs = np.asarray(
            sorted(glob.glob(self.main_folder + "*/*.[jp]*")))

        self.batch_size = batch_size

        if indices is not None:
            self.indices = indices
            self.ndata = len(indices)
        else:
            self.ndata = len(self.imgs)
            self.indices = np.arange(self.ndata)

        print(f"Found data with {self.ndata} images")

    def get_from_indices(self, indices):
        img_fnames = self.imgs[indices]

        imgs = []
        labels = np.zeros(((len(img_fnames), self.labels.size)))

        for j, imgi in enumerate(img_fnames):
            fname_parts = os.path.normpath(imgi).split(os.sep)

            label = fname_parts[-2]

            img = read_image(imgi) / 255.

            imgs.append(img.numpy())
            labels[j, np.where(self.labels == label)[0]] = 1

        return np.asarray(imgs), labels


class SnapshotDataGenerator(DataGenerator):
    resize = Resize(256)

    def __init__(self, basefolder, labelfile, batch_size, indices=None):
        self.basefolder = basefolder
        self.labels = ascii.read(labelfile, format='csv')
        self.unique_labels = sorted(self.labels.colnames[1:])

        capture_IDs = np.asarray(self.labels['captureID'])

        for capID in capture_IDs:
            if not os.path.isfile(f'{basefolder}/{capID}.png'):
                raise FileNotFoundError(f'Capture ID: {capID} does not exist!')

        self.batch_size = batch_size

        if indices is not None:
            self.indices = indices
            self.ndata = len(indices)
        else:
            self.ndata = len(self.labels)
            self.indices = np.arange(self.ndata)

    def get_from_indices(self, batch_indices):
        label_subset = self.labels[batch_indices]

        imgs = []
        labels = []

        for i, row in enumerate(label_subset):
            img = read_image(f"{self.basefolder}/{row['captureID']}.png") / 255.
            if self.resize is not None:
                img = self.resize(img)
            labi = [row[lab] for lab in self.unique_labels]

            imgs.append(img.numpy())
            labels.append(labi)

        return np.asarray(imgs), np.asarray(labels)


class GZ2DataGenerator(DataGenerator):
    resize = None

    def __init__(self, basefolder, labelfile, batch_size, indices=None):
        self.basefolder = basefolder
        self.labels = ascii.read(labelfile, format='csv')
        self.unique_labels = sorted(np.unique(self.labels['label']).tolist())

        self.batch_size = batch_size

        if indices is not None:
            self.indices = indices
            self.ndata = len(indices)
        else:
            self.ndata = len(self.labels)
            self.indices = np.arange(self.ndata)

    def set_resize(self, resize):
        self.resize = Resize(resize)

    def get_from_indices(self, batch_indices):
        label_subset = self.labels[batch_indices]

        imgs = []
        labels = []

        for i, row in enumerate(label_subset):
            img = read_image(self.basefolder + row['filename']) / 255.
            labi = [0] * len(self.unique_labels)
            labi[self.unique_labels.index(row['label'])] = 1

            if self.resize is not None:
                img = self.resize(img)

            imgs.append(img.numpy())
            labels.append(labi)

        return np.asarray(imgs), np.asarray(labels)


def create_generators(generator, val_split=0.1, **kwargs):
    gen = generator(**kwargs)

    ndata = gen.ndata

    print(f"Creating generators from {ndata} images")

    inds = np.arange(ndata)
    np.random.shuffle(inds)

    val_split_ind = int(ndata * val_split)
    val_ind = inds[:val_split_ind]
    training_ind = inds[val_split_ind:]

    train_data = generator(**kwargs, indices=training_ind)
    val_data = generator(**kwargs, indices=val_ind)

    return train_data, val_data

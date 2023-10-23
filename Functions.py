import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter1d

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import torchaudio

from torch import nn
from torchsummary import summary
from torchvision.models import resnet18, ResNet18_Weights


class SwahiliDataset(Dataset):
    """
    Custom dataset class for loading and transforming the training data
    
    Input:
    * annotations_file
    * audio_dir
    * transformation
    * target_sample_rate
    * num_samples
    * device
    
    Returns:
    * signal
    * label
    """
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label


    def _resample_if_necessary(self, signal, sr):
        """
        All sounds need to have the same sampling rate.
        resampling if the sampling rate differs from the wanted sampling rate.
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        """
        All sounds need to have the same amount of layers.
        If a sound was recorded in stereo, it is mixed down to mono.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _cut_if_necessary(self, signal):
        """
        As most sounds have a silence at the beginning and end and also disturbing sounds like crackling or laughs,
        the signal is centered to the loudest part and cut to the wanted length.
        The loudest part is found by filtering a moving minimum filter of 1/16th second.
        This way, very short loud impulse sounds are removed.
        """
        if signal.shape[1] > self.num_samples:
            # Cut first 0.5 seconds, because of signal problems that disturb the mininum_filter1d
            signal=signal[:,int(self.target_sample_rate/2):]
            # Take the minimum of every 16th second moving filter. 
            # This filters out short maxima created by impulse noises that might be louder than the speaker.
            min_filter = minimum_filter1d(abs(signal), size=int(self.target_sample_rate/16), mode='constant')
            ind_max = min_filter[0].argmax()
            window_range = int(self.num_samples/2)
            if ind_max<=window_range:
                ind_lrange=0
            else:
                ind_lrange=int(ind_max-window_range)

            if (signal.shape[1]-ind_max)<=window_range:
                ind_rrange=int(signal.shape[1])
            else:
                ind_rrange=int(ind_max+window_range)
            signal=signal[:,ind_lrange:ind_rrange]
        return signal

    def _right_pad_if_necessary(self, signal):
        """
        Pad the sounds to the same length.
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


    def _get_audio_sample_path(self, index):
        """
        Path of the audio files.
        The individual file names are found in the annotaions file in column 0
        """
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        """
        Numerically encoded label.
        The target label is found in column 1 of the annotations file. 
        Column 1 is the swahili word, Column 2 is the english equivalent. 
        """
        labels = ['hapana',
                  'kumi',
                  'mbili',
                  'moja',
                  'nane',
                  'ndio',
                  'nne',
                  'saba',
                  'sita',
                  'tano',
                  'tatu',
                  'tisa']
        label = labels.index(self.annotations.iloc[index, 1])
        return label
    




def audio_transforms_spec(signal):
    """
    Transformations applied to the data at loading.
    Audio files are transformed into spectrograms.
    The values are transformed to dB scale.
    High frequencies are cut off as these are not containing relevant information for spoken words.
    
    Input: sound signal (amplitude per time)
    Returns: spectrogram (frequency per time)
    """
    N_FFT = 1024
    
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        hop_length=100,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=100)
    
    spec = spectrogram(signal).to(device)
    spec_db = to_db(spec[:,:int(N_FFT/4),:])
    return spec_db





def audio_transforms_mel(signal):
    """
    Transformations applied to the data at loading.
    Audio files are transformed into spectrograms.
    The values are transformed to dB scale.
    High frequencies are cut off as these are not containing relevant information for spoken words.
    
    Input: sound signal (amplitude per time)
    Returns: spectrogram (frequency per time)
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=100,
        n_mels=200
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=100)
    
    spec = mel_spectrogram(signal).to(device)
    spec_db = to_db(spec)
    return spec_db



def create_data_loaders(train_data, batch_size, val_split, shuffle_dataset, random_seed):
    """
    Data Loader for training and validation set from the same custom Dataset for trainings data.
    
    Input:
    train_data (Custom Dataset with signal and label)
    batch_size
    val_split (split size)
    shuffle_dataset (bool, if dataset is shuffled for train/val-split)
    random_seed
    
    Returns:
    train_dataloader
    val_dataloader
    """
    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                    sampler=val_sampler)
    
    return train_dataloader, val_dataloader


def train_single_epoch(model, train_data_loader, val_data_loader, loss_fn, optimiser, scheduler, device):
    for input, target in train_data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        train_prediction = model(input)
        train_loss = loss_fn(train_prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

    
    for input, target in val_data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        val_prediction = model(input)
        val_loss = loss_fn(val_prediction, target)
    
    
    scheduler.step()
    print(f"training loss: {train_loss.item()}, validation loss: {val_loss.item()}")


def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, scheduler, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, train_data_loader, val_data_loader, loss_fn, optimiser, scheduler, device)
        print("---------------------------")
    print("Finished training")



class SwahiliDataset_Testset(SwahiliDataset):
    """
    Custom Dataset for test data.
    The label is not known for the test data, therefore the Dataset for training data cannot be used.
    This Dataset inherits the functions defined in SwahiliDataset.
    The same transformations are applied to the signal.
    The getitem function is adjusted to return only the signal.
    """
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal
    

def predict_testset(model, data_loader):
    """
    Predictions on the test set
    Returns:
    predictions with values between 0 and 1 due to added softmax layer.
    """
    model.eval()
    with torch.no_grad():
        for input in data_loader:
            input = input.to(device)
            predictions = nn.Softmax(dim=1)(model(input))
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
    return predictions

def create_test_data_loader(test_data):
    """
    test data loader with full test data set as batch
    """
    dataset_size = len(test_data)
    test_dataloader = DataLoader(test_data, batch_size=dataset_size)
    return test_dataloader    
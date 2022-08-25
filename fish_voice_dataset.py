from dataclasses import dataclass
import librosa
import glob

from torch.nn.modules import transformer
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from scipy.signal import resample
from itertools import chain


def load_audio(path, sr=None):
    y, _ = librosa.load(path, sr=None)
    y = resample(y, num=sr*2)
    return y

# def get_wav_name(split='strong'):
#     """
#     params: str
#         middle, none, strong, weak
#     """
#     path = '/vol/research/Fish_tracking_master/sound'
#     wav_dir = os.path.join(path, split, '*', '*.wav')
#     return glob.glob(wav_dir)
def get_wav_name(split='strong'):
    """
    params: str
        middle, none, strong, weak
    """
    # path = '/vol/research/Fish_tracking_master/audio_dataset'
    path = '/vol/research/Fish_tracking_master/fish_num/15'
    audio =[]
    l1 = os.listdir(path)
    for dir in l1:
        l2 = os.listdir(os.path.join(path,dir))
        for dir1 in l2:
            wav_dir = os.path.join(path, dir, dir1, split, '*.wav')
            audio.append(glob.glob(wav_dir))
    return list(chain.from_iterable(audio))


def data_generator(seed=20, test_sample_per_class=100):
    """
    class to label mapping:
    none: 0
    strong: 1
    middle: 2
    weak: 3
    """

    random_state = np.random.RandomState(seed)
    strong_list = get_wav_name(split='strong')
    middle_list = get_wav_name(split='medium')
    weak_list = get_wav_name(split='weak')
    none_list = get_wav_name(split='none')

    random_state.shuffle(strong_list)
    random_state.shuffle(middle_list)
    random_state.shuffle(weak_list)
    random_state.shuffle(none_list)

    strong_train = strong_list[test_sample_per_class:2120]
    middle_train = middle_list[test_sample_per_class:1775]
    weak_train = weak_list[test_sample_per_class:1950]
    none_train = none_list[test_sample_per_class:980]

    strong_test = strong_list[:test_sample_per_class]
    middle_test = middle_list[:test_sample_per_class]
    weak_test = weak_list[:test_sample_per_class]
    none_test = none_list[:test_sample_per_class]

    train_dict = []
    test_dict = []

    for wav in strong_train:
        train_dict.append([wav, 1])
    
    for wav in middle_train:
        train_dict.append([wav, 2])
    
    for wav in weak_train:
        train_dict.append([wav, 3])

    for wav in none_train:
        train_dict.append([wav, 0])
    
    for wav in strong_test:
        test_dict.append([wav, 1])
    
    for wav in middle_test:
        test_dict.append([wav, 2])
    
    for wav in weak_test:
        test_dict.append([wav, 3])

    for wav in none_test:
        test_dict.append([wav, 0])
    
    random_state.shuffle(train_dict)

    return train_dict, test_dict
    
class Fish_Voice_Dataset(Dataset):
    def __init__(self, split='train', sample_rate=None):
        """
        split: train or test
        if sample_rate=None, read audio with the default sr
        """

        train_dict, test_dict = data_generator(seed=20, test_sample_per_class=100)
        if split == 'train':
            self.data_dict = train_dict
        elif split == 'test':
            self.data_dict = test_dict
        self.sample_rate = sample_rate
    
    def __len__(self):


        return len(self.data_dict)
    
    def __getitem__(self, index):
        wav_name, target = self.data_dict[index]
        wav = load_audio(wav_name, sr=self.sample_rate)

        wav = np.array(wav)
        # change 'eye(num)' if using different class nums
        target = np.eye(4)[target]

        data_dict = {'audio_name': wav_name, 'waveform': wav, 'target': target}

        return data_dict


def collate_fn(batch):
    wav_name = [data['audio_name'] for data in batch]
    wav = [data['waveform'] for data in batch]
    target = [data['target'] for data in batch]

    wav = torch.FloatTensor(np.array(wav))
    target = torch.FloatTensor(np.array(target))

    return {'audio_name': wav_name, 'waveform': wav, 'target': target}


def get_dataloader(split,
                   batch_size,
                   sample_rate,
                   # seed,
                   shuffle=False,
                   drop_last=False,
                   num_workers=8):

    dataset = Fish_Voice_Dataset(split=split, sample_rate=sample_rate)

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
    train_loader = get_dataloader(split='test', batch_size=32, sample_rate=44000)
    for item in train_loader:
        print(item)
        pass


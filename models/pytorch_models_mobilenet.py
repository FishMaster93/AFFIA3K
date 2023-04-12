import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from pytorch.pytorch_utils import do_mixup, interpolate, pad_framewise_output
from pytorch.models import init_layer, init_bn


class Mobilev2Block(nn.Module):
    def __init__(self, in_channels, out_channels, r=1):
        super(Mobilev2Block, self).__init__()

        size = 3
        pad = size // 2
        r = r

        self.conv1a = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels * r,
                              kernel_size=(1, 1), stride=(1, 1), 
                              dilation=(1, 1), 
                              padding=(0, 0), bias=False)

        self.conv1b = nn.Conv2d(in_channels=out_channels * r, 
                              out_channels=out_channels * r,
                              kernel_size=(size, size), stride=(1, 1), 
                              dilation=(1, 1), groups=out_channels * r,
                              padding=(pad, pad), bias=False)

        self.conv1c = nn.Conv2d(in_channels=out_channels * r, 
                              out_channels=out_channels,
                              kernel_size=(1, 1), stride=(1, 1), 
                              dilation=(1, 1), 
                              padding=(0, 0), bias=False)

        self.conv2a = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels * r,
                              kernel_size=(1, 1), stride=(1, 1), 
                              dilation=(1, 1), 
                              padding=(0, 0), bias=False)

        self.conv2b = nn.Conv2d(in_channels=out_channels * r, 
                              out_channels=out_channels * r,
                              kernel_size=(size, size), stride=(1, 1), 
                              dilation=(1, 1), groups=out_channels * r,
                              padding=(pad, pad), bias=False)

        self.conv2c = nn.Conv2d(in_channels=out_channels * r, 
                              out_channels=out_channels,
                              kernel_size=(1, 1), stride=(1, 1), 
                              dilation=(1, 1), 
                              padding=(0, 0), bias=False)

        self.bn1a = nn.BatchNorm2d(in_channels)
        self.bn1b = nn.BatchNorm2d(out_channels * r)
        self.bn1c = nn.BatchNorm2d(out_channels)
        self.bn2a = nn.BatchNorm2d(out_channels)
        self.bn2b = nn.BatchNorm2d(out_channels * r)
        self.bn2c = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, 
                out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()


    def init_weights(self):
        init_layer(self.conv1a)
        init_layer(self.conv1b)
        init_layer(self.conv1c)
        init_layer(self.conv2a)
        init_layer(self.conv2b)
        init_layer(self.conv2c)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_bn(self.bn1c)
        init_bn(self.bn2a)
        init_bn(self.bn2b)
        init_bn(self.bn2c)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):        

        origin = input
        x = self.conv1a(F.leaky_relu_(self.bn1a(origin), negative_slope=0.01))
        x = self.conv1b(F.leaky_relu_(self.bn1b(x), negative_slope=0.01))
        x = self.conv1c(F.leaky_relu_(self.bn1c(x), negative_slope=0.01))

        if self.is_shortcut:
            origin = self.shortcut(origin) + x
        else:
            origin = origin + x

        x = self.conv2a(F.leaky_relu_(self.bn2a(origin), negative_slope=0.01))
        x = self.conv2b(F.leaky_relu_(self.bn2b(x), negative_slope=0.01))
        x = self.conv2c(F.leaky_relu_(self.bn2c(x), negative_slope=0.01))

        x = origin + x

        x = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)
        
        return x


class Cnn14_mobilev2_16k(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        
        super(Cnn14_mobilev2_16k, self).__init__() 

        # assert sample_rate == 16000
        # assert window_size == 512
        # assert hop_size == 160
        # assert mel_bins == 64
        # assert fmin == 50
        # assert fmax == 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = Mobilev2Block(in_channels=1, out_channels=16)
        self.conv_block2 = Mobilev2Block(in_channels=16, out_channels=32)
        self.conv_block3 = Mobilev2Block(in_channels=32, out_channels=64)
        self.conv_block4 = Mobilev2Block(in_channels=64, out_channels=128)
        self.conv_block5 = Mobilev2Block(in_channels=128, out_channels=128)
        self.conv_block6 = Mobilev2Block(in_channels=128, out_channels=128)

        self.fc1 = nn.Linear(128, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu_(self.fc1(x), negative_slope=0.01)
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x)
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Cnn14_mobilev2_DecisionLevelMax_16k(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        
        super(Cnn14_mobilev2_DecisionLevelMax_16k, self).__init__() 

        assert sample_rate == 16000
        assert window_size == 512
        assert hop_size == 160
        assert mel_bins == 64
        assert fmin == 50
        assert fmax == 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.interpolate_ratio = 32

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = Mobilev2Block(in_channels=1, out_channels=16)
        self.conv_block2 = Mobilev2Block(in_channels=16, out_channels=32)
        self.conv_block3 = Mobilev2Block(in_channels=32, out_channels=64)
        self.conv_block4 = Mobilev2Block(in_channels=64, out_channels=128)
        self.conv_block5 = Mobilev2Block(in_channels=128, out_channels=128)
        self.conv_block6 = Mobilev2Block(in_channels=128, out_channels=128)

        self.fc1 = nn.Linear(128, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # if self.training:
        #     x = self.spec_augmenter(x)

        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.leaky_relu_(self.fc1(x), negative_slope=0.01)
        # x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        '''
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)
        '''
        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}

        # from IPython import embed; embed(using=False); os._exit(0)

        return output_dict


class Cnn10_mobilev2_16k(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):

        super(Cnn10_mobilev2_16k, self).__init__()

        # assert sample_rate == 16000
        # assert window_size == 512
        # assert hop_size == 160
        # assert mel_bins == 64
        # assert fmin == 50
        # assert fmax == 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = Mobilev2Block(in_channels=1, out_channels=16)
        self.conv_block2 = Mobilev2Block(in_channels=16, out_channels=32)
        self.conv_block3 = Mobilev2Block(in_channels=32, out_channels=64)
        self.conv_block4 = Mobilev2Block(in_channels=64, out_channels=128)


        self.fc1 = nn.Linear(128, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu_(self.fc1(x), negative_slope=0.01)
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x)

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
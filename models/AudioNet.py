import sys
sys.path.append('/vol/research/Fish_tracking_master/knowledge_distillation/kd_fish_voice_recognition/')
import torch.nn as  nn
import torch
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from pytorch.models import ConvBlock5x5, init_bn, init_layer
from models.module import TransitionBlock, BroadcastedBlock


class AudioNet(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(AudioNet, self).__init__()

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

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.l1_align = nn.Linear(512, 36, bias=True)
        self.l2_align = nn.Linear(15, 9, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

        init_layer(self.l1_align)
        init_layer(self.l2_align)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        embedding_l2 = F.relu_(self.l2_align(x)) # [bs, 512, 15] -> [bs, 512, 9]
        embedding_l2 = embedding_l2.permute(0, 2, 1) # [bs, 9, 512]
        embedding_l2 = F.dropout(embedding_l2, p=0.2, training=self.training) # [bs, 9, 512]
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))

        embedding_l1 = self.l1_align(x)
        embedding_l1 = F.dropout(embedding_l1, p=0.2, training=self.training) # [bs, 36]

        if self.training:
            clipwise_output = self.fc_audioset(x)
        else:
            clipwise_output = torch.softmax(self.fc_audioset(x), dim=-1)
        
        output_dict = {'embedding_l1': embedding_l1, 'embedding_l2': embedding_l2, 'clipwise_output': clipwise_output}

        return output_dict


class BC_ResNet(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, c=4, norm=False):
        
        super(BC_ResNet, self).__init__()

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

        c = 10 * c
        self.conv1 = nn.Conv2d(1, 2 * c, 5, stride=(2, 2), padding=(2, 2))
        self.block1_1 = TransitionBlock(2 * c, c)
        self.block1_2 = BroadcastedBlock(c)

        self.block2_1 = nn.MaxPool2d(2)

        self.block3_1 = TransitionBlock(c, int(1.5 * c))
        self.block3_2 = BroadcastedBlock(int(1.5 * c))

        self.block4_1 = nn.MaxPool2d(2)

        self.block5_1 = TransitionBlock(int(1.5 * c), int(2 * c))
        self.block5_2 = BroadcastedBlock(int(2 * c))

        self.block6_1 = TransitionBlock(int(2 * c), int(2.5 * c))
        self.block6_2 = BroadcastedBlock(int(2.5 * c))
        self.block6_3 = BroadcastedBlock(int(2.5 * c))

        self.block7_1 = nn.Conv2d(int(2.5 * c), classes_num, 1)

        self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = norm
        self.fc_audioset = nn.Linear(1, classes_num, bias=True)
        if norm:
           self.one = nn.InstanceNorm2d(1)
           self.two = nn.InstanceNorm2d(int(1))
           self.three = nn.InstanceNorm2d(int(1))
           self.four = nn.InstanceNorm2d(int(1))
           self.five = nn.InstanceNorm2d(int(1))

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc_audioset)


 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)
        
        out = x
        if self.norm:
               out =self.lamb*out + self.one(out)
        out = self.conv1(out)

        out = self.block1_1(out)

        out = self.block1_2(out)
        if self.norm:
           out =self.lamb*out + self.two(out)

        out = self.block2_1(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        if self.norm:
           out =self.lamb*out + self.three(out)

        out = self.block4_1(out)

        out = self.block5_1(out)
        out = self.block5_2(out)
        if self.norm:
           out =self.lamb*out + self.four(out)

        out = self.block6_1(out)
        out = self.block6_2(out)
        out = self.block6_3(out)
        if self.norm:
           out =self.lamb*out + self.five(out)

        out = self.block7_1(out)

        out = self.block8_1(out)
        out = self.block8_1(out)

        embedding = F.dropout(out, p=0.2, training=self.training)
        clipwise_output = torch.squeeze(torch.squeeze(out,dim=2),dim=2)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding}

        return output_dict



if __name__ == '__main__':
    model_params = {'sample_rate': 128000,
                    'window_size': 2048,
                    'hop_size': 1024,
                    'mel_bins': 64,
                    'fmin': 50,
                    'fmax': 14000,
                    'classes_num': 4}

    input = torch.randn(4, 2 * 128000)
    model = AudioNet(**model_params)
    print(model(input).shape)
    pass

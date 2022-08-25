from posixpath import split

from matplotlib import pyplot as plt
from sklearn import metrics
from fish_voice_dataset import get_dataloader
import torch.optim as optim
import torch
from pytorch.losses import get_loss_func
import os
import time
import logging as log_config
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
from pytorch.evaluate import Evaluator
from early_stopping import EarlyStopping
from pytorch.evaluate import Evaluator
import torch.nn.functional as F
import argparse
from early_stopping import save_model
from pytorch.models import Cnn14, Cnn4, Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, MobileNetV1, MobileNetV2
from panns import PANNS_Cnn10, PANNS_Cnn6, PANNS_Cnn14
from models.AudioNet import BC_ResNet, AudioNet
from tqdm import tqdm


def save_model(path, model, optimizer, ave_precision, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ave_precision': ave_precision,
    }, path)


def train(model, train_loader, test_loader, epoch, device):
    logger.info("Starting new training run")

    evaluator = Evaluator(model=model)
    best_mAP = 0
    best_message = ''
    list1 = []
    for epoch in range(epoch):
        epoch = epoch + 1

        loss_func = get_loss_func('clip_ce')

        mean_loss = 0
        for data_dict in tqdm(train_loader):
            data_dict['waveform'] = data_dict['waveform'].to(device)
            data_dict['target'] = data_dict['target'].to(device)

            model.train()

            output_dict = model(data_dict['waveform'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            target_dict = {'target': data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

            loss = loss_func(output_dict, target_dict)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss = loss.item()

            mean_loss += loss

        epoch_loss = mean_loss / len(train_loader)
        logger.info(f"Training loss {epoch_loss} at epoch {epoch}")

        # Evaluate
        test_statistics = evaluator.evaluate(test_loader)
        ave_precision = np.mean(test_statistics['average_precision'])
        ave_acc = np.mean(test_statistics['accuracy'])
        message = test_statistics['message']

        list1.append(ave_precision)

        logger.info(f'Validate test mAP: {ave_precision}, accuracy: {ave_acc}')
        logger.info(f'Metrics report by class: {message}')

        if ave_precision > best_mAP:
            best_mAP = ave_precision
            best_message = message

            save_model(best_ckpt_name, model, optimizer, ave_precision, epoch)

    # report final metrics
    logger.info(f'Best test mAP: {best_mAP}, accuracy: {ave_acc}')
    logger.info(f'Best metrics report by class: {best_message}')
    dataframe = pd.DataFrame(list1)
    print(dataframe)
    dataframe.to_csv('list15.csv', index=False, header=False)
    plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--exp_name', type=str, default='100_Cnn6')
    parser.add_argument('--model_type', type=str, default='Cnn6')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--sample_rate', type=int, default=128000)  # 16k, 32k, 44k, 64k, 128k
    parser.add_argument('--window_size', type=int, default=2048)
    parser.add_argument('--hop_size', type=int, default=1024)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20)

    args = parser.parse_args()

    exp_name = args.exp_name

    # Training parameters
    model_type = args.model_type
    batch_size = args.batch_size
    epoch = args.epoch
    learning_rate = args.learning_rate
    seed = args.seed

    # STFT parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins

    ckpt_dir = os.path.join('Fish_workspace8_23', exp_name, 'save_models')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt_name = os.path.join(ckpt_dir, 'best.pt')
    log_dir = os.path.join('Fish_workspace8_23', exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_config.basicConfig(
        level=log_config.INFO,
        format=' %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_config.FileHandler(os.path.join(log_dir,
                                                '%s-%d.log' % (exp_name, time.time()))),
            log_config.StreamHandler()
        ]
    )

    logger = log_config.getLogger()

    # change 'classes_num' if using different class nums
    model_params = {'sample_rate': sample_rate,
                    'window_size': window_size,
                    'hop_size': hop_size,
                    'mel_bins': mel_bins,
                    'fmin': 50,
                    'fmax': None,
                    'classes_num': 4}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = eval(model_type)
    model = Model(**model_params)
    print(model)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    train_loader = get_dataloader(split='train', batch_size=batch_size, sample_rate=sample_rate)
    test_loader = get_dataloader(split='test', batch_size=batch_size, sample_rate=sample_rate)

    logger.info(f"Experiments running on {device}")
    train(model, train_loader, test_loader, epoch, device)

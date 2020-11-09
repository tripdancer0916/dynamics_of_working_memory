"""Training models"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.append('../')

from torch.autograd import Variable

from static_input_dataset import StaticInput
from model import RecurrentNeuralNetwork


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'CHECK_TIMING' not in cfg['DATALOADER']:
        cfg['DATALOADER']['CHECKTIMING'] = 5

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/static_input', exist_ok=True)
    save_path = f'trained_model/static_input/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    model = RecurrentNeuralNetwork(n_in=1, n_out=1, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   use_bias=cfg['MODEL']['USE_BIAS']).to(device)

    train_dataset = StaticInput(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                time_scale=cfg['MODEL']['ALPHA'],
                                value_min=cfg['DATALOADER']['VALUE_MIN'],
                                value_max=cfg['DATALOADER']['VALUE_MAX'],
                                signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                sigma_in=cfg['DATALOADER']['SIGMA_IN'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)
    print('Epoch Loss')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, target = data
            # print(inputs.shape)
            inputs, target = inputs.float(), target.float()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)

            # hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE'])
            hidden_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()
            hidden_list, output, hidden = model(inputs, hidden)

            check_timing = np.random.randint(-cfg['DATALOADER']['CHECK_TIMING'], 0)
            loss = torch.nn.MSELoss()(output[:, check_timing], target)
            if 'FIXED_DURATION' in cfg['DATALOADER']:
                for j in range(1, cfg['DATALOADER']['FIXED_DURATION'] + 1):
                    loss += torch.nn.MSELoss()(output[:, check_timing - j], target)
            dummy_zero = torch.zeros([cfg['TRAIN']['BATCHSIZE'],
                                      cfg['DATALOADER']['TIME_LENGTH'] + 1,
                                      cfg['MODEL']['SIZE']]).float().to(device)
            active_norm = torch.nn.MSELoss()(hidden_list, dummy_zero)

            loss += cfg['TRAIN']['ACTIVATION_LAMBDA'] * active_norm
            loss.backward()
            optimizer.step()

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            print(f'{epoch}, {loss.item():.4f}')
            print('output: ',
                  output[0, check_timing - cfg['DATALOADER']['FIXED_DURATION']: check_timing, 0].cpu().detach().numpy())
            print('target: ', target[0, 0].cpu().detach().numpy())
        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)

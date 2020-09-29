import argparse
import json
import os
import shutil
import time
import glob

import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
import dataset
from model import LaneNet
from utils.transforms import *
from utils.lr_scheduler import PolyLR
from utils.postprocess import embedding_post_process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp5")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device(exp_cfg['device'])

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
transform_train = Compose(Resize(resize_shape), Darkness(5), Rotation(2),
                          ToTensor(), Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
# ------------ val data ------------
transform_val = Compose(Resize(resize_shape), ToTensor(),
                        Normalize(mean=mean, std=std))
val_dataset = Dataset_Type(Dataset_Path[dataset_name], "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=exp_cfg['dataset']['batch_size'], collate_fn=val_dataset.collate, num_workers=0, pin_memory=True)

# ------------ preparation ------------
net = LaneNet(pretrained=True, **exp_cfg['net'])
net = net.to(device)

if 'momentum' in exp_cfg['optim']:
    optimizer = optim.SGD(net.parameters(), **exp_cfg['optim'])
else:
    optimizer = optim.Adam(net.parameters(), **exp_cfg['optim'])
lr_scheduler = PolyLR(optimizer, 0.9, exp_cfg['MAX_ITER'])

def val(epoch):
    print("Val Epoch: {}".format(epoch))

    net.eval()
    val_loss = 0
    val_loss_bin_seg = 0
    val_loss_var = 0
    val_loss_dist = 0
    val_loss_reg = 0
    progressbar = tqdm(range(len(val_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)

            output = net(img, segLabel)
            embedding = output['embedding']
            binary_seg = output['binary_seg']
            seg_loss = output['seg_loss']
            var_loss = output['var_loss']
            dist_loss = output['dist_loss']
            reg_loss = output['reg_loss']
            loss = output['loss']

            val_loss += loss.item()
            val_loss_bin_seg += seg_loss.item()
            val_loss_var += var_loss.item()
            val_loss_dist += dist_loss.item()
            val_loss_reg += reg_loss.item()

            progressbar.set_description("batch loss: {:.5f}".format(loss.item()))
            progressbar.update(1)
        progressbar.close()

    print(
        'Loss: {:.4f}|'
        'Bin seg loss: {:.4f}|'
        'Var loss: {:.4f}|'
        'Dist loss: {:.4f}|'
        'Reg loss: {:.4f}'.format(
            val_loss,
            val_loss_bin_seg,
            val_loss_var,
            val_loss_dist, 
            val_loss_reg
            )
        )

def main():
    best_model = glob.glob(os.path.join(exp_dir, '*_best.pth'))[0]
    save_dict = torch.load(best_model)
    net.load_state_dict(save_dict['net'])
    optimizer.load_state_dict(save_dict['optim'])
    lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
    epoch = save_dict['epoch'] + 1

    print("\nValidation For Experiment: ", exp_dir)
    print(time.strftime('%H:%M:%S', time.localtime()))
    val(epoch)

if __name__ == "__main__":
    main()

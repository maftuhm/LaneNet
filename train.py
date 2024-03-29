import argparse
import json
import os
import shutil
import time

import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import *
import dataset
from model import LaneNet
from utils.transforms import *
from utils.lr_scheduler import PolyLR
from utils.postprocess import embedding_post_process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp13")
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device(exp_cfg['device'])
writer = SummaryWriter(exp_dir + '/log')

# ------------ train data ------------
# # CULane mean, std
# mean=(0.3598, 0.3653, 0.3662)
# std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
transform_train = Compose(Resize(resize_shape), Darkness(5), Rotation(2),
                          ToTensor(), Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
train_dataset = Dataset_Type(Dataset_Path[dataset_name], "train", transform_train)
train_loader = DataLoader(train_dataset, batch_size=exp_cfg['dataset']['batch_size'], shuffle=True, collate_fn=train_dataset.collate, num_workers=0, pin_memory=True)

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
best_val_loss = 1e6


def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()
    train_loss = 0
    train_loss_bin_seg = 0
    train_loss_var = 0
    train_loss_dist = 0
    train_loss_reg = 0

    progressbar = tqdm(range(len(train_loader)))

    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].to(device)
        segLabel = sample['segLabel'].to(device)

        optimizer.zero_grad()
        output = net(img, segLabel)
        embedding = output['embedding']
        binary_seg = output['binary_seg']
        seg_loss = output['seg_loss']
        var_loss = output['var_loss']
        dist_loss = output['dist_loss']
        reg_loss = output['reg_loss']
        loss = output['loss']

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        iter_idx = epoch * len(train_loader) + batch_idx
        train_loss += loss.item()
        train_loss_bin_seg += seg_loss.item()
        train_loss_var += var_loss.item()
        train_loss_dist += dist_loss.item()
        train_loss_reg += reg_loss.item()
        progressbar.set_description("batch loss: {:.5f}".format(loss.item()))
        progressbar.update(1)

        lr = optimizer.param_groups[0]['lr']
        gap_num = 5
        if batch_idx%gap_num == 0 and batch_idx < 50 * gap_num:
                color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8') # bgr
                display_imgs = []
                embedding = embedding.detach().cpu().numpy()
                bin_seg_prob = binary_seg.detach().cpu().numpy()
                bin_seg_pred = np.argmax(bin_seg_prob, axis=1)

                for b in range(len(img)):
                    img_name = sample['img_name'][b]
                    img = cv2.imread(img_name) # BGR
                    img = cv2.resize(img, resize_shape)

                    bin_seg_img = np.zeros_like(img)
                    bin_seg_img[bin_seg_pred[b]==1] = [0, 0, 255]

                    # # ----------- cluster ---------------
                    seg_img = np.zeros_like(img)
                    embedding_b = np.transpose(embedding[b], (1, 2, 0))
                    lane_seg_img = embedding_post_process(embedding_b, bin_seg_pred[b], exp_cfg['net']['delta_v'])
                    embed_unique_idxs = np.unique(lane_seg_img)
                    embed_unique_idxs = embed_unique_idxs[embed_unique_idxs!=0]
                    for i, lane_idx in enumerate(embed_unique_idxs):
                        seg_img[lane_seg_img==lane_idx] = color[i]
                    img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bin_seg_img = cv2.cvtColor(bin_seg_img, cv2.COLOR_BGR2RGB)

                    display_imgs.append(img)
                    display_imgs.append(bin_seg_img)
                display_imgs = np.asarray(display_imgs)
                display_imgs = np.transpose(display_imgs, (0, 3, 1, 2))
                writer.add_images("Train/img_{}".format(batch_idx), display_imgs, epoch)


    progressbar.close()
    writer.add_scalar("Train/loss", train_loss, epoch)
    writer.add_scalar("Train/learning_rate", lr, epoch)
    writer.add_scalars("Train/losses", {'loss_bin_seg':train_loss_bin_seg,
                                        'loss_var':train_loss_var,
                                        'loss_dist':train_loss_dist,
                                        'loss_reg':train_loss_reg}, epoch)
    writer.flush()
    print(
        'Loss: {:.4f}|'
        'Bin seg loss: {:.4f}|'
        'Var loss: {:.4f}|'
        'Dist loss: {:.4f}|'
        'Reg loss: {:.4f}'.format(
            train_loss,
            train_loss_bin_seg,
            train_loss_var,
            train_loss_dist, 
            train_loss_reg
            )
        )

    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))
    print("------------------------")


def val(epoch):
    global best_val_loss

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

            # visualize validation every 5 frame, 50 frames in all
            gap_num = 5
            if batch_idx%gap_num == 0 and batch_idx < 50 * gap_num:
                color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8') # bgr
                display_imgs = []
                embedding = embedding.detach().cpu().numpy()
                bin_seg_prob = binary_seg.detach().cpu().numpy()
                bin_seg_pred = np.argmax(bin_seg_prob, axis=1)

                for b in range(len(img)):
                    img_name = sample['img_name'][b]
                    img = cv2.imread(img_name) # BGR
                    img = cv2.resize(img, resize_shape)

                    bin_seg_img = np.zeros_like(img)
                    bin_seg_img[bin_seg_pred[b]==1] = [0, 0, 255]

                    # # ----------- cluster ---------------
                    seg_img = np.zeros_like(img)
                    embedding_b = np.transpose(embedding[b], (1, 2, 0))
                    lane_seg_img = embedding_post_process(embedding_b, bin_seg_pred[b], exp_cfg['net']['delta_v'])
                    embed_unique_idxs = np.unique(lane_seg_img)
                    embed_unique_idxs = embed_unique_idxs[embed_unique_idxs!=0]
                    for i, lane_idx in enumerate(embed_unique_idxs):
                        seg_img[lane_seg_img==lane_idx] = color[i]
                    img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bin_seg_img = cv2.cvtColor(bin_seg_img, cv2.COLOR_BGR2RGB)

                    display_imgs.append(img)
                    display_imgs.append(bin_seg_img)
                display_imgs = np.asarray(display_imgs)
                display_imgs = np.transpose(display_imgs, (0, 3, 1, 2))
                writer.add_images("Val/img_{}".format(batch_idx), display_imgs, epoch)

            val_loss += loss.item()
            val_loss_bin_seg += seg_loss.item()
            val_loss_var += var_loss.item()
            val_loss_dist += dist_loss.item()
            val_loss_reg += reg_loss.item()

            progressbar.set_description("batch loss: {:.5f}".format(loss.item()))
            progressbar.update(1)

    progressbar.close()
    writer.add_scalar("Val/val_loss", val_loss, epoch)
    writer.add_scalars("Val/losses", {'loss_bin_seg': val_loss_bin_seg,
                                    'loss_var': val_loss_var,
                                    'loss_dist': val_loss_dist,
                                    'loss_reg':val_loss_reg}, epoch)

    writer.flush()

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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        copy_name = os.path.join(exp_dir, '{:.4f}'.format(val_loss) + '_' + exp_dir.split('/')[-1] + '_best.pth')
        shutil.copyfile(save_name, copy_name)
        print("best model is saved: {}".format(copy_name))
    print("------------------------\n\n")


def main():
    global best_val_loss
    if args.resume:
        save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'))
        net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict.get("best_val_loss", 6.671) #my data exp2 41.6771 exp3 42.01 exp4 44.82
    else:
        start_epoch = 0

    for epoch in range(start_epoch, exp_cfg['NUM_EPOCH']):
        train(epoch)
        if epoch % 1 == 0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)


if __name__ == "__main__":
    main()

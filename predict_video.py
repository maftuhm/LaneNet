import argparse
from config import *

from model import LaneNet
from utils.transforms import *
from utils.postprocess import *
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", '-i', type=str, default="C:/Users/uin/Documents/Maftuh Mashuri/Project/videos/data_projects_co_id_02.mp4", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, default="./experiments/exp6/exp6_best.pth", help="Path to model weights")
    parser.add_argument("--band_width", '-b', type=float, default=0.5, help="Value of delta_v")
    parser.add_argument("--output_path", '-o', action="store_true", default="C:/Users/uin/Documents/Maftuh Mashuri/Project/videos", help="Visualize the result")
    args = parser.parse_args()
    return args

args = parse_args()
weight_path = args.weight_path
video_path = args.video_path
output_path = args.output_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
save_dict = torch.load(weight_path, map_location=device)
net.load_state_dict(save_dict['net'])
net = net.to(device)

vidcap = cv2.VideoCapture(video_path)
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
size_video = (1920, 1080)
out = cv2.VideoWriter(os.path.join(output_path,'output_video.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, size_video)

_set = "IMAGENET"
mean = IMG_MEAN[_set]
std = IMG_STD[_set]

transform_img_ori = Resize(size_video)
transform_img = Resize((640, 360))
transform_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform = Compose(transform_img, transform_x)

color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8') # bgr

def predict_image(image_frame):
    img = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB) # RGB for net model input
    img = transform_img({'img': img})['img']
    x = transform_x({'img': img})['img']
    x.unsqueeze_(0)
    x = x.to(device)
    output = net(x)
    embedding = output['embedding']
    embedding = embedding.detach().cpu().numpy()
    embedding = np.transpose(embedding[0], (1, 2, 0))
    binary_seg = output['binary_seg']
    bin_seg_prob = binary_seg.detach().cpu().numpy()
    bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    seg_img = np.zeros_like(img)
    lane_seg_img = embedding_post_process(embedding, bin_seg_pred, args.band_width, 4)

    lanes_loc = []
    for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        if lane_idx==0:
            continue
        seg_img[lane_seg_img == lane_idx] = color[i-1]
        lanes_loc.append(np.where(lane_seg_img == lane_idx))

    lanes_coordinates = []
    for lane_loc in lanes_loc:
        lanes_coordinates.append(list(zip(lane_loc[0], lane_loc[1])))

    seg_img = transform_img_ori({'img': seg_img})['img']
    img_output = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=image_frame, beta=1., gamma=0.)
    return img_output

def main():

    progressbar = tqdm(range(total_frames))

    net.eval()
    with torch.no_grad():

        success, image = vidcap.read()

        while success:
            pred = predict_image(image)
            out.write(pred)

            success, image = vidcap.read()
            progressbar.update(1)

    progressbar.close()
    out.release()

if __name__ == "__main__":
    main()
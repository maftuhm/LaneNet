import argparse
from config import *

from model import LaneNet
from utils.transforms import *
from utils.postprocess import *
from utils.prob2lines import getLane
import os, json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_src", '-v', type=str, help="Path to video")
    parser.add_argument("--weight_path", '-w', type=str, help="Path to model weights")
    parser.add_argument("--band_width", '-b', type=float, default=1.5, help="Value of delta_v")
    parser.add_argument("--line", '-l', type=str, default='dot', help="Kind of line segmentation or dots")
    parser.add_argument("--output_path", '-o', type=str, help="Path to output result")
    args = parser.parse_args()
    return args

args = parse_args()
weight_path = args.weight_path
kind_line = args.line
video_src = args.video_src
video_name = os.path.split(video_src)[-1].split('.')[0]
output_path = args.output_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
save_dict = torch.load(weight_path, map_location=device)
net.load_state_dict(save_dict['net'])
net = net.to(device)

vidcap = cv2.VideoCapture(video_src)
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
size_video = (1920, 1080)
out = cv2.VideoWriter(os.path.join(output_path, video_name + '_output.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 30, size_video)

_set = "IMAGENET"
mean = IMG_MEAN[_set]
std = IMG_STD[_set]

transform_img_ori = Resize(size_video)
transform_img = Resize((640, 360))
transform_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform = Compose(transform_img, transform_x)

color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8') # bgr

dump_to_json = []

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
    lane_coords = getLane.polyfit2coords_tusimple(lane_seg_img, resize_shape=(size_video[1], size_video[0]), y_px_gap=10, pts=56)
    for i in range(len(lane_coords)):
        lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])

    json_dict = {}
    json_dict['lanes'] = []
    json_dict['h_sample'] = []
    json_dict['raw_file'] = '' # video_src
    json_dict['run_time'] = 0
    for l in lane_coords:
        if len(l) == 0:
            continue
        json_dict['lanes'].append([])
        for (x, y) in l:
            json_dict['lanes'][-1].append(int(x))
    for (x, y) in lane_coords[0]:
        json_dict['h_sample'].append(y)

    lanes_loc = []
    for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        if lane_idx==0:
            continue
        seg_img[lane_seg_img == lane_idx] = color[i-1]
        lanes_loc.append(np.where(lane_seg_img == lane_idx))

    lanes_coordinates = []
    for lane_loc in lanes_loc:
        lanes_coordinates.append(list(zip(lane_loc[0], lane_loc[1])))

    if kind_line == 'dot':
        for l in lane_coords:
            l = [(x, y) for (x, y) in l if x >= 0 and y >= 0]
            for pt in l:
                cv2.circle(img, pt, radius=5, color=(0, 0, 255), thickness=-1)
        img_output = img

    elif kind_line == 'seg':
        # seg_img = transform_img_ori({'img': seg_img})['img']
        img_output = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

    else:
        img_output = img

    return img_output, json.dumps(json_dict)

def main():

    progressbar = tqdm(range(total_frames))
    dump_to_json = []
    net.eval()

    with torch.no_grad():

        success, image = vidcap.read()

        while success:
            pred, json_output = predict_image(image)
            out.write(pred)
            dump_to_json.append(json_output)

            success, image = vidcap.read()
            progressbar.update(1)

    progressbar.close()
    out.release()

    with open(os.path.join(output_path, video_name + '_output.json'), 'w') as f:
        for line in dump_to_json:
            print(line, end="\n", file=f)

if __name__ == "__main__":
    main()

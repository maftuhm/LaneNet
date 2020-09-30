import argparse
import json
import os
import glob
import numpy as np
import time
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

class CreateLane:
    def __init__(self, lane):
        self.lane = lane

    def clear(self, lane):
        lane = set(lane)
        self.lane = list(lane)
        return self.lane

    def sort(self, lane):
        self.lane = sorted(lane, key = lambda y: y[1])
        return self.lane

    def interpolate(self, lane):
        lane = self.clear(lane)
        lane = self.sort(lane)

        new_lane = []
        for i in range(len(lane) - 1):
            difx = abs(lane[i][0] - lane[i+1][0])
            dify = abs(lane[i][1] - lane[i+1][1])
            if difx > dify:
                dif = difx
            else:
                dif = dify

            x = np.linspace(lane[i][0], lane[i+1][0], dif, dtype = np.int32)
            y = np.linspace(lane[i][1], lane[i+1][1], dif, dtype = np.int32)
            new_lane.extend(list(zip(x.tolist(), y.tolist())))

        lane = new_lane
        lane = self.clear(lane)
        lane = self.sort(lane)
        return lane

    def clear_ypoint(self, lane):
        points = []
        for i, point in enumerate(lane[:-1]):
            if lane[i][1] != lane[i+1][1]:
                points.append(point)
        points.append(lane[-1])
        self.lane = points
        return self.lane

    def clear_step(self, lane, point = 'y', size = (1920, 1080), step = 10):
        if point == 'y':
            param = size[1]
        else:
            return False

        self.lane = [(x, y) for (x, y) in lane if y in range(0, param, step)]
        return self.lane
    
    def clear_duplicate(self, lane, point = 'y'):
        points = []

        for i, point in enumerate(lane[:-1]):
            if lane[i][1] != lane[i+1][1]:
                points.append(point)

        points.append(lane[-1])
        self.lane = points
        return self.lane

    def get(self):
        self.interpolate(self.lane)
        self.clear_step(self.lane)
        self.clear_duplicate(self.lane)
        return self.lane

    def get_split(self):
        lane = self.get()
        x, y = [a[0] for a in lane], [a[1] for a in lane]
        return x, y

class DictLanes(CreateLane):
    dict_lanes = dict()
    def __init__(self, lanes, index = 0):
        super(CreateLane, self).__init__()
        self.lanes = lanes
        self.lane = lanes[index]

    def min_max(self, lanes, point = 'y', size = (1920, 1080), step = 10):
        width, height = 1920, 1080
        min_y = [l[0][1] for l in lanes]
        max_y = [l[len(l)-1][1] for l in lanes]
        min_y, max_y = min(min_y), max(max_y)
        return (min_y, max_y)
        
    def get_lanes(self):
        new_lanes = []
        for lane in self.lanes:
            new_lane = CreateLane(lane).get()
            new_lanes.append(new_lane)
        self.lanes = new_lanes
        return self.lanes

    def get(self):
        lanes = self.get_lanes()
        min_y, max_y = self.min_max(lanes)

        h_samples = list(range(min_y, max_y + 1, 10))
        new_lanes = []
        for lane in lanes:
            new_lane = []
            i = 0
            for h in h_samples:
                if h in [l[1] for l in lane]:
                    new_lane.append(lane[i][0])
                    i += 1
                else:
                    new_lane.append(-2)
            new_lanes.append(new_lane)
        self.dict_lanes['lanes'] = new_lanes
        self.dict_lanes['h_samples'] = h_samples
        return self.dict_lanes

class JsonLanes:
    def __init__(self, src_dir):
        self.src_dir = src_dir

    def get(self):
        with open(self.src_dir) as f:
            json_file = json.load(f)
        data_lanes = []
        for index in json_file:
            data = {}
            regions = json_file[index]['regions']
            filename = json_file[index]['filename']
            lanes = []
            if len(regions) == 0:
                print("Image " + filename + " is not anotated.")
                print(json_file[index])
                break
            else:
                for lane in regions:
                    x = lane['shape_attributes']['all_points_x']
                    y = lane['shape_attributes']['all_points_y']
                    lane = [(a, b) for (a, b) in zip(x, y)]
                    lanes.append(lane)

            dict_lanes = DictLanes(lanes).get()
            data['lanes'] = dict_lanes['lanes']
            data['h_samples'] = dict_lanes['h_samples']
            data['raw_file'] = 'clips/' + filename.split('.')[0] + '/20.jpg'
            data_lanes.append(data)
        return data_lanes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--save_samples", type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    json_path = args.json_path
    image_path = '/'.join(os.path.split(json_path)[:-1])
    DATA = JsonLanes(args.json_path).get()

    with open(image_path + '/label_lanenet_' + time.strftime('%d%m%Y_%H%M', time.localtime()) + '.json', 'w') as json_file:
        for line in DATA:
            json_file.write(json.dumps(line) + '\n')

    if args.save_samples:

        samples_dir = image_path + '/samples'
        os.makedirs(samples_dir, exist_ok=True)

        for image_label in tqdm(DATA):
            gt_lanes = image_label['lanes']
            y_samples = image_label['h_samples']
            raw_file = image_label['raw_file']

            lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
            img_raw = raw_file.split('/')[-2]
            img = plt.imread(image_path + '/' + raw_file)

            for lane in lanes_vis:
                for pt in lane:
                    cv2.circle(img, pt, radius=5, color=(0, 255, 0), thickness=-1)

            plt.imsave(samples_dir + '/' + img_raw +'.jpg', img)

if __name__ == '__main__':
    main()
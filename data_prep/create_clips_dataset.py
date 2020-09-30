import argparse
import json
import os
import os.path as ops

import cv2

def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='Path to store dataset')
    parser.add_argument('--video_path', type=str, help='Path to video source')
    parser.add_argument('--frame_per_sec', '-fps', type=int, default=30, help='Set how many frame per second')
    parser.add_argument('--frame_per_dir', '-fpd', type=int, default=20, help='Set how many frame per directory')

    return parser.parse_args()

args = init_args()
src_dir = args.src_dir
video_path = args.video_path

fps = args.frame_per_sec
fpd = args.frame_per_dir

def convert_video_frame():

    clips_path = ops.join(src_dir, 'clips') # untuk dataset
    labeling_path = ops.join(src_dir, 'labeling') # untuk melebeli dengan vgg anotator supaya lebih mudah
    os.makedirs(clips_path, exist_ok=True)
    os.makedirs(labeling_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, int(1000/fps))    
    success, image = vidcap.read()

    clip_dir_num = int(60 * fpd/fps)

    count, i = 0, 1

    while success:

        if count % fpd == 0:

            dir_clip_name = os.path.join(clips_path, str(clip_dir_num))
            os.makedirs(dir_clip_name, exist_ok=True)
            labeling_name = ops.join(labeling_path, str(clip_dir_num) + '.jpg')
            cv2.imwrite(labeling_name, image)

            clip_dir_num += int(60 * fpd/fps)
            i = 1

        image_name_path = ops.join(dir_clip_name, str(i) + '.jpg')
        cv2.imwrite(image_name_path, image)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count + 1) * int(1000/fps))
        success, image = vidcap.read()

        print('Read a new frame:', image_name_path, success)

        i +=1
        count += 1

if __name__ == '__main__':

    convert_video_frame()
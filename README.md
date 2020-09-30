# LaneNet lane detection in Pytorch

LaneNet is a segmentation-tasked lane detection algorithm, described in [1] "[Towards end-to-end lane detection: an instance segmentation approach](https://arxiv.org/pdf/1802.05591.pdf)" . The key idea of instance segmentation should be referred to [2] "[Semantic instance segmentation with a discriminative loss function](https://arxiv.org/pdf/1708.02551.pdf)". This repository contains a re-implementation in Pytorch.

## Demo Test

For single image demo test:

```Bash
python demo_test.py -i path/to/img 
                    -w path/to/weight
                    -b band_width
                    [--visualize / -v]
```
Contoh:
```Bash
python demo_test.py -i demo/demo.jpg -w experiments/exp10/exp10_best.pth -b 0.5 [--visualize / -v]
```

![](demo/demo_result.jpg "demo_result")

## Predict Video

For single image demo test:

```Bash
python predict_video.py -v path/to/video 
                    -w path/to/weight
                    -b band_width
                    -o path/to/folder/output
```
Contoh:
```Bash
python predict_video.py -i demo/demo.mp4 -w experiments/exp10/exp10_best.pth -b 0.5 -o experiments/exp10/
```

![](demo/demo_video_result.mp4 "demo_result")

## Persiapan data

### Membuat dataset sendiri
Dataset yang dibuat mengikuti format tusimple ya itu sebagai berikut
```
My_dataset_path
├── clips
├── label_train.json
├── label_val.json
└── label_test.json
```
#### Membuat dataset dari video

```Bash
python create_clips_dataset.py 	--src_dir path/to/store/dataset
			                    --video_path path/to/video/source
                    			-fps 30 (setting berapa frame yg akan diambil per detik)
                    			-fpd 20 (setting berapa frame yg akan disimpan per folder)
```
Contoh:
```Bash
python create_clips_dataset.py --src_dir data/project_data --video_path /data/documents/video.mp4 -fps 30 -fpd 20
```

Hasil outputnya adalah
```
My_dataset_path
├── clips
└── labelling
```
Berikutnya adalah melabeli dengan vgg anotator, dapat di download di "[VGG Image Annotator (VIA)
](http://www.robots.ox.ac.uk/~vgg/software/via)". Kemudian simpan semua label `json` di folder `My_dataset_path`.

**Note**
- Nama folder `My_dataset_path` boleh diganti apa saja sesuai keinginan/keperluan.

#### Konvert label.json ke format label tusimple

```
Kode dan file python menyusul
```

## Train 

1. Specify an experiment directory, e.g. `experiments/exp0`.  Assign the path to variable `exp_dir` in `train.py`.

2. Modify the hyperparameters in `experiments/exp0/cfg.json`.

3. Start training:

   ```python
   python train.py [-r]
   ```

4. Monitor on tensorboard:

   ```Bash
   tensorboard --logdir experiments/exp0/log
   ```
   load multiple log tensorboard
   ```Bash
   tensorboard --logdir_sec exp0:experiments/exp0/log,exp1:experiments/exp1/log,exp2:experiments/exp2/log,...
   ```


## Reference

[1]. Neven, Davy, et al. "[Towards end-to-end lane detection: an instance segmentation approach.](https://arxiv.org/pdf/1802.05591.pdf)" *2018 IEEE Intelligent Vehicles Symposium (IV)*. IEEE, 2018.

[2]. De Brabandere, Bert, Davy Neven, and Luc Van Gool. "[Semantic instance segmentation with a discriminative loss function.](https://arxiv.org/pdf/1708.02551.pdf)" *arXiv preprint arXiv:1708.02551* (2017).


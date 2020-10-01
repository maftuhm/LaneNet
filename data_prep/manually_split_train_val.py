import json, os
import random, time

index_div = [(0, 80), (80, 140), (140, 175), (175, 310), (310, 395)]

random.seed(0)
index_split = []
for div in index_div:
    d = list(range(div[0], div[1]))
    index_split.append(random.sample(d, len(d)))


train_data_index, val_data_index, test_data_index = [], [], []

for div in index_split:
    train = div[:int(len(div)*0.75)]
    val = div[int(len(div)*0.75):]
#     test = div[int(len(div)*0.8):]
    train_data_index.extend(train)
    val_data_index.extend(val)

print("Train:", len(train_data_index))
print("Val:", len(val_data_index))
print("Total:", len(train_data_index) + len(val_data_index))

TRAIN_DATA, VAL_DATA = [], []
for i in train_data_index:
    TRAIN_DATA.append(DATA[i])
for i in val_data_index:
    VAL_DATA.append(DATA[i])

tm = time.strftime('%d%m%Y', time.localtime())

with open(image_path + '/train_label_lanenet_' + tm + '.json', 'w') as json_file:
    for line in TRAIN_DATA:
        json_file.write(json.dumps(line) + '\n')

with open(image_path + '/val_label_lanenet_' + tm + '.json', 'w') as json_file:
    for line in VAL_DATA:
        json_file.write(json.dumps(line) + '\n')
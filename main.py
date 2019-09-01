import datetime
import os
import random

from sklearn.metrics import f1_score

from data_preparation import generate_arrays
from process_results import saveResult
from unet import unet
from helpers import save_testing_images

x_unet = 256
y_unet = 256

data_path = '../scratch/liver-data/py_wsi/db/'
images_path = 'images-3/'
masks_path = 'masks-3/'

QUICK_TEST = False
UNET_X = 256
UNET_Y = 256
FILTER_NO = 32
OPTIMIZER = 'adadelta'
LOSS = 'dice_coef'
METRICS = ['accuracy']
VALIDATION_SPLIT = 0.2
SHUFFLE = True
BATCH_SIZE = 32
EPOCHS = 300
timestamp = datetime.datetime.now()
timestamp = timestamp.strftime('%Y-%m-%dT%H:%M:%S')

image_names = sorted(next(os.walk('{}{}'.format(data_path, images_path)))[2])
mask_names = sorted(next(os.walk('{}{}'.format(data_path, masks_path)))[2])

zipped_list = list(zip(image_names, mask_names))
random.shuffle(zipped_list)
shuffled_image_names, shuffled_mask_names = zip(*zipped_list)
shuffled_image_names = [*shuffled_image_names]
shuffled_mask_names = [*shuffled_mask_names]

training_amout_factor = 0.8
train_set_amount = int(len(image_names) * training_amout_factor)
print('blabla')
# print('All image count: {}'.format(len(image_names))
print('All image count: {}'.format(len(image_names)))
print('Train set amount: {}'.format(train_set_amount))
test_set_amount = int(len(image_names) * (1 - training_amout_factor))

train_image_names = shuffled_image_names[:train_set_amount]
train_mask_names = shuffled_mask_names[:train_set_amount]

test_image_names = shuffled_image_names[train_set_amount:]
test_mask_names = shuffled_mask_names[train_set_amount:]

images_train, masks_train = generate_arrays(train_image_names, train_mask_names, 256, 256,
                                            '{}{}'.format(data_path, images_path), '{}{}'.format(data_path, masks_path),
                                            quick_test=QUICK_TEST)

model = unet(UNET_X, UNET_Y, FILTER_NO, optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

results = model.fit(images_train, masks_train, validation_split=VALIDATION_SPLIT, shuffle=SHUFFLE,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS)

images_test, masks_test = generate_arrays(test_image_names, test_mask_names, 256, 256,
                                          '{}{}'.format(data_path, images_path), '{}{}'.format(data_path, masks_path),
                                          quick_test=QUICK_TEST)

os.mkdir('data/results/{}'.format(timestamp))
save_testing_images(images_test, timestamp, image_name='test_img')
save_testing_images(masks_test, timestamp, image_name='mask_ground_truth')
results = model.predict(images_test)

f1_sum = 0
count = 0
for idx, single_result in enumerate(results):
    single_result = single_result.reshape(65536, ) * 255
    single_result = single_result.astype(int)
    gt_mask = masks_test[idx].reshape(65536,) *255
    gt_mask = gt_mask.astype(int)
    f1_sum +=f1_score(gt_mask, single_result, average='macro')
    count = idx

print(f1_sum/count)

saveResult("data/results/{}".format(timestamp), results)

print("end")

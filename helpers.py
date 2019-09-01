import getopt
import json
import os
import sys
from matplotlib import pyplot as plt

import numpy


# from data_preparation import trainGenerator


def parse_arguments(argv):
    REGENERATE_DATA = False
    QUICK_TEST = False
    argv = argv[1:]
    try:
        opts, args = getopt.getopt(argv, "rq", [])
    except getopt.GetoptError:
        print('test.py [-r --regenerate, -q --quick]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py [-r --regenerate, -q --quick]')
            sys.exit()
        elif opt in ("-r", "--regenerate"):
            REGENERATE_DATA = True
        elif opt in ("-q", "--quick"):
            QUICK_TEST = True
    print("Quicktest: {} \nRegeneration of data: {}".format(QUICK_TEST, REGENERATE_DATA))
    return REGENERATE_DATA, QUICK_TEST


def save_testing_images(data_test, timestamp, image_name='test'):
    for idx, test_image in enumerate(data_test):
        test_image = test_image.reshape(256, 256)

        plt.imsave('data/results/{}/{}_{}.png'.format(timestamp, idx, image_name), test_image, cmap='gray')


def extract_validation_sample(results, timestamp):
    sample = results[1, :, :].reshape(256, 256)
    mask_pred = (sample > 0.5).astype(numpy.uint8)
    plt.imsave('./result-validation-{}'.format(timestamp), mask_pred, cmap='gray')


def load_set_configuration_from_json():
    with open('data/dog/all/sets.json', 'r', encoding='utf-8') as file:
        set_configuration = json.load(file)
    return set_configuration

# convert to 256
# for file in convert_data('data/dog/original/Images/Normal',next(os.walk('data/dog/original/Images/Normal'))[2]):
#    plt.imsave('data/dog/original/Images-256/Normal/crop_{}.png'.format(file.filename), file, cmap='gray')

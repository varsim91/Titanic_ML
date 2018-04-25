import configparser
import os

parser = configparser.ConfigParser()
parser.read([os.path.dirname(__file__) + r"\..\config.ini"])

_train_images_loc = parser.get("FILE_LOCATIONS", "train_images")
_train_labels_loc = parser.get("FILE_LOCATIONS", "train_image_labels")

_test_images_loc = parser.get("FILE_LOCATIONS", "test_images")
_sample_submission_loc = parser.get("FILE_LOCATIONS", "sample_submission")


def get_train_image_loc():
    return _train_images_loc


def get_train_labels_loc():
    return _train_labels_loc


def get_test_image_loc():
    return _test_images_loc


def get_sample_submission_loc():
    return _sample_submission_loc

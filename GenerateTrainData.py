from LoadImagesLabels import *

TRAIN_DIR = 'dataset/train/'
is_train_data = True

gen_data = LoadImagesLabels()
gen_data.load_data(TRAIN_DIR, True)
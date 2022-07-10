import numpy as np
from LoadImagesLabels import *
import os
import cv2

TRAIN_DIR = 'dataset/train'
TRAIN_IMGS = 'flattened_images.txt'
TRAIN_LABELS = 'final_labels.txt'
TEST_DIR = 'dataset/test/'
is_train_data = False


class TrainAndTest:
    train_images = None
    train_labels = None
    test_images = LoadImagesLabels()
    knn = cv2.ml.KNearest.create()
    predicted_label = ''

    def load_train_data(self):
        self.train_images = np.loadtxt(TRAIN_IMGS, np.float32)
        self.train_labels = np.loadtxt(TRAIN_LABELS, np.float32)
        # self.train_labels = self.train_labels.reshape((self.train_labels.size, 1))

    def process_test_images(self, path):
        self.test_images.load_data(path, is_train_data)

    def train_model(self):
        self.knn.train(self.train_images, cv2.ml.ROW_SAMPLE, self.train_labels)

    def find_nearest_neighbor(self):
        # float_test_images = np.float32(self.test_images.flattened_images)
        float_test_images = self.test_images.flattened_images
        for test_img in float_test_images:
            # print(float_test_images.dtype())
            retval, results, neigh_resp, dists = self.knn.findNearest(test_img, k=1)
            self.find_pred_label(results[0][0])

    def find_pred_label(self, float_label):
        int_label = int(float_label)
        filenames = os.listdir(TRAIN_DIR)
        for filename in filenames:
            if filename.find(int_label + '_') != -1:
                self.predicted_label = filename[filename.find('_') + 1 : len(filename) - 4]
                print(self.predicted_label)


# =============================================================================

def main():
    knn_data = TrainAndTest()
    knn_data.load_train_data()
    print('done loading train data...')
    knn_data.process_test_images(TEST_DIR)
    print('done processing test images...')
    knn_data.train_model()
    knn_data.find_nearest_neighbor()
    print('done training model...')


if __name__ == '__main__':
    main()

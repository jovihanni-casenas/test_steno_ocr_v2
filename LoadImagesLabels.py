import cv2
import numpy as np
import os


TRAIN_DIR = 'dataset/train/'
TEST_DIR = 'dataset/test/'


class LoadImagesLabels:

    flattened_images = []
    labels = []
    new_width = 30
    new_height = 30
    min_contour_area = 10

    # def union(rects):
    #     rects_len = range(len(rects))
    #     x = min(rects[i][0] for i in rects_len)
    #     y = min(rects[i][1] for i in rects_len)
    #     w = max(rects[i][0] + rects[i][2] for i in rects_len) - x
    #     h = max(rects[i][1] + rects[i]
    #     [3] for i in rects_len) - y
    #     return (x, y, w, h)

    def load_images(self, path):
        self.flattened_images = np.empty((0, self.new_width * self.new_height))
        files = os.listdir(path)
        for file_name in files:
            file_name = path + file_name
            read_img = cv2.imread(file_name)
            img_gray = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)
            img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
            img_thresh = cv2.adaptiveThreshold(img_blurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)
            # cv2.imshow("img_thresh", img_thresh)

            img_thresh_copy = img_thresh.copy()
            contours, heirarchy = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects = []
            for contour in contours:
                col = []
                if cv2.contourArea(contour) > self.min_contour_area:
                    [x, y, width, height] = cv2.boundingRect(contour)
                    col.append(x)
                    col.append(y)
                    col.append(width)
                    col.append(height)
                    rects.append(col)
                # end if

            rects_len = range(len(rects))
            x = min(rects[i][0] for i in rects_len)
            y = min(rects[i][1] for i in rects_len)
            w = max(rects[i][0] + rects[i][2] for i in rects_len) - x
            h = max(rects[i][1] + rects[i][3] for i in rects_len) - y

            # attach bounding box to character
            cv2.rectangle(read_img,  # draw rectangle on original training image
                          (x, y),  # upper left corner
                          (x + w, y + h),  # lower right corner
                          (0, 0, 255),  # red
                          2)

            cropped_img = img_thresh[y:y+h, x:x+w]
            # must resize image to make all images uniform
            cropped_resized_img = cv2.resize(cropped_img, (30, 30))
            flatten_img = cropped_resized_img.reshape((1, 30 * 30))
            self.flattened_images = np.append(self.flattened_images, flatten_img, 0)

            self.load_labels(file_name)

            # cv2.imshow('box character', read_img)
            # cv2.imshow('cropped image', cropped_img)
            # cv2.waitKey(0)

            # end for
        # end for


        float_labels = np.array(self.labels, np.float32)
        final_labels = float_labels.reshape((float_labels.size, 1))

        np.savetxt('flattened_images.txt', self.flattened_images)
        np.savetxt('final_labels.txt', final_labels)

        # cv2.destroyAllWindows()
    # end load_images



    def load_labels(self, file_name):
        print(str(file_name[len(TRAIN_DIR):file_name.find('_')]))
        self.labels.append(file_name[len(TRAIN_DIR):file_name.find('_')])
    # end load_labels





#=====================================================================

def main():
    train = LoadImagesLabels()
    train.load_images(TRAIN_DIR)
    print('Loaded all test data...')
    # test = LoadImagesLabels()
    # test.load_images(TEST_DIR)
    # print('Loaded all train data...')


if __name__ == '__main__':
    main()
import os
import cv2

IN_DIR = 'C:/Users/Admin/Documents/BISU files per sem/Year 4, Sem 1/Thesis/train_old_dataset_unedited/'
OUT_DIR = 'dataset/train/'
filenames = os.listdir(IN_DIR)

i = 0

for filename in filenames:
    img = cv2.imread(IN_DIR + filename)
    cv2.imwrite(OUT_DIR + str(i) + '_' + filename)
    i += 1

# still have not checked if this works as intended
# rewritten the code kay na delete ag orig huhu

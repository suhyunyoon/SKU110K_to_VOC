import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt

import glob
import os

class ItemLoader:
    def __init__(self, img_dir='./items'):
        self.img_dir = img_dir
        self.img_list = glob.glob(img_dir + '/*.png')
        self.label = []

        # read img
        self.img = []
        self.aspect_ratio = []
        for i, v in enumerate(self.img_list):
            temp = Image.open(v).convert('RGBA')
            # remove transparency area
            temp = temp.crop(temp.getbbox())
            self.img.append(temp)
            # calc aspect ratio
            x, y = temp.size
            self.aspect_ratio.append(x/y)

        self.aspect_ratio = np.array(self.aspect_ratio)
        self.num_img = len(self.img)
        # set max min aspect ratio
        #self.max_ratio_index = np.argmax(self.aspect_ratio)
        #self.min_ratio_index = np.argmin(self.aspect_ratio)

        # count items used
        self.cnt = np.zeros(self.num_img)

        # make label
        self.label = []
        for name in self.img_list:
            l = os.path.split(name)[1].replace('.png', '')
            self.label.append(l)

    # generate Label txt from images
    def make_label(self):
        with open('dataset/labels.txt', 'w') as f:
            for name in self.label:
                f.write(name+'\n')

    # get N random images
    def get_random_images(self, num=1):
        ret = []
        index = np.random.choice(len(self.img), num, replace=False)
        for i in index:
            ret.append(self.img[i])
        return ret

    # get similar random image by aspect ratio
    def get_index_by_similar_ratio(self, ratio):
        index = np.argmin(np.abs(self.aspect_ratio - ratio))
        self.cnt[index] += 1
        return index

    # get image index by label name
    def get_index_by_label(self, name):
        return self.label.index(name)

    def get_img(self, index=None):
        if index:
            return self.img[index].copy()
        else:
            return self.img.copy()

    def get_label(self, index):
        return self.label[index]

    def get_aspect_ratio(self, index):
        return self.aspect_ratio[index]


if __name__ == '__main__':
    loader = ItemLoader()
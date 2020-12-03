from item_loader import ItemLoader

import glob
import os
import shutil
from xml.etree.ElementTree import Element, SubElement, ElementTree

import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt


def showimg(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()


class SKU110KSampler:
    def __init__(self, dir='./SKU110K_fixed/', num_files=100, skip_p=0.0, val_p=0.1, test_p=0.3):
        self.dir = dir if dir[-1] != '/' else dir+'/'
        self.num_files = num_files
        self.skip_p = skip_p
        self.val_p = val_p
        self.test_p = test_p

        # Init ItemLoader
        self.loader = ItemLoader()

        # new annotation
        #self.new_annotations = pd.DataFrame([], columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'name', 'width', 'height'])
        self.new_annotations = []

        # Get image list
        print('Sampling image lists...')
        img_list = glob.glob(self.dir + 'images/*.jpg')

        # Make image file name list
        self.random_list = []
        # random index
        for i in np.random.choice(len(img_list), self.num_files, replace=False):
            self.random_list.append(os.path.split(img_list[i])[1])

        # Get annotations
        print('Making annotations...')
        path = self.dir + 'annotations/'
        # read csv
        df = pd.concat((pd.read_csv(path + 'annotations_train.csv', sep=',', header=None),
                        pd.read_csv(path + 'annotations_val.csv', sep=',', header=None),
                        pd.read_csv(path + 'annotations_test.csv', sep=',', header=None)))
        df.columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'name', 'width', 'height']

        # inner join with img_list
        self.annotations = pd.merge(df, pd.DataFrame(self.random_list, columns=['filename']), left_on='filename', right_on='filename', how='inner')

    # annotation of overlapped item
    def generate_new_annotation(self, filename, xmin, ymin, xmax, ymax, width, height):
        item = self.loader.get_index_by_similar_ratio((xmax-xmin)/(ymax-ymin))
        ratio = self.loader.get_aspect_ratio(item)
        return [filename, xmin, ymin, xmin+int(ratio * (ymax-ymin)), ymax, self.loader.get_label(item), width, height]

    def generate_img(self, img_file, filename):
        # open image file
        bg = Image.open(self.dir + 'images/' + img_file)
        # segments of the img_file
        seg = self.annotations.loc[self.annotations['filename'] == img_file]
        '''
        # get random image
        # 몇개뽑을지 랜덤도 가중치 둬야함
        rand_num = np.random.randint(1, self.loader.num_img)
        imgs = self.loader.get_random_images(num=rand_num)
        num_imgs = len(imgs)
        '''
        # skip segments
        num_seg = len(seg)
        seg = seg.iloc[np.random.choice(num_seg, int(num_seg*(1.0 - self.skip_p)), replace=False)]

        new_annotation = [self.generate_new_annotation(filename, row[0], row[1], row[2], row[3], row[4], row[5])
                                for row in seg[['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']].to_numpy()]
        self.new_annotations += new_annotation

        for a in new_annotation:
            img = self.loader.get_img(self.loader.get_index_by_label(a[5]))
            img = img.resize((a[3]-a[1], a[4]-a[2]))

            # overlay on background img
            bg.paste(img, (a[1], a[2]), img)

        return bg
        '''
        # for segments
        for i, row in seg.iterrows():
            # check skip
            if np.random.rand() >= self.skip_p:
                # choose a item
                rand_i = np.random.randint(num_imgs)
                img_ = imgs[rand_i]
                # size
                x, y = img_.size
                new_y = row['ymax'] - row['ymin']
                new_size = (int(x / y * new_y), new_y)
                # resize into exist annotation
                img_ = img_.resize(new_size)

                # overlay on background img
                bg.paste(img_, (row['xmin'], row['ymin']), img_)

                # generate new annotations
                self.new_annotations = pd.concat((self.new_annotations, row))
        '''

    # generate merged image bg with items
    def generate_img_dataset(self, dir='images/'):
        cnt = 0
        num_list = len(self.random_list)
        for img in self.random_list:
            filename = '{}{}.jpg'.format(dir, cnt)
            img_ = self.generate_img(img, filename)
            img_.save(filename)
            cnt += 1
            if cnt % (num_list // 10) == 0:
                print('{}/{} Image Generated.'.format(cnt, num_list))

    def generate_xml(self):
        print('Making XML annotations...')
        # make ann by img file
        ann_dict = {}
        for row in self.new_annotations:
            f = os.path.split(row[0])[1][:-4]
            if f not in ann_dict:
                ann_dict[f] = []
            ann_dict[f].append(row[1:])
        # make xml
        for x in ann_dict:
            root = Element('annotation')
            SubElement(root, 'folder').text = 'images'
            SubElement(root, 'filename').text = x + '.jpg'
            SubElement(root, 'path').text = './images/' + x + '.jpg'
            source = SubElement(root, 'source')
            SubElement(source, 'database').text = 'Unknown'

            size = SubElement(root, 'size')
            SubElement(size, 'width').text = str(ann_dict[x][0][5])
            SubElement(size, 'height').text = str(ann_dict[x][0][6])
            SubElement(size, 'depth').text = '3'

            SubElement(root, 'segmented').text = '0'

            for row in ann_dict[x]:
                obj = SubElement(root, 'object')
                SubElement(obj, 'name').text = row[4]
                SubElement(obj, 'pose').text = 'Unspecified'
                SubElement(obj, 'truncated').text = '0'
                SubElement(obj, 'difficult').text = '0'
                bbox = SubElement(obj, 'bndbox')
                SubElement(bbox, 'xmin').text = str(row[0])
                SubElement(bbox, 'ymin').text = str(row[1])
                SubElement(bbox, 'xmax').text = str(row[2])
                SubElement(bbox, 'ymax').text = str(row[3])

            tree = ElementTree(root)
            tree.write('dataset/annotations/' + x + '.xml')

    def generate_ids(self, filename, start, end):
        with open(filename, 'w') as f:
            for i in range(start, end):
                f.write(str(i)+'\n')

    # save XML from annotations
    def generate_annotations(self):
        # make directory
        dirs = ['dataset', 'dataset/annotations', 'dataset/dataset_ids']
        for d in dirs:
            # remove exist dir
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.mkdir(d)
        # make label txt
        self.loader.make_label()
        # split data
        val = int((1.0-self.val_p-self.test_p) * len(self.random_list))
        test = int((1.0-self.test_p) * len(self.random_list))
        # make dataset_ids
        print('Making dataset ids...')
        self.generate_ids('dataset/dataset_ids/train.txt', 0, val)
        self.generate_ids('dataset/dataset_ids/trainval.txt', 0, test)
        self.generate_ids('dataset/dataset_ids/val.txt', val, test)
        self.generate_ids('dataset/dataset_ids/test.txt', test, len(self.random_list))
        # make XML
        self.generate_xml()

        return dir

if __name__ == '__main__':
    sampler = SKU110KSampler(num_files=1000, skip_p=0.5)
    sampler.generate_img_dataset()
    sampler.generate_annotations()

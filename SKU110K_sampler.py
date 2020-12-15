from item_loader import ItemLoader

import glob
import os
import shutil
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree

import pandas as pd
import numpy as np
from skimage.util import random_noise
import cv2
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt


def showimg(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def show_annotation(index=None, color=(0,255,0)):
    for num in index:
        img = cv2.cvtColor(cv2.imread('images/{}.jpg'.format(num)), cv2.COLOR_BGR2RGB)
        tree = ET.parse('dataset/annotations/{}.xml'.format(num))
        root = tree.getroot()
        for box in root[6:]:
            img = cv2.rectangle(img, (int(box[4][0].text), int(box[4][1].text)), (int(box[4][2].text), int(box[4][3].text)), color, 2)

        plt.axis("off")
        plt.imshow(img)
        plt.show()

# 잘못된 xml 수정 (임시)
def fix_error():
    index = glob.glob('dataset/annotations/*.xml')
    for idx, f in enumerate(index):
        tree = ET.parse(f)
        root = tree.getroot()
        remove_list = []
        for box in root[6:]:
            for i in range(4):
                if int(box[4][i].text) < 0:
                    box[4][i].text = '0'
                if int(box[4][i].text) >= 640:
                    box[4][i].text = '639'
            if int(box[4][0].text) >= int(box[4][2].text) or int(box[4][1].text) >= int(box[4][3].text):
                remove_list.append(box)
        for i in remove_list[::-1]:
            root.remove(i)
        tree.write(f)


class SKU110KSampler:
    def __init__(self, dir='./SKU110K_fixed/', num_files=100, patch_size=None, skip_p=0.5, val_p=0.1, test_p=0.1):
        self.dir = dir if dir[-1] != '/' else dir+'/'
        self.num_files = num_files
        self.patch_size = patch_size
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
    def generate_new_annotation(self, filename, xmin, ymin, xmax, ymax, width, height, patch_size=None):
        item = self.loader.get_index_by_similar_ratio((xmax-xmin)/(ymax-ymin))
        ratio = self.loader.get_aspect_ratio(item)
        xmax_ = xmin+int(ratio * (ymax-ymin))
        # 보정값
        if patch_size:
            return [filename, xmin if xmin >= 0 else 0, ymin if ymin >= 0 else 0, xmax_ if xmax < patch_size else patch_size - 1,
                    ymax if ymax < patch_size else patch_size - 1, self.loader.get_label(item), width, height]
        else:
            return [filename, xmin, ymin, xmax_, ymax, self.loader.get_label(item), width, height]

    def generate_noisy_img(self, img, bg_blurry):
        np_img = np.array(img, dtype=np.uint8)
        img_blurry = np.log(np.log(cv2.Laplacian(np_img, cv2.CV_64F).var()))
        # y = -(a/2)x + a
        img_gaussian = (img_blurry - bg_blurry) * 2 / img_blurry if img_blurry != 0 else 2
        # print(img_blurry, img_gaussian)
        if img_gaussian < 0:
            img_gaussian = 0

        # gaussian blur
        img_filter = ImageFilter.GaussianBlur(img_gaussian)
        img_ = img.filter(img_filter)

        # random gaussian noise
        rand_mean = np.random.random() / 10
        rand_var = np.random.random() / 100
        img_ = random_noise(np.array(img_, dtype=np.uint8), mode='gaussian', seed=None, clip=True, mean=rand_mean,
                            var=rand_var)

        img_ = (img_ * 255 * 0.8).astype(np.uint8)
        img_[:, :, -1] = np_img[:, :, -1]

        return Image.fromarray(img_)

    def generate_img(self, img_file, dir, cnt):
        # open image file
        bg = Image.open(self.dir + 'images/' + img_file)
        bg_ = np.array(bg, dtype=np.uint8)

        # load item images
        imgs = sampler.loader.get_img()

        # detect blurry (prevent big size assert)
        bg_blurry = np.log(np.log(cv2.Laplacian(cv2.resize(bg_, (bg_.shape[0]//2, bg_.shape[1]//2)), cv2.CV_64F).var()))
        #print(bg_blurry)

        # generate blur, noisy item
        for i, im in enumerate(imgs):
            # make noise
            imgs[i] = self.generate_noisy_img(im, bg_blurry)

        # extract histogram
        '''bg_val, bg_cnt = np.unique(bg_, return_counts=True)
        bg_hist = np.cumsum(bg_cnt).astype(np.float64)
        bg_hist /= bg_hist[-1]'''


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
        # center (use in patch split)
        seg['xcenter'] = (seg['xmin'] + seg['xmax']) // 2
        seg['ycenter'] = (seg['ymin'] + seg['ymax']) // 2

        new_annotation = [self.generate_new_annotation('', row[0], row[1], row[2], row[3], row[4], row[5])
                            for row in seg.loc[:, ['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']].to_numpy()]

        for i, a in enumerate(new_annotation):
            img = imgs[self.loader.get_index_by_label(a[5])]
            img_shape = (a[3] - a[1], a[4] - a[2])
            img = img.resize(img_shape)

            # overlay on background img
            bg.paste(img, (a[1], a[2]), img)
            # matching histogram (not developed)

        ret = []
        # patch size보다 작으면 resize
        bg_size = bg.size
        if bg_size[0] < self.patch_size or bg_size[1] < self.patch_size:
            new_size_w, new_size_h = bg_size
            if bg_size[0] < self.patch_size:
                new_size_w = self.patch_size
            if bg_size[1] < self.patch_size:
                new_size_h = self.patch_size
            bg = bg.resize((new_size_w, new_size_h))
        # patch로 나눠서 image와 annotation 추가
        if self.patch_size:
            width, height = bg.size
            w_num, h_num = width // self.patch_size, height // self.patch_size
            w_pad, h_pad = width % self.patch_size, height % self.patch_size
            w_stride, h_stride = self.patch_size - w_pad // w_num, self.patch_size - h_pad // h_num

            # (w * self.patch_size, h * self.patch_size)
            for w in range(w_num):
                for h in range(h_num):
                    xmin, xmax = w * w_stride, w * w_stride + self.patch_size
                    ymin, ymax = h * h_stride, h * h_stride + self.patch_size
                    bg_patch = bg.crop((xmin, ymin, xmax, ymax))

                    # save image
                    filename = '{}{}.jpg'.format(dir, cnt)

                    # make annotations of patch
                    seg_ = seg.loc[(seg['xcenter'] >= xmin) & (seg['xcenter'] < xmax) & (seg['ycenter'] >= ymin) & (seg['ycenter'] < ymax)].copy()

                    # except image with no annotation
                    if len(seg_) > 0:
                        # adjust patch
                        seg_.loc[:, ['xmin', 'xmax']] -= xmin
                        seg_.loc[:, ['ymin', 'ymax']] -= ymin
                        new_annotation = [self.generate_new_annotation(filename, row[0], row[1], row[2], row[3],
                                            self.patch_size, self.patch_size, self.patch_size)
                                          for row in seg_.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()]
                        # add
                        self.new_annotations += new_annotation
                        bg_patch.save(filename)
                        cnt += 1

        # patch로 나눌 필요가 없어서 annotation 추가
        else:
            filename = '{}{}.jpg'.format(dir, cnt)
            cnt += 1
            new_annotation = [[filename]+row[1:] for row in new_annotation]
            self.new_annotations += new_annotation
            bg.save(filename)

        return ret, cnt

    # generate merged image bg with items
    def generate_img_dataset(self, dir='images/'):
        cnt = 0
        num_list = len(self.random_list)

        # make directory (remove exist dir)
        if os.path.isdir('images/'):
            shutil.rmtree('images/')
        os.mkdir('images/')

        print('Generating Images from {} Background Images...'.format(num_list))
        filename = '{}{}.jpg'.format(dir, cnt)
        for i, img in enumerate(self.random_list):
            img_list, cnt = self.generate_img(img, dir, cnt)

            if num_list < 10 or (i+1) % (num_list // 10) == 0:
                print('{}% Done, {} Image Generated.'.format(int((i+1) / num_list * 100), cnt, num_list))
        return cnt

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

    def generate_ids(self, filename, index):
        with open(filename, 'w') as f:
            for i in index:
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
        cnt = len(glob.glob('images/*.jpg'))
        index = np.arange(cnt)
        np.random.shuffle(index)
        val = int((1.0-self.val_p-self.test_p) * cnt)
        test = int((1.0-self.test_p) * cnt)
        # make dataset_ids
        print('Making dataset ids...')
        self.generate_ids('dataset/dataset_ids/train.txt', index[:val])
        self.generate_ids('dataset/dataset_ids/val.txt', index[val:test])
        self.generate_ids('dataset/dataset_ids/test.txt', index[test:])
        # make XML
        self.generate_xml()

        return dir

    # check error from generated files
    def check_error(self):
        imgs = glob.glob('images/*.jpg')
        xmls = glob.glob('dataset/annotations/*.xml')

        xmls = list(map(lambda a: int(os.path.split(a)[1][:-4]), xmls))
        imgs = list(map(lambda a: int(os.path.split(a)[1][:-4]), imgs))

        xmls.sort()
        imgs.sort()

        flag = True
        for xml, img in zip(xmls, imgs):
            if xml != img:
                if xml < img:
                    print('{}.jpg ~ {}.jpg Not Generated!'.format(img+1, xml))
                else:
                    print('{}.xml ~ {}.xml Not Generated!'.format(xml+1, img))
                flag = False

        if flag:
            img_len = len(imgs)
            xml_len = len(xmls)
            if xml_len > img_len:
                print('{}.jpg Not Generated!'.format(img_len))
            elif xml_len < img_len:
                print('{}.xml Not Generated!'.format(xml_len))
            else:
                print('All files Generated.')


if __name__ == '__main__':
    sampler = SKU110KSampler(num_files=11700, patch_size=640, skip_p=0.3)
    sampler.generate_img_dataset()
    sampler.generate_annotations()
    sampler.check_error()

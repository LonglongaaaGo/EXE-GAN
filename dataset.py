from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from op.utils_train import listdir
import cv2

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class MultiResolution_mask_Dataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

            key = f'{self.resolution}-{str(index).zfill(5)}_mask'.encode('utf-8')
            mask_img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        buffer = BytesIO(mask_img_bytes)
        mask_img = Image.open(buffer)

        flip = random.randint(0, 1)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)

        mask_img = self.transform(mask_img)
        img = self.transform(img)

        return img,mask_img


class  ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size
    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.ANTIALIAS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img)

        return img


def dilate_demo(d_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 椭圆形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))  # 十字形
    image = cv2.dilate(d_image, kernel)  # 膨胀操作
    # plt_show_Image_image(image)
    return image


def erode_demo(e_image):
    kernel_size = random.randint(3,7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))  # 定义结构元素的形状和大小  矩形
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 椭圆形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))  # 十字形
    image = cv2.erode(e_image, kernel)  # 腐蚀操作
    # plt_show_Image_image(image)
    return image
    # 腐蚀主要就是调用cv2.erode(img,kernel,iterations)，这个函数的参数是
    # 第一个参数：img指需要腐蚀的图
    # 第二个参数：kernel指腐蚀操作的内核，默认是一个简单的3X3矩阵，我们也可以利用getStructuringElement（）函数指明它的形状
    # 第三个参数：iterations指的是腐蚀次数，省略是默认为1


class ImageFolder_with_edges(Dataset):
    """
    load images and edge maps
    """
    def __init__(self, image_root, edge_root,transform=None,im_size=(256,256)):
        """
        :param image_root: root for the images
        :param edge_root: root for the masks
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolder_with_edges, self).__init__()
        self.image_root = image_root
        self.edge_root = edge_root

        self.edge_frame = self._parse_frame(self.edge_root)
        self.frame = self._parse_frame(self.image_root)

        self.transform = transform
        self.im_size = im_size

    def _parse_frame(self,root):
        img_names = []
        listdir(root,img_names)
        img_names.sort()
        frame = []
        for i in range(len(img_names)):
            image_path = os.path.join(root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        edge_file = self.edge_frame[idx]

        img = Image.open(file).convert('RGB')
        edge_img = Image.open(edge_file).convert('L')

        w,h = img.size
        edge_img = edge_img.resize((w,h),Image.NEAREST)

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.ANTIALIAS)
            edge_img_scaled = edge_img.resize((new_w,new_h),Image.ANTIALIAS)

            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)

            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            edge_img = edge_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))

        # RandomHorizontalFlip
        flip = random.randint(0, 1)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            edge_img = edge_img.transpose(Image.FLIP_LEFT_RIGHT)


        edge_img = np.array(edge_img)
        if edge_img.ndim == 2:
            mask = np.expand_dims(edge_img, axis=0)
        else:
            mask = edge_img[0:1, :, :]

        mask[mask < 128] = 1.0
        mask[mask >= 128] = 0

        rand_ = random.randint(1,2)
        # mask = erode_demo(mask)
        if rand_ == 1:
            mask = erode_demo(mask)
        # elif rand_ == 2:
        #     pass
        #     # mask = dilate_demo(mask)
        # else:
        #     pass

        # print(np.max(mask))
        if self.transform:
            img = self.transform(img)


        return img,mask


class ImageFolder_with_mask(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, mask_root,mask_file,transform=None,im_size=(256,256)):
        """
        :param root: root for the images
        :param mask_root: root for the masks
        :param transform:
        :param im_size:  (h,w)
        """
        super(ImageFolder_with_mask, self).__init__()
        self.root = root
        self.mask_root = mask_root
        self.mask_file = mask_file
        self._get_mask_list()
        self.frame = self._parse_frame()

        self.transform = transform
        self.im_size = im_size

    def _get_mask_list(self):
        mask_list = []

        file = open(self.mask_file)
        lines = file.readlines()
        for line in lines:
            mask_path = os.path.join(self.mask_root,line.strip())
            mask_list.append(mask_path)
        file.close()
        mask_list.sort()
        self.mask_list = mask_list

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')

        mask_idx = np.random.randint(0,len(self.mask_list)-1)
        mask_path = self.mask_list[mask_idx]
        mask_img = Image.open(mask_path).convert('P')

        w,h = img.size

        mask_img = mask_img.resize((w,h),Image.NEAREST)
        # plt.imshow(mask_img)
        # plt.show()

        mask_img = np.array(mask_img)
        if mask_img.ndim == 2:
            mask = np.expand_dims(mask_img, axis=0)
        else:
            mask = mask_img[0:1, :, :]
        mask[mask > 0] = 1.0

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.ANTIALIAS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        if self.transform:
            img = self.transform(img)

        return img,mask

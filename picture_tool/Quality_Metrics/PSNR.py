"""
Video Quality_Metrics
Copyright (c) 2014 Alex Izvorski <aizvorski@gmail.com>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from matplotlib.pyplot import imread
import numpy as np
import math
import pathlib
from tqdm import tqdm
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
import os
import utils_train as ut

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def PSNR2(img1, img2):
	mse = np.mean( (img1/255. - img2/255.) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))




def psnr_single_given_paths(path1, path2,suffix1="jpg",suffix2="jpg",psnr=psnr):

    path1 = pathlib.Path(path1)
    files1 = list(path1.glob('*%s' % suffix1))
    files1.sort()
    path2 = pathlib.Path(path2)
    files2 = list(path2.glob('*%s' % suffix2))
    files2.sort()

    match_tag = ut.match_list_str(files1,files2,suffix1,suffix2)

    if match_tag == False:
        print("files name not match break!")
        return

    psnr_values = []
    for i in tqdm(range(len(files2))):

        npImg1 = imread(files1[i])
        npImg2 = imread(files2[i])

        psnr_value = psnr(npImg1,npImg2)

        psnr_values.append(psnr_value)

        # img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
        # img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0
        #
        # if torch.cuda.is_available():
        #     img1 = img1.cuda()
        #     img2 = img2.cuda()
        #
        # ssim_value = ssim_single(img1, img2)
        # ssim_values.append(ssim_value)
    return psnr_values


def psnr_single_given_files(files1, files2):

    # path1 = pathlib.Path(path1)
    # files1 = list(path1.glob('*.%s' % suffix))
    #
    # path2 = pathlib.Path(path2)
    # files2 = list(path2.glob('*.%s' % suffix))

    psnr_values = []
    for i in tqdm(range(len(files2))):
        # print(files1[i])
        npImg1 = imread(str(files1[i]))
        npImg2 = imread(str(files2[i]))

        psnr_value = psnr(npImg1,npImg2)

        psnr_values.append(psnr_value)

        # img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
        # img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0
        #
        # if torch.cuda.is_available():
        #     img1 = img1.cuda()
        #     img2 = img2.cuda()
        #
        # ssim_value = ssim_single(img1, img2)
        # ssim_values.append(ssim_value)
    return psnr_values


def psnr_single_given_paths_cv2(path1, path2,suffix1,suffix2):
    path1 = pathlib.Path(path1)
    files1 = list(path1.glob('*%s' % suffix1))
    files1.sort()
    path2 = pathlib.Path(path2)
    files2 = list(path2.glob('*%s' % suffix2))
    files2.sort()

    match_tag = ut.match_list_str(files1, files2, suffix1, suffix2)

    if match_tag == False:
        print("files name not match break!")
        return
    psnr_values = []
    for i in tqdm(range(len(files2))):
        # print(files1[i])
        npImg1 = cv2.imread(str(files1[i]))
        npImg2 = cv2.imread(str(files2[i]))
        psnr_value = cv2.PSNR(npImg1, npImg2)

        psnr_values.append(psnr_value)


    return psnr_values




def psnr_single_given_filescv2(files1, files2):

    psnr_values = []
    for i in tqdm(range(len(files2))):
        # print(files1[i])
        npImg1 = cv2.imread(str(files1[i]))
        npImg2 = cv2.imread(str(files2[i]))
        psnr_value = cv2.PSNR(npImg1, npImg2)

        psnr_values.append(psnr_value)


    return psnr_values




if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path2real', type=str, help=('Path to the real images'))
    parser.add_argument('--path2fake', type=str, help=('Path to generated images'))
    parser.add_argument('-c', '--gpu', default='', type=str, help='GPU to use (leave blank for CPU only)')
    parser.add_argument('--images_suffix', default='jpg', type=str, help='image file suffix')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    path1 = args.path2real
    path2 = args.path2fake
    psnr_values = psnr_single_given_paths(path1, path2, suffix=args.images_suffix)

    psnr_values = np.asarray(psnr_values, dtype=np.float32)
    np.save('PSNR', psnr_values)
    print('PSNR: ', psnr_values.mean())



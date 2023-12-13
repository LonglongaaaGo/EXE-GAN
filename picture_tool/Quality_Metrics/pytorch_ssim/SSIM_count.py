import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from matplotlib.pyplot import imread
from tqdm import tqdm
from picture_tool.Quality_Metrics import pytorch_ssim
from skimage.measure import compare_ssim
# from skimage.structural_similarity import compare_ssim

import utils_train as ut

def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ssim_loss_example():
    npImg1 = cv2.imread("einstein.png")
    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.rand(img1.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
    ssim_value = pytorch_ssim.ssim(img1, img2).data[0]
    print("Initial ssim:", ssim_value)

    # Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
    ssim_loss = pytorch_ssim.SSIM()

    optimizer = optim.Adam([img2], lr=0.01)

    while ssim_value < 0.95:
        optimizer.zero_grad()
        ssim_out = -ssim_loss(img1, img2)
        ssim_value = - ssim_out.data[0]
        print(ssim_value)
        ssim_out.backward()
        optimizer.step()


def ssim_single(img1,img2):
    # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    # print("Initial ssim:", ssim_value)

    # Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
    # ssim_loss = pytorch_ssim.SSIM()

    return ssim_value




def ssim_single_given_paths(path1, path2,suffix1="jpg",suffix2="jpg",ssim_single = ssim_single):

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

    ssim_values = []

    for i in tqdm(range(len(files2))):

        npImg1 = imread(files1[i])
        npImg2 = imread(files2[i])

        img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
        img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0

        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()

        ssim_value = ssim_single(img1, img2)
        ssim_values.append(ssim_value)

    return ssim_values




def ssim_single_given_files(files1, files2):

    # path1 = pathlib.Path(path1)
    # files1 = list(path1.glob('*.%s' % suffix))
    #
    # path2 = pathlib.Path(path2)
    # files2 = list(path2.glob('*.%s' % suffix))

    ssim_values = []
    for i in tqdm(range(len(files2))):

        npImg1 = imread(str(files1[i]))
        npImg2 = imread(str(files2[i]))

        img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
        img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0

        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()

        ssim_value = ssim_single(img1, img2)
        ssim_values.append(ssim_value)

    return ssim_values




def ssim_single_given_paths_skimage(path1, path2,suffix1,suffix2):
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

    ssim_values = []
    for i in tqdm(range(len(files2))):

        # ssim_value  = compare_ssim(to_grey(cv2.imread(str(files1[i]))),
        #                          to_grey(cv2.imread(str(files2[i]))))
        ssim_value = compare_ssim(to_grey(cv2.imread(str(files2[i]))),
                                  to_grey(cv2.imread(str(files1[i]))))
        ssim_values.append(ssim_value)

    return ssim_values



def ssim_single_given_files_skimage(files1, files2):

    ssim_values = []
    for i in tqdm(range(len(files2))):

        ssim_value  = compare_ssim(to_grey(cv2.imread(str(files1[i]))),
                                 to_grey(cv2.imread(str(files2[i]))))
        ssim_values.append(ssim_value)

    return ssim_values




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
    ssim_values = ssim_single_given_paths(path1, path2,suffix=args.images_suffix)

    ssim_values = np.asarray(ssim_values, dtype=np.float32)
    np.save('SSIM', ssim_values)
    print('SSIM: ', ssim_values.mean())
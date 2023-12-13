import cv2
import os
import sys
import math
import time
import json
import glob
import argparse
import urllib.request
from PIL import Image, ImageFilter
from numpy import random
import numpy as np
import torch
from picture_tool.Quality_Metrics import metric as module_metric
from picture_tool.Quality_Metrics.SIFID.inception import InceptionV3
from picture_tool.Quality_Metrics.metric import calculate_activation_statistics, calculate_frechet_distance


# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args





def test_matrix(path1,postfix1,path2,postfix2,test_name):
    out_dic = {}

    real_names = list(glob.glob('{}/*{}'.format(path1,postfix1)))
    fake_names = list(glob.glob('{}/*{}'.format(path2,postfix2)))
    real_names.sort()
    fake_names.sort()
    print(len(real_names))
    print(len(fake_names))

    ###
    up_name = []
    for name_ in test_name:
        if  name_ in ['mae', 'psnr', 'ssim']:
            up_name.append(name_)

    ###

    # metrics prepare for image assesments
    metrics = {met: getattr(module_metric, met) for met in up_name}
    # infer through videos
    real_images = []
    fake_images = []

    print("opening files")

    evaluation_scores = {key: 0 for key, val in metrics.items()}

    for rname, fname in zip(real_names, fake_names):
        rimg = Image.open(rname)
        fimg = Image.open(fname)
        real_images.append(np.array(rimg))
        fake_images.append(np.array(fimg))

    print("calculating image quality assessments")
    # calculating image quality assessments
    for key, val in metrics.items():
        evaluation_scores[key] = val(real_images, fake_images)
        out_dic[key] = evaluation_scores[key]
    print(' '.join(['{}: {:6f},'.format(key, val) for key, val in evaluation_scores.items()]))


    if "fid" not in test_name:
        print('Finish evaluation from {}'.format(path2))
        return
    dims = 2048
    batch_size = 2
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = set_device(InceptionV3([block_idx]))

    # calculate fid statistics for real images
    print("calculate fid statistics for real images")
    real_images = np.array(real_images).astype(np.float32) / 255.0
    real_images = real_images.transpose((0, 3, 1, 2))
    real_m, real_s = calculate_activation_statistics(real_images, model, batch_size, dims)

    # calculate fid statistics for fake images
    print("calculate fid statistics for fake images")
    fake_images = np.array(fake_images).astype(np.float32) / 255.0
    fake_images = fake_images.transpose((0, 3, 1, 2))
    fake_m, fake_s = calculate_activation_statistics(fake_images, model, batch_size, dims)

    print("calculate fid statistics for fake images")
    fid_value = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)
    print('FID: {}'.format(round(fid_value, 5)))
    out_dic["fid"] = fid_value
    print('Finish evaluation from {}'.format(path2))

    return out_dic


def eval_other_(inpaint_root,gt_root,gt_postfix,inpainting_postfix):
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--path1', type=str)
    parser.add_argument('--path2', type=str)
    parser.add_argument('--postfix1', type=str)
    parser.add_argument('--postfix2', type=str)

    args = parser.parse_args()

    # gt_postfix = "_gt.png"
    # inpainting_postfix = "_inpaint.png"

    # gt_root = "/home/k/Longlongaaago/inpainting_gmcnn-amax2/pytorch/" \
    #           "test_20210305-025945_celebahq_gmcnn_s256x256_gc32"
    # inpaint_root = "/home/k/Longlongaaago/inpainting_gmcnn-amax2/pytorch/" \
    #                "test_results/test_20210305-025945_celebahq_gmcnn_s256x256_gc32"

    args.path1 = gt_root
    args.postfix1 = gt_postfix

    args.path2 = inpaint_root
    args.postfix2 = inpainting_postfix
    test_name = ["fid"]
    test_matrix(args,test_name=test_name)


def test_all():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--root1', default="",help="dataset",type=str)
    parser.add_argument('--root2', default="", help="test_ root_ for all",type=str)
    parser.add_argument('--postfix1', default=".jpg", type=str)
    parser.add_argument('--postfix2', default=".png", type=str)

    args = parser.parse_args()

    test_name = ["fid"]
    name_list = ["20210411072149","20210411072225"]

    dir_list = os.listdir(args.root2)
    for file in dir_list:
        if file not in name_list: continue
        print(file)
        path = os.path.join(args.root2, file)
        if os.path.isdir(path):
            #running file

            if not os.path.exists(path): continue

            print("%s is existing, now we testing!" % path)

            index= -1
            score = 1000
            for j in range(0,10):

                new_path = os.path.join(path,"eval_%d0000/img"%(j))
                if not os.path.exists(new_path): continue
                print("test in %s"%new_path)
                out_dic = test_matrix(path1=args.root1,postfix1=args.postfix1
                            ,path2=new_path,postfix2=args.postfix2, test_name=test_name)
                remove_id = j
                #如果是刚开始
                if index == -1:
                    index = 0
                    score = out_dic["fid"]
                elif score>out_dic["fid"]:
                    score = out_dic["fid"]
                    remove_id = index
                    index = j

                remove_path = os.path.join(path, "eval_%d0000" % (remove_id))
                os.remove(remove_path)





# dir_list = ut.listdir(args.root2)


def test_single():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--path1', default="", type=str)
    parser.add_argument('--path2', default="", type=str)
    parser.add_argument('--postfix1', default=".jpg", type=str)
    parser.add_argument('--postfix2', default=".jpg", type=str)

    args = parser.parse_args()

    # gt_postfix = "_gt.png"
    # inpainting_postfix = "_inpaint.png"

    # gt_root = "/home/k/Longlongaaago/inpainting_gmcnn-amax2/pytorch/" \
    #           "test_20210305-025945_celebahq_gmcnn_s256x256_gc32"
    # inpaint_root = "/home/k/Longlongaaago/inpainting_gmcnn-amax2/pytorch/" \
    #                "test_results/test_20210305-025945_celebahq_gmcnn_s256x256_gc32"

    # args.path1 = gt_root
    # args.postfix1 = gt_postfix

    # args.path2 = inpaint_root
    # args.postfix2 = inpainting_postfix
    test_name = ["fid"]
    test_matrix(args, test_name=test_name)



if __name__ == '__main__':
    test_all()

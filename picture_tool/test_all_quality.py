
import numpy as np
from picture_tool.Quality_Metrics.pytorch_ssim.SSIM_count import ssim_single_given_paths,ssim_single_given_paths_skimage,ssim_single
from picture_tool.Quality_Metrics.pytorch_ssim.SSIM_2 import SSIM
from picture_tool.Quality_Metrics.SIFID.sifid_score import calculate_sifid_given_paths
from picture_tool.Quality_Metrics.PSNR import psnr_single_given_paths,psnr,PSNR2,psnr_single_given_paths_cv2
from piq import ssim

import torch
from picture_tool.eval import eval_other_





if __name__ == '__main__':

    gt_postfix ="_gt.png"
    inpainting_postfix = "_inpaint.png"

    gt_root = "/home/k/Workspace/inpainting_gmcnn_ori/test_results/" \
              "test_20210315-082733_celebahq_gmcnn_s256x256_gc32"
    inpaint_root = "/home/k/Workspace/inpainting_gmcnn_ori/test_results/" \
                   "test_20210315-082733_celebahq_gmcnn_s256x256_gc32"

    # psnr_values = psnr_single_given_paths(gt_root,inpaint_root,gt_postfix,inpainting_postfix,psnr=psnr)
    # psnr_values = np.asarray(psnr_values,dtype=np.float32)
    # print("PSNR1: ",psnr_values.mean())
    # #
    # psnr_values = psnr_single_given_paths(gt_root, inpaint_root, gt_postfix, inpainting_postfix,psnr=PSNR2)
    # psnr_values = np.asarray(psnr_values, dtype=np.float32)
    # print("PSNR2: ", psnr_values.mean())
    # #
    # #
    # psnr_values = psnr_single_given_paths_cv2(gt_root, inpaint_root, gt_postfix, inpainting_postfix)
    # psnr_values = np.asarray(psnr_values, dtype=np.float32)
    # print("PSNR_cv2: ", psnr_values.mean())

    # del psnr_values

    # ssim_values = ssim_single_given_paths(gt_root,inpaint_root,gt_postfix,inpainting_postfix,ssim_single=ssim_single)
    # ssim_values = np.asarray(ssim_values, dtype=np.float32)
    # print('SSIM1: ', ssim_values.mean())
    #
    # ssim_values = ssim_single_given_paths(gt_root, inpaint_root, gt_postfix, inpainting_postfix,ssim_single = ssim)
    # ssim_values = np.asarray(ssim_values, dtype=np.float32)
    # print('SSIM2: ', ssim_values.mean())

    ssim_values = ssim_single_given_paths_skimage(gt_root, inpaint_root, gt_postfix, inpainting_postfix)
    ssim_values = np.asarray(ssim_values, dtype=np.float32)
    print('ski_SSIM: ', ssim_values.mean())
    #
    del ssim_values


    sifid_values = calculate_sifid_given_paths(gt_root,inpaint_root,1,
                                               torch.cuda.is_available(),64,gt_postfix,inpainting_postfix)

    sifid_values = np.asarray(sifid_values,dtype=np.float32)
    print("SIFID: ", sifid_values.mean())


    del sifid_values

    eval_other_(gt_root,inpaint_root,gt_postfix,inpainting_postfix)





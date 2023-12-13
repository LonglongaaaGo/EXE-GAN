import argparse
import math
import random
import os
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import lpips_local
from op.utils import get_mask,get_completion,mkdirs,delete_dirs,copy_dirs
from op.diffaug import DiffAugment_withsame_trans
from fid_eval import test_matrix,get_temp_fid_activation,get_final_fid_activation
import re
from pytorch_fid import fid_score

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import op.utils_train as ut
import numpy as np

# 来源：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


# 合并结果的函数
# 1. all_gather，将各个进程中的同一份数据合并到一起。
#   和all_reduce不同的是，all_reduce是平均，而这里是合并。
# 2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
# 3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]



def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def distribued_test(args,test_loader,generator,device,inter,mask_shapes):

    eval_dict = os.path.join(args.eval_dir, f"{inter}_rank_{get_rank()}")
    delete_dirs(eval_dict)
    mkdirs(eval_dict)

    generator.eval()
    print("testing!!! len:%d" % (len(test_loader.dataset)))
    with torch.no_grad():
        for jjj, data in tqdm(enumerate(test_loader)):
            real_imgs = data.to(device)
            ##get mask
            gin, gt_local, mask, mask_01, im_ins = get_mask(real_imgs, mask_type="center", im_size=args.size,mask_shapes=mask_shapes)
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img = generator(gin, noise)
            completed_img = get_completion(fake_img, real_imgs.detach(), mask_01.detach())
            torch.cuda.empty_cache()
            for j, g_img in enumerate(completed_img):
                real_img = real_imgs[j].squeeze()
                im_in = im_ins[j].squeeze()
                utils.save_image(
                    g_img,
                    f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_inpaint.png",
                    nrow=int(1), normalize=True, range=(-1, 1), )
                utils.save_image(
                    real_img,
                    f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_gt.png",
                    nrow=int(1), normalize=True, range=(-1, 1), )

                utils.save_image(
                    im_in,
                    f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_mask.png",
                    nrow=int(1), normalize=True, range=(-1, 1), )



def calculate_all(args,iter,eval_dict,loger,best_fid,best_path,best_evel_batch,device):
    tmp_fid = fid_score.calculate_fid_given_paths_postfix(path1=eval_dict, postfix1="_gt.png",
                                                          path2=eval_dict, postfix2="_inpaint.png",
                                                          batch_size=best_evel_batch, device=device,
                                                          dims=2048, num_workers=args.num_workers)
    print("fid_score_:%g" % tmp_fid)

    test_name = ['mae', 'psnr', 'ssim']
    out_dic = test_matrix(path1=eval_dict, postfix1="_gt.png"
                          , path2=eval_dict, postfix2="_inpaint.png", test_name=test_name)

    loger.update(iter=iter, mae=out_dic['mae'], psnr=out_dic['psnr'],
                 ssim=out_dic['ssim'], fid=tmp_fid)

    print("mae:%g, psnr:%g, ssim:%g,fid:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim'], tmp_fid))

    if tmp_fid < best_fid:
        delete_dirs(best_path)
        best_fid = tmp_fid
        best_path = eval_dict
    else:
        delete_dirs(eval_dict)

    print("best_fid:%g, best_path:%s", best_fid,best_path)
    return  out_dic['mae'], out_dic['psnr'], out_dic['ssim'], tmp_fid,best_fid,best_path


def calculate_all_PIDS(args,iter,eval_dict,loger,best_fid,best_path,best_evel_batch,device,postfix1="_gt.png",postfix2="_inpaint.png"):
    tmp_fid = fid_score.calculate_fid_given_paths_postfix(path1=eval_dict, postfix1=postfix1,
                                                          path2=eval_dict, postfix2=postfix2,
                                                          batch_size=best_evel_batch, device=device,
                                                          dims=2048, num_workers=args.num_workers)
    print("fid_score_:%g" % tmp_fid)

    fid_value, U_IDS_score, P_IDS_score = fid_score.calculate_P_IDS_U_IDS_given_paths_postfix(path1=eval_dict,
                                                                                              postfix1=postfix1,
                                                                                              path2=eval_dict,
                                                                                              postfix2=postfix2,
                                                                                              batch_size=args.batch,
                                                                                              device=device,
                                                                                              dims=2048,
                                                                                              num_workers=args.num_workers)
    test_name = ['mae', 'psnr', 'ssim']
    out_dic = test_matrix(path1=eval_dict, postfix1=postfix1
                          , path2=eval_dict, postfix2=postfix2, test_name=test_name)

    loger.update(iter=iter, mae=out_dic['mae'], psnr=out_dic['psnr'],
                 ssim=out_dic['ssim'], fid=tmp_fid,
                 U_IDS_score=U_IDS_score, P_IDS_score=P_IDS_score)

    print("mae:%g, psnr:%g, ssim:%g,fid:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim'], tmp_fid))

    print("fid_value:%g, U_IDS_score:%g, P_IDS_score:%g" % (fid_value, U_IDS_score, P_IDS_score))

    if tmp_fid < best_fid:
        # delete_dirs(best_path)
        best_fid = tmp_fid
        best_path = eval_dict
    # else:
    #     delete_dirs(eval_dict)

    print("best_fid:%g, best_path:%s", best_fid,best_path)
    out_dics = {}
    out_dics['mae'] = out_dic['mae']
    out_dics['psnr'] = out_dic['psnr']
    out_dics['ssim'] = out_dic['ssim']
    out_dics['fid_value'] = fid_value
    out_dics['U_IDS_score'] = U_IDS_score
    out_dics['P_IDS_score'] = P_IDS_score
    out_dics['tmp_fid'] = tmp_fid
    out_dics['best_fid'] = best_fid
    out_dics['best_path'] = best_path
    return  out_dics


def get_bigger_batch(data_len,max_num=100):

    for i in range(max_num,1,-1):
        if i>data_len: return data_len
        if data_len%(i) == 0:
            return i

    return 1


def distributed_copy(eval_dict,out_dir,distributed=False):
    """
    from nodes to the local node
    :param args:
    :param eval_dict:
    :return:
    """
    print("distributed_copy!!")
    if distributed == False:
        delete_dirs(out_dir)
        mkdirs(out_dir)
        ut.copy_Dir2Dir(eval_dict, out_dir)
    else:
        if get_rank() == 0:
            delete_dirs(out_dir)
            mkdirs(out_dir)
        torch.distributed.barrier()
        ut.copy_Dir2Dir(eval_dict, out_dir)
        torch.distributed.barrier()


def save_images(args,iter,completed_img,real_imgs,im_ins,eval_dict):
    for j, g_img in enumerate(completed_img):
        real_img = real_imgs[j].squeeze()
        im_in = im_ins[j].squeeze()
        utils.save_image(
            g_img,
            f"{str(eval_dict)}/{str(iter * args.batch + j).zfill(6)}_{str(get_rank())}_inpaint.png",
            nrow=int(1), normalize=True, range=(-1, 1), )
        utils.save_image(
            real_img,
            f"{str(eval_dict)}/{str(iter * args.batch + j).zfill(6)}_{str(get_rank())}_gt.png",
            nrow=int(1), normalize=True, range=(-1, 1), )
        utils.save_image(
            im_in,
            f"{str(eval_dict)}/{str(iter * args.batch + j).zfill(6)}_{str(get_rank())}_mask.png",
            nrow=int(1), normalize=True, range=(-1, 1), )



def non_distributed_eval(out_dir,batch_size,fid_test=False):
    print("###########check################")
    tmp_fid = -1
    if fid_test == True:
        test_name = ["fid"]
        out_dic = test_matrix(path1=out_dir, postfix1="_gt.png"
                              , path2=out_dir, postfix2="_inpaint.png", test_name=test_name,
                              batch_size=batch_size)
        tmp_fid = out_dic["fid"]
        print("fid___:%g" % out_dic["fid"])

    test_name = ['mae', 'psnr', 'ssim']
    out_dic = test_matrix(path1=out_dir, postfix1="_gt.png"
                          , path2=out_dir, postfix2="_inpaint.png", test_name=test_name)

    print("mae_:%g, psnr_:%g, ssim_:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim']))
    print("###########check end###############")

    return out_dic['mae'], out_dic['psnr'], out_dic['ssim'],tmp_fid

#
# def distributed_copy(eval_dir,out_dir,iter):
#     for ii in range(get_world_size()):
#         tmp_eval_dict = os.path.join(eval_dir, f"{iter}_rank_{ii}")
#         ut.copy_Dir2Dir(tmp_eval_dict, out_dir)



def distributed_eval(eval_dict,data_len,distirbuted,device,batch_size,fid_flag=False):

    if distirbuted == False:
        tmp_fid = -1
        if fid_flag == True:
            test_name = ["fid"]
            out_dic = test_matrix(path1=eval_dict, postfix1="_gt.png"
                                  , path2=eval_dict, postfix2="_inpaint.png", test_name=test_name,batch_size=batch_size)
            tmp_fid = out_dic["fid"]

        test_name = ['mae', 'psnr', 'ssim']
        out_dic = test_matrix(path1=eval_dict, postfix1="_gt.png"
                              , path2=eval_dict, postfix2="_inpaint.png", test_name=test_name)

        return out_dic['ssim'],out_dic['psnr'],out_dic['mae'],tmp_fid

    test_name = ['mae', 'psnr', 'ssim']
    out_dic = test_matrix(path1=eval_dict, postfix1="_gt.png"
                          , path2=eval_dict, postfix2="_inpaint.png", test_name=test_name)
    out_dic["num"] = data_len
    print("out_dic[num]%d"%data_len)
    ssim_list = [out_dic["ssim"]]
    psnr_list = [out_dic["psnr"]]
    mae_list = [out_dic["mae"]]
    num_list = [out_dic["num"]]

    ssim_list = np.array(ssim_list)
    psnr_list = np.array(psnr_list)
    mae_list = np.array(mae_list)
    num_list = np.array(num_list)

    ssim_list = distributed_concat(torch.from_numpy(ssim_list).to(device), num_total_examples=torch.distributed.get_world_size())
    psnr_list = distributed_concat(torch.from_numpy(psnr_list).to(device), num_total_examples=torch.distributed.get_world_size())
    mae_list = distributed_concat(torch.from_numpy(mae_list).to(device), num_total_examples=torch.distributed.get_world_size())
    num_list = distributed_concat(torch.from_numpy(num_list).to(device), num_total_examples=torch.distributed.get_world_size())

    ssim_sum = 0
    psnr_sum = 0
    mae_sum = 0
    num_sum = 0
    for kk, num in enumerate(num_list):
        ssim_sum += ssim_list[kk] * num
        psnr_sum += psnr_list[kk] * num
        mae_sum += mae_list[kk] * num *1000
        num_sum += num

    print("num_sum:%d"%num_sum)

    ssim = ssim_sum / num_sum
    psnr = psnr_sum / num_sum
    mae = mae_sum / num_sum / 1000

    del ssim_list,psnr_list,mae_list,num_list

    fid_score = -1
    if fid_flag == True:
        real_acts, fake_acts = get_temp_fid_activation(path1=eval_dict, postfix1="_gt.png",
                                                       path2=eval_dict,postfix2="_inpaint.png",batch_size=batch_size)

        real_acts = torch.from_numpy(real_acts).to(device)
        fake_acts = torch.from_numpy(fake_acts).to(device)
        real_acts = distributed_concat(real_acts, num_total_examples=num_sum)
        fake_acts = distributed_concat(fake_acts, num_total_examples=num_sum)
        print("len_real_acts:%d"%len(real_acts))
        print("num_sum:%d"%num_sum)
        real_acts = real_acts.cpu().numpy()
        fake_acts = fake_acts.cpu().numpy()
        fid_score = get_final_fid_activation(real_acts,fake_acts)
        torch.cuda.empty_cache()

    print(f"ssim:{ssim},psnr:{psnr},mae:{mae},fid_score:{fid_score}")

    return ssim,psnr,mae,fid_score


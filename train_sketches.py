# -*- coding: UTF-8 -*-
#coding=utf-8
import sys
# sys.path.append('./')
# print(sys.path)
import argparse
import math
import random
import os
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
import lpips
import shutil

from Loss.id_loss import IDLoss

from distributed_test import get_bigger_batch,calculate_all,calculate_all_PIDS
from Logger.Logger import Logger
from models.GaussianBlurLayers import ConfidenceDrivenMaskLayer

from op.utils import get_mask,get_completion,mkdirs,delete_dirs,dic_2_str,set_random_seed
from op.diffaug import DiffAugment_withsame_trans
from op.mask_generator import co_mod_mask


from Loss.psp_embedding import Psp_Embedding,embedding_loss

try:
    import wandb

except ImportError:
    wandb = None


from dataset import MultiResolutionDataset,ImageFolder,ImageFolder_with_edges
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch




def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()



def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


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


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def get_random_index(tar_size,part_size):
    range_ = int(tar_size/part_size)

    index_y = random.randint(0,range_-1)
    index_x = random.randint(0,range_-1)

    return index_y,index_x

def crop_image_by_part(image, ind_y,ind_x,part_size):
    y_start = ind_y*part_size
    x_satrt = ind_x*part_size
    return image[:, :, y_start:y_start+part_size,x_satrt:x_satrt +part_size]



def train(args, loader,test_loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    save_inter = 500
    show_inter = 2000
    eval_inter = 20000
    if args.debug == True:
        save_inter = 10
        show_inter = 10
        eval_inter = 10

    pbar = range(args.iter)
    best_fid =args.best_fid
    best_path=args.best_path

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
        logger = Logger(path=args.logger_path, continue_=True)


    mean_path_length = 0

    mask_shapes = [128,128]

    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)

    percept_loss = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

    if args.id_loss_weight>0:
        id_loss = IDLoss(args.arcface_path)

    data_len = len(test_loader) * args.batch
    print("data_len:%d" % data_len)
    best_evel_batch = get_bigger_batch(data_len, max_num=32)
    print("best_evel_batch:%d" % best_evel_batch)


    policy = 'color,translation'

    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator


    ##init embedding
    print("g_module.start_latent :%d" % g_module.start_latent)
    print("g_module.n_psp_latent :%d" % g_module.n_psp_latent)
    psp_embedding = Psp_Embedding(args.psp_checkpoint_path,g_module.start_latent,g_module.n_psp_latent ).to(device)

    os.makedirs("./checkpoint",exist_ok=True)
    os.makedirs("./sample",exist_ok=True)

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    print("embedding_weight:%g!!!"%args.embedding_weight)

    confidence_mask_layer = ConfidenceDrivenMaskLayer( size=65, sigma=1.0/40, iters=7,pad=32)

    rand_end = args.rand_end
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img, s_mask = next(loader)
        # real_img = next(loader)
        real_img = real_img.to(device)
        s_mask = s_mask.to(device)

        rand_num = random.randint(1,rand_end)
        # for inference
        if rand_num == rand_end:
            infer_img = real_img
        else:
            infer_img = torch.flip(real_img, dims=[0])

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        ##get mask
        gin, gt_local, mask, mask_01, im_in = get_mask(real_img, mask_type="stroke_rect", im_size=args.size, mask_shapes=mask_shapes)
        rand_s_mask_num = random.randint(1,rand_end)
        # s_mask[s_mask >= 0] = 0
        # s_mask[s_mask < 0] = 1
        s_mask = s_mask * mask_01.detach()

        # randomly remove the sketches
        if rand_s_mask_num == rand_end:
            s_mask = torch.zeros_like(s_mask) * mask_01.detach()

        gin = torch.cat([gin,s_mask],dim=1)

        mask_weight = confidence_mask_layer(mask_01)
        ada_embedding_mask = (1.0- mask_weight)*mask_01
        ada_embedding_mask = (ada_embedding_mask- torch.min(ada_embedding_mask))/(torch.max(ada_embedding_mask) - torch.min(ada_embedding_mask))

        infer_embedding = psp_embedding(infer_img)
        fake_img = generator(gin,infer_embedding,noise)
        completed_img = get_completion(fake_img,real_img.detach(),mask_01.detach())

        if args.augment:
            #.clone().detach() 保证原tensor不变
            real_img_aug, _ = augment(real_img.clone().detach(), ada_aug_p)
            # completed_img_aug, _ = augment(completed_img, ada_aug_p)
            completed_img_aug,aug_mask_01 = DiffAugment_withsame_trans(completed_img, mask_01.clone().detach(), policy=policy)

        else:
            real_img_aug = real_img
            completed_img_aug = completed_img
            aug_mask_01 = mask_01
            aug_s_mask = s_mask

        fake_pred = discriminator(completed_img_aug.detach())
        real_pred= discriminator(real_img_aug.detach())
        d_seg_loss = torch.zeros(1).mean().cuda()

        d_loss = d_logistic_loss(real_pred, fake_pred)
        # d_loss = d_hing_loss(real_pred, fake_pred)
        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        loss_dict["d_seg_loss"] = d_seg_loss
        d_loss += d_seg_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            temp_real_img = real_img.detach().clone()
            temp_real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(temp_real_img, ada_aug_p)

            else:
                real_img_aug = temp_real_img

            real_pred = discriminator(real_img_aug)

            r1_loss = d_r1_loss(real_pred, temp_real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        #train G
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        infer_embedding = psp_embedding(infer_img)
        fake_img = generator(gin,infer_embedding,noise)

        completed_img = get_completion(fake_img,real_img.detach(),mask_01.detach())

        if args.augment:
            # completed_img_aug, _ = augment(completed_img, ada_aug_p)
            completed_img_aug,aug_mask_01 = DiffAugment_withsame_trans(completed_img.clone(), mask_01.clone().detach(), policy=policy)
        else:
            completed_img_aug = completed_img
            aug_mask_01 = mask_01


        fake_pred= discriminator(completed_img_aug)
        g_seg_loss = torch.zeros(1).mean().cuda()

        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss
        loss_dict["g_seg_loss"] = g_seg_loss
        g_loss += g_seg_loss

        #embedding loss
        if args.embedding_weight >0:
            fake_psp_latents = psp_embedding(completed_img, weight_map=ada_embedding_mask)
            infer_psp_latents = psp_embedding(infer_img.detach())
            g_embedding_loss = embedding_loss(fake_psp_latents,infer_psp_latents) * args.embedding_weight
            loss_dict["g_embedding_loss"] = g_embedding_loss
            g_loss += g_embedding_loss
        else:
            loss_dict["g_embedding_loss"] = torch.zeros(1).mean().cuda()

        #id loss
        if args.id_loss_weight >0:
            g_id_loss = id_loss(completed_img,infer_img.detach(),weight_map=None)*args.id_loss_weight
            loss_dict["g_id_loss"] = g_id_loss
            g_loss += g_id_loss
        else:
            loss_dict["g_id_loss"] = torch.zeros(1).mean().cuda()

        if rand_num == rand_end:
            g_percept_loss = percept_loss(completed_img, real_img.detach(),
                                          weight_map=mask_weight).sum() * args.percept_loss_weight
            loss_dict["g_percept_loss"] = g_percept_loss
            g_loss+=g_percept_loss
        else:
            loss_dict["g_percept_loss"] = torch.zeros(1).mean().cuda()

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            # fake_img, latents = generator(gin[:path_batch_size,:,:,:],noise, return_latents=True)
            infer_embedding = psp_embedding(infer_img[:path_batch_size,:,:,:])
            fake_img, latents = generator(gin[:path_batch_size,:,:,:],infer_embedding,noise,return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()

        # d_rec_loss_val = loss_dict["d_rec_loss"].mean().item()

        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        g_embedding_loss_val = loss_dict["g_embedding_loss"].mean().item()
        # g_l1_loss_val = loss_reduced["g_l1_loss"].mean().item()
        g_percept_loss_val =  loss_dict["g_percept_loss"].mean().item()

        g_id_loss_val = loss_dict["g_id_loss"].mean().item()

        if i% eval_inter == 0 and i>2:
            print("evaling !!!")
            eval_dict = os.path.join(args.eval_dir, str(i))

            if get_rank() == 0:
                delete_dirs(eval_dict)
                mkdirs(eval_dict)

            if args.distributed == True:
                torch.distributed.barrier()
            generator.eval()
            print("testing!!! len:%d"%(len(test_loader.dataset)))
            with torch.no_grad():
                for jjj, data in tqdm(enumerate(test_loader)):
                    if args.debug == True and jjj > 10: break
                    real_imgs, s_masks = data
                    real_imgs = real_imgs.to(device)
                    s_masks = s_masks.to(device)

                    # s_masks[s_masks >= 0] = 0
                    # s_masks[s_masks < 0] = 1

                    infer_imgs = torch.flip(real_imgs, dims=[0])

                    ##get mask
                    gin, gt_local, mask, mask_01, im_ins = get_mask(real_imgs, mask_type="stroke_rect", im_size=args.size,
                                                                   mask_shapes=mask_shapes)

                    # gin, mask_01, im_ins = co_mod_mask(real_imgs, im_size=(args.size,args.size))
                    s_masks = s_masks * mask_01
                    gin = torch.cat([gin, s_masks], dim=1)

                    noise = mixing_noise(args.batch, args.latent, args.mixing, device)
                    infer_embeddings = psp_embedding(infer_imgs)
                    fake_img = g_ema(gin,infer_embeddings,noise)
                    completed_img = get_completion(fake_img, real_imgs.detach(), mask_01.detach())
                    torch.cuda.empty_cache()
                    for j, g_img in enumerate(completed_img):
                        real_img = real_imgs[j].squeeze()
                        tp_im_in = im_ins[j].squeeze()
                        ins_img = infer_imgs[j].squeeze()
                        tp_s_mask = s_masks[j].squeeze()

                        utils.save_image(
                            g_img,
                            f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_inpaint.png",
                            nrow=int(1),normalize=True,range=(-1, 1), )
                        utils.save_image(
                            real_img,
                            f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_gt.png",
                            nrow=int(1),normalize=True,range=(-1, 1), )
                        utils.save_image(
                            tp_im_in,
                            f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_mask.png",
                            nrow=int(1),normalize=True,range=(-1, 1), )
                        utils.save_image(
                            ins_img,
                            f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_instance.png",
                            nrow=int(1), normalize=True, range=(-1, 1), )
                        utils.save_image(
                            tp_s_mask,
                            f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_s_mask.png",
                            nrow=int(1), normalize=True, range=(0, 1), )

            if args.distributed == True:
                torch.distributed.barrier()

            if get_rank() == 0:
                pre_best_fid = best_fid
                out_dics = calculate_all_PIDS(args, i, eval_dict, logger,best_fid,
                                              best_path,best_evel_batch, device)
                outstr_ = dic_2_str(out_dics)
                best_fid = out_dics["best_fid"]
                pbar.set_description((outstr_))
                print(outstr_)

                if pre_best_fid > best_fid:
                    torch.save(
                        {
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "args": args,
                            "ada_aug_p": ada_aug_p,
                            "iter": i,
                            "best_path": best_path,
                            "best_fid": best_fid,
                        },
                        f"checkpoint/best_model.pt",
                    )
                    shutil.copy(f"checkpoint/best_model.pt",f"checkpoint/a_recent_model.pt")

            if get_rank() == 0 and args.delete_test == True:
                delete_dirs(eval_dict)

            generator.train()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"g_percept_loss_val: {g_percept_loss_val:.4f}"
                    f"augment: {ada_aug_p:.4f}; "
                    f"g_embedding_loss_val: {g_embedding_loss_val:.4f}; "
                    f"g_id_loss_val: {g_id_loss_val:.4f}; "

                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % show_inter == 0:
                with torch.no_grad():
                    utils.save_image(
                        torch.cat([completed_img_aug,
                                   infer_img,
                                   im_in,
                                   s_mask.repeat([1,3,1,1]),
                                   ada_embedding_mask.repeat([1, 3, 1, 1]),
                                   ]),
                        f"sample/{str(i).zfill(6)}_.png",
                        # nrow=int(rec_out.shape[0] ** 0.5),
                        nrow=int(args.batch),
                        normalize=True,
                        range=(-1, 1),
                    )


            if i % save_inter == 0:
                print("saving!!!")
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                        "best_path": best_path,
                        "best_fid": best_fid,
                    },
                    f"checkpoint/a_recent_model.pt",)



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='exe_gan', help='models architectures (stylegan2 | stylegan2_4 | swagan | stylegan_fun)')
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument( "--batch", type=int, default=1, help="batch sizes for each gpu" )

    parser.add_argument("--n_sample",type=int,default=8,help="number of the samples generated during training",)
    parser.add_argument("--size", type=int, default=256, help="image sizes for the models")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument( "--path_regularize", type=float, default=2, help="weight of the path length regularization",)
    parser.add_argument("--path_batch_shrink",type=int,default=2,help="batch size reducing factor for the path length regularization (reduce memory consumption)", )
    parser.add_argument("--d_reg_every",type=int,default=16,help="interval of the applying r1 regularization",)
    parser.add_argument("--g_reg_every",type=int, default=4,help="interval of the applying path length regularization",)
    parser.add_argument("--mixing", type=float, default=0.5, help="probability of latent code mixing")
    parser.add_argument("--embedding_weight", type=float, default=0.1, help="weight of the segmentation loss")
    parser.add_argument("--percept_loss_weight", type=float, default=1.5, help="weight of the percept loss")
    parser.add_argument("--rand_end", type=int, default=5, help="length of the random space")
    parser.add_argument("--id_loss_weight", type=float, default=0, help="weight of the id loss")
    parser.add_argument( "--ckpt", type=str,  default=None, help="path to the checkpoints to resume training",)
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier",type=int, default=2,  help="channel multiplier factor for the models. config-f = 2, else = 1",)
    parser.add_argument("--debug",type=bool,default=False,help = "for debugging")
    parser.add_argument("--delete_test", type=bool, default=False, help="delete_test files")

    # parser.add_argument("--GSWGL_LPIPS", type=bool, default=False, help="apply the GSWGL in LPIPS")
    # parser.add_argument("--GSWGL_embedding", type=bool, default=False, help="apply the GSWGL in embedding loss")

    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging" )
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training" )
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p",type=float,default=0,help="probability of applying augmentation. 0 = use adaptive augmentation",)
    parser.add_argument("--ada_target",type=float,default=0.6,help="target augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_length",type=int,default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_every",type=int,default=256,help="probability update interval of the adaptive augmentation", )
    parser.add_argument("--num_workers",type=int,default=8,help="number of workers",)

    parser.add_argument("--resume", type=bool,default=False, help="reload => False, resume = > True ",)
    parser.add_argument("--logger_path", type=str, default="./logger.txt", help="path to the output the generated images")
    parser.add_argument("--arcface_path", type=str, default="/home/k/Workspace/stylegan2-rosinality/pretrained_model/Arcface.pth", help="Arcface model pretrained model")
    parser.add_argument("--psp_checkpoint_path", type=str, default="/home/k/Workspace/stylegan2-rosinality/pretrained_model/psp_ffhq_encode.pt", help="psp model pretrained model")
    parser.add_argument("--out_dir", type=str, default="./out_dir", help="path to the output the final generated images")
    parser.add_argument("--eval_dir", type=str, default="./eval_dir", help="path to the output the generated images")
    parser.add_argument("--test_path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--sketches_path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--sketches_test_path", type=str, help="path to the lmdb dataset")

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.local_rank = args.local_rank
            args.current_device = args.local_rank
        elif 'SLURM_LOCALID' in os.environ:  # for slurm scheduler
            #ngpus_per_node 一个节点有几个可用的GPU
            ngpus_per_node = torch.cuda.device_count()
            #local_rank 在一个节点中的第几个进程，local_rank 在各个节点中独立
            args.local_rank = int(os.environ.get("SLURM_LOCALID"))
            #在所有进程中的rank是多少
            args.rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + args.local_rank

            # args.local_rank = int(os.environ['SLURM_PROCID'])
            # args.gpu = args.local_rank % torch.cuda.device_count()

            available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ""))

            args.current_device = int(available_gpus[args.local_rank])
        import datetime
        torch.cuda.set_device(args.current_device)
        torch.distributed.init_process_group(backend="nccl", init_method="env://",world_size=n_gpu,rank=args.rank,timeout=datetime.timedelta(0,7200))
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0

    from models.exe_gan import Generator, Discriminator

    psp_start_latent = 4
    num_psp_latent = 10

    if args.size == 512:
        num_psp_latent = 12
    elif args.size == 1024:
        num_psp_latent = 14

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        input_channel=5,psp_start_latent=psp_start_latent, num_psp_latent=num_psp_latent,).to(device)

    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
        input_channel=5,psp_start_latent=psp_start_latent, num_psp_latent=num_psp_latent).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    set_random_seed(1)

    args.best_path = ""
    args.best_fid = 1000

    # print("GSWGL in embedding loss",args.GSWGL_embedding)
    # print("GSWGL in _LPIPS loss ",args.GSWGL_LPIPS)

    resume = args.resume
    if args.ckpt is not None:
        print("load models:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            # args.start_iter = int(os.path.splitext(ckpt_name)[0])
            if resume == True:
                args.start_iter = int(ckpt["iter"])
                if "best_path" in ckpt:
                    args.best_path = ckpt["best_path"]
                if "best_fid" in ckpt:
                    args.best_fid = ckpt["best_fid"]

        except ValueError:
            pass
        if resume == True:
            generator.load_state_dict(ckpt["g"])
            discriminator.load_state_dict(ckpt["d"])
            g_ema.load_state_dict(ckpt["g_ema"])

            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])
        else:
            model_dict = generator.state_dict()
            pretrained_dict = {k: v for k, v in ckpt["g"].items() if (k in model_dict and ('down_from_big.0' not in k))}
            model_dict.update(pretrained_dict)
            generator.load_state_dict(model_dict)

            model_dict = g_ema.state_dict()
            pretrained_dict = {k: v for k, v in ckpt["g_ema"].items() if
                               (k in model_dict and ('down_from_big.0' not in k))}
            model_dict.update(pretrained_dict)
            g_ema.load_state_dict(model_dict)

            discriminator.load_state_dict(ckpt["d"], strict=False)


    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.current_device],
            output_device=args.current_device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.current_device],
            output_device=args.current_device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    # if "lmdb" in args.path:
    #     dataset = MultiResolutionDataset(args.path, transform, args.size)
    #     test_data = MultiResolutionDataset(path=args.test_path, transform=test_transform, resolution=args.size)
    # else:
    dataset = ImageFolder_with_edges(image_root=args.path,edge_root=args.sketches_path, transform=transform)
    test_data = ImageFolder_with_edges(image_root=args.test_path,edge_root=args.sketches_test_path, transform=test_transform)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    test_loader = data.DataLoader(
        test_data,
        batch_size=args.batch,
        sampler=data_sampler(test_data, shuffle=False, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="exe-GAN")

    train(args, loader, test_loader, generator, discriminator, g_optim, d_optim, g_ema, device)

"""
从 train_inpainting_transfer.py版本演化而来
embedding loss
mask 破坏，并且没有 随机权重

加入了 对embedding loss 的调整
就是实验发现，如果单纯的用embedding loss 的话,一些局部区域会变得非常不协调
为了减少这种协调。
实际上就是让网络不要太关注一些边缘区域。
所以引入了一种边缘先验。

#paris
/root/workspace/Workspace/Data/paris_data/paris_train_original
/root/workspace/Workspace/Data/paris_data/paris_eval_gt

#celeba
/root/workspace/Workspace/Data/celeba-256/train
/root/workspace/Workspace/Data/celeba-256/test
"""
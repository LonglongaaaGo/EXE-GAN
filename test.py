import argparse
import torch
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
from op.utils import get_mask,mkdirs,delete_dirs
from fid_eval import test_matrix
from dataset import ImageFolder,ImageFolder_with_mask

from pytorch_fid import fid_score
import os
from models.exe_gan_model import EXE_GAN
from models.stylegan2_co_mod_gan import co_mod_GAN
import random
import numpy as np
from op.mask_generator import co_mod_mask_only


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)



def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def eval_(args, generator, device,mask_root,mask_file,eval_dict,):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    if mask_file == "large_mask":
        dataset = ImageFolder(root=args.path, transform=transform,exe_root=args.exe_path,im_size=(args.size,args.size))
    elif mask_file != "":
        dataset = ImageFolder_with_mask(args.path, mask_root, mask_file, transform,exe_root=args.exe_path,im_size=(args.size,args.size))
    else:
        dataset = ImageFolder(root=args.path, transform=transform,exe_root=args.exe_path,im_size=(args.size,args.size))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=False, distributed=False),
        drop_last=True,
    )
    mask_shapes = [128,128]



    fid_list = []
    U_IDS_score_list = []
    P_IDS_score_list = []
    mae_list = []
    psnr_list = []
    ssim_list = []
    for jj in range(args.repeat_times):
        print("++++"*100)
        print(f"{jj+1}-th  repeat ")
        set_random_seed(jj)
        with torch.no_grad():
            generator.generator.eval()
            for i, datas in tqdm(enumerate(loader)):
                torch.cuda.empty_cache()
                if i>10 and args.debug == True: break
                if mask_file == "large_mask":
                    real_imgs, exemplar = datas
                    real_imgs = real_imgs.to(device)
                    exemplar = exemplar.to(device)
                    gin, gt_local, mask, mask_01, im_ins = get_mask(real_imgs, mask_type="stroke_rect", im_size=args.size,mask_shapes=mask_shapes)
                    # mask_01 = co_mod_mask_only(real_imgs.shape[0], im_size=args.size, device=device)
                    # im_ins = real_imgs * (1 - mask_01)
                elif mask_file != "":  # if we give the masks
                    real_imgs, mask_01, exemplar = datas
                    real_imgs = real_imgs.to(device)
                    mask_01 = mask_01.to(device)
                    im_ins = real_imgs * (1 - mask_01)

                else:
                    real_imgs,exemplar = datas
                    real_imgs = real_imgs.to(device)
                    exemplar = exemplar.to(device)

                    gin, gt_local, mask, mask_01, im_ins = get_mask(real_imgs, mask_type="center", im_size=args.size,mask_shapes=mask_shapes)

                if args.exe_path is None or args.arch == "cmod_gan":
                    completed_img, _, infer_imgs = generator.forward(real_imgs, mask_01)
                else:
                    completed_img, _, infer_imgs = generator.forward(real_imgs, mask_01,infer_imgs=exemplar)

                for j, g_img in enumerate(completed_img):
                    utils.save_image(
                        g_img.add(1).mul(0.5),
                        f"{str(eval_dict)}/{str(i * args.batch + j).zfill(6)}_inpaint.png",
                        nrow=int(1),
                         )

                    utils.save_image(
                        real_imgs[j:j+1].add(1).mul(0.5),
                        f"{str(eval_dict)}/{str(i * args.batch + j).zfill(6)}_gt.png",
                        nrow=int(1),
                     )

                    utils.save_image(
                        im_ins[j:j+1].add(1).mul(0.5),
                        f"{str(eval_dict)}/{str(i * args.batch + j).zfill(6)}_mask.png",
                        nrow=int(1),
                       )

                    utils.save_image(
                        infer_imgs[j:j+1].add(1).mul(0.5),
                        f"{str(eval_dict)}/{str(i * args.batch + j).zfill(6)}_infer.png",
                        nrow=int(1),
                    )

        torch.cuda.empty_cache()
        # test_name = ["fid"]
        fid_value, U_IDS_score, P_IDS_score = fid_score.calculate_P_IDS_U_IDS_given_paths_postfix(path1=eval_dict,
                                                                                                  postfix1="_gt.png",
                                                                                                  path2=eval_dict,
                                                                                                  postfix2="_inpaint.png",
                                                                                                  batch_size=args.batch,
                                                                                                  device=device,
                                                                                                  dims=2048,
                                                                                                  num_workers=args.num_workers)

        print('FID: ', fid_value)
        print('U_IDS_score: ', U_IDS_score)
        print('P_IDS_score: ', P_IDS_score)
        print("fid_score_:%g" % fid_value)

        test_name = ['mae', 'psnr', 'ssim' ]
        out_dic = test_matrix(path1=eval_dict, postfix1="_gt.png"
                              , path2=eval_dict, postfix2="_inpaint.png", test_name=test_name)
        print("mae:%g,psnr:%g,ssim:%g,fid:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim'], fid_value))
        #
        fid_list.append(float(complex(fid_value).real))
        U_IDS_score_list.append(U_IDS_score)
        P_IDS_score_list.append(P_IDS_score)
        mae_list.append(out_dic['mae'])
        psnr_list.append(out_dic['psnr'])
        ssim_list.append(out_dic['ssim'])

    fid_mean = np.array(fid_list).mean()
    U_IDS_score_mean = np.array(U_IDS_score_list).mean()
    P_IDS_score_mean = np.array(P_IDS_score_list).mean()
    mae_mean = np.array(mae_list).mean()
    psnr_mean = np.array(psnr_list).mean()
    ssim_mean = np.array(ssim_list).mean()

    print( f"fid_mean:{fid_mean},U_IDS_score_mean:{U_IDS_score_mean},P_IDS_score_mean:{P_IDS_score_mean},"
           f"mae_mean:{mae_mean},psnr_mean:{psnr_mean},ssim_mean:{ssim_mean}," )

    fid_std = np.array(fid_list).std()
    U_IDS_score_std = np.array(U_IDS_score_list).std()
    P_IDS_score_std = np.array(P_IDS_score_list).std()
    mae_std = np.array(mae_list).std()
    psnr_std = np.array(psnr_list).std()
    ssim_std = np.array(ssim_list).std()

    print(f"fid_std:{fid_std},U_IDS_score_std:{U_IDS_score_std},P_IDS_score_std:{P_IDS_score_std},"
          f"mae_std:{mae_std},psnr_std:{psnr_std},ssim_std:{ssim_std},")



def get_model(args,model_path,psp_path):
    generator = None
    if args.arch == "exe_gan":
        print("model name: exe_gan !!!!!!!!!!!!!!!")
        generator = EXE_GAN(exe_ckpt_path=model_path, psp_ckpt_path=psp_path,size=args.size)
    elif args.arch == "cmod_gan":
        print("model name: exe_gan !!!!!!!!!!!!!!!")
        generator = co_mod_GAN(exe_ckpt_path=model_path)

    return generator

def eval_all():
    device = "cuda"

    parser = argparse.ArgumentParser(description="EXE-GAN tester")

    parser.add_argument("--path", type=str, help="path to the ground-truth images")
    parser.add_argument("--exe_path", type=str, default=None, help="path to the exemplar images")
    parser.add_argument('--arch', type=str, default='exe_gan', help='models architectures (exe_gan | cmod_gan)')
    parser.add_argument("--batch", type=int, default=8, help="batch sizes for each gpu"    )
    parser.add_argument("--eval_dir", type=str, default="./eval_dir", help="path to the output the generated images")
    parser.add_argument("--num_workers",type=int, default=8,help="number of workers", )

    parser.add_argument("--size", type=int, default=256, help="image sizes for the models"    )
    parser.add_argument("--debug",type=bool,default=False,help = "for debugging")
    parser.add_argument("--psp_checkpoint_path", type=str, default="./pre-train/psp_ffhq_encode.pt", help="psp model pretrained model")
    parser.add_argument("--mixing", type=float, default=0.5, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default="./checkpoint/EXE_GAN_model.pt", help="psp model pretrained model")

    # if masked image is not provided, the mask will be generated automatically
    parser.add_argument("--mask_root", type=str, default="",help=" mask root for the irregualr masks")
    parser.add_argument("--mask_file_root", type=str, default="", help="file names for the irregualr masks")
    parser.add_argument("--mask_type", type=str, default="all", help=" mask type: [center,test_2.txt,test_3.txt,test_4.txt,test_5.txt,test_6.txt] ")

    parser.add_argument("--repeat_times", type=int, default=1, help="repeat times to test"    )


    args = parser.parse_args()

    delete_dirs(args.eval_dir)
    mkdirs(args.eval_dir)

    if args.mask_type == "all":
        mask_types = [ "center","test_2.txt", "test_3.txt", "test_4.txt", "test_5.txt", "test_6.txt", ]
    else:
        mask_types = [args.mask_type, ]

    generator = get_model(args, model_path=args.ckpt, psp_path=args.psp_checkpoint_path)
    for mask_type_ in mask_types:
        eval_dict_ = os.path.join(args.eval_dir,mask_type_)
        delete_dirs(eval_dict_)
        mkdirs(eval_dict_)

        if mask_type_ == "center": mask_file = ""
        elif mask_type_ == "large_mask":  mask_file = "large_mask"
        else: mask_file = os.path.join(args.mask_file_root,mask_type_)

        eval_(args, generator, device,args.mask_root,mask_file, eval_dict_)


if __name__ == "__main__":
    eval_all()


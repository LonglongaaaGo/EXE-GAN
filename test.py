import argparse
import torch
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
from op.utils import get_mask,get_completion,mkdirs,delete_dirs
from fid_eval import test_matrix
from dataset import ImageFolder,ImageFolder_with_mask

from pytorch_fid import fid_score
from models.exe_gan import Generator
from Loss.psp_embedding import Psp_Embedding,embedding_loss
from train import  mixing_noise



class EXE_GAN():

    def __init__(self,exe_ckpt_path,psp_ckpt_path,
                 latent = 512,n_mlp = 8,size = 256,channel_multiplier = 2,psp_start_latent=4,num_psp_latent=10,mixing=0.5,device="cuda"):
        """
        :param exe_ckpt_path: EXE GAN checkpoint path
        :param psp_ckpt_path: pSp checkpoint path

        #### parameters shonw below don't need change
        :param latent:  latent size
        :param n_mlp:  the number for the MLP layer
        :param size:  input size
        :param channel_multiplier:
        :param psp_start_latent:
        :param num_psp_latent:  number of the psp latent
        :param mixing: probability
        :param device:  device for GPU
        """
        self.latent = latent
        self.mixing = mixing
        self.device = device
        self.generator = Generator(
            size, latent, n_mlp, channel_multiplier=channel_multiplier,
            psp_start_latent=psp_start_latent, num_psp_latent=num_psp_latent
        ).to(device)

        ##init embedding
        print("start_latent :%d" % psp_start_latent)
        print("n_psp_latent :%d" % num_psp_latent)
        self.psp_embedding = Psp_Embedding(psp_ckpt_path, psp_start_latent, num_psp_latent).to(device)

        print("load models:", exe_ckpt_path)
        ckpt = torch.load(exe_ckpt_path, map_location=lambda storage, loc: storage)

        self.generator.load_state_dict(ckpt["g_ema"])

        self.psp_embedding.eval()
        self.generator.eval()

    def forward(self,real_imgs,mask_01,infer_imgs=None,truncation=1):
        """
        :param real_imgs: with size [b,c,h,w]
        :param mask_01: with size [b,1,h,w]    masked pixel = 1, others = 0
        :param infer_imgs:  with size [b,c,h,w] or None
        :return: inpainted img with size  [b,c,h,w]
        """
        #if infer imgs is None,  copy and get flipped batch
        if infer_imgs == None:
            infer_imgs = torch.flip(real_imgs, dims=[0])
        #
        im_in = real_imgs * (1 - mask_01)
        gin = torch.cat((im_in, mask_01 - 0.5), 1)
        # generate noises
        noise = mixing_noise(real_imgs.shape[0], self.latent, self.mixing, self.device)
        #embedding
        infer_embeddings = self.psp_embedding(infer_imgs)
        #get fake
        fake_img = self.generator(gin, infer_embeddings, noise,truncation=truncation)
        #get completed
        completed_imgs = get_completion(fake_img, real_imgs.detach(), mask_01.detach())

        # img_mask = img*(1-mask) + mask
        return completed_imgs,gin,infer_imgs

    def psp_score(self,image_a,image_b):
        a_embedding = self.psp_embedding(image_a)
        b_embeddings = self.psp_embedding(image_b)
        embedding_score = embedding_loss(a_embedding, b_embeddings)
        return embedding_score


    def get_inherent_stoc(self,real_imgs,mask_01,infer_imgs=None,truncation=1):
        """
                :param real_imgs: with size [b,c,h,w]
                :param mask_01: with size [b,1,h,w]    masked pixel = 1, others = 0
                :param infer_imgs:  with size [b,c,h,w] or None
                :return: inpainted img with size  [b,c,h,w]
                """
        # if infer imgs is None,  copy and get flipped batch
        if infer_imgs == None:
            infer_imgs = torch.flip(real_imgs, dims=[0])
        #
        im_in = real_imgs * (1 - mask_01)
        gin = torch.cat((im_in, mask_01 - 0.5), 1)
        # generate noises
        noise = mixing_noise(real_imgs.shape[0], self.latent, self.mixing, self.device)
        # embedding
        infer_embeddings = self.psp_embedding(infer_imgs)

        trunc_latent = self.generator.mean_latent(4096,device=self.device)

        # get fake
        fake_img = self.generator(gin, infer_embeddings, noise, truncation_latent=trunc_latent,truncation=truncation)

        # get completed
        completed_imgs = get_completion(fake_img, real_imgs.detach(), mask_01.detach())

        img_mask = real_imgs * (1 - mask_01) + mask_01
        return completed_imgs, gin, infer_imgs,img_mask


    def latent_mixing(self,latent_a,latent_b,index):
        """
        :param latent_a:
        :param latent_b:
        :param index: from [0, len(latent_a)].
        :return:
        """
        assert len(latent_a.shape) == len(latent_b.shape)
        assert index <= latent_b.shape[1] and index>=0
        latent_new = latent_b.clone().detach()
        latent_new[:,:index] = latent_a[:,:index].clone().detach()
        return latent_new

    def mixing_forward(self, real_img, mask_01, infer_imgs_a, infer_imgs_b):
        """
        :param gin: torch.tensor =>[b,c,h,w] =>[-1,1]        b=1
        :param real_img: torch.tensor =>[b,c,h,w] =>[-1,1]
        :param mask_01: torch.tensor =>[b,1,h,w] =>[0,1]
        :param infer_imgs_a: torch.tensor =>[b,c,h,w] =>[-1,1]
        :param infer_imgs_b: torch.tensor =>[b,c,h,w] =>[-1,1]
        :param device: "cpu" or "cuda"
        :return:
        """
        # real_img
        # mask_01 =>[1,1,h,w] =>[0,1]
        # infer_imgs =>[b,c,h,w] =>[-1,1]
        im_in = real_img * (1 - mask_01)
        gin = torch.cat((im_in, mask_01 - 0.5), 1).to(self.device)
        real_img = real_img.to(self.device)
        mask_01 = mask_01.to(self.device)
        infer_imgs_a = infer_imgs_a.to(self.device)
        infer_imgs_b = infer_imgs_b.to(self.device)

        noise = mixing_noise(gin.size(0), self.latent, prob=self.mixing, device=self.device)

        with torch.no_grad():
            # size [batch,latent_num,512]
            infer_embeddings_a = self.psp_embedding(infer_imgs_a)
            infer_embeddings_b = self.psp_embedding(infer_imgs_b)

            out_list = []
            for jj in range(infer_embeddings_a.shape[1] + 1):
                # if jj == 3: break
                mixed_latent = self.latent_mixing(infer_embeddings_a, infer_embeddings_b, jj)

                fake_img = self.generator(gin, mixed_latent, noise)
                completed_img = get_completion(fake_img, real_img.detach(), mask_01.detach())
                out_list.append(completed_img)
            Tensor = torch.cat(out_list, dim=0)
        return Tensor








def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def eval_(args, generator, device,eval_dict,):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    if args.mask_root != "":
        dataset = ImageFolder_with_mask(args.path, args.mask_root, args.mask_file, transform)
    else:
        dataset = ImageFolder(root=args.path, transform=transform)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=False, distributed=False),
        drop_last=True,
    )

    mask_shapes = [128,128]
    iter_num = int(args.sample_num/args.batch)
    main_mae,main_psnr,main_ssim,main_fid_value, main_U_IDS_score, main_P_IDS_score=0,0,0,0,0,0
    with torch.no_grad():
        generator.generator.eval()
        for i, datas in tqdm(enumerate(loader)):
            torch.cuda.empty_cache()
            if i > iter_num: break

            real_imgs = datas.to(device)

            gin, gt_local, mask, mask_01, im_ins = get_mask(real_imgs, mask_type="stroke_rect", im_size=args.size,mask_shapes=mask_shapes)
            completed_img, _, infer_imgs = generator.forward(real_imgs, mask_01)

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

    test_name = ['mae', 'psnr', 'ssim', ]
    out_dic = test_matrix(path1=eval_dict, postfix1="_gt.png"
                          , path2=eval_dict, postfix2="_inpaint.png", test_name=test_name)

    print("mae:%g,psnr:%g,ssim:%g,fid:%g" % (out_dic['mae'], out_dic['psnr'], out_dic['ssim'], fid_value))


    print("Main!!!mae:%g,psnr:%g,ssim:%g,fid:%g,U_IDS:%g,P_IDS:%g" %
          (main_mae, main_psnr, main_ssim, main_fid_value,main_U_IDS_score,main_P_IDS_score))






def get_model(name,model_path,psp_path):
    generator = None
    if name == "exe_gan":
        print("model name: exe_gan !!!!!!!!!!!!!!!")
        generator = EXE_GAN(exe_ckpt_path=model_path, psp_ckpt_path=psp_path)

    return generator

def eval_all():
    device = "cuda"

    parser = argparse.ArgumentParser(description="EXE-GAN tester")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='exe_gan', help='models architectures (co_mod_gan | exe_gan)')
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpu"
    )
    parser.add_argument("--sample_num", type=int, default=10000, help="path to the lmdb dataset")
    parser.add_argument("--eval_dir", type=str, default="./eval_dir", help="path to the output the generated images")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers",
    )

    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the models"
    )

    parser.add_argument("--psp_checkpoint_path", type=str, default="./pre-train/psp_ffhq_encode.pt", help="psp model pretrained model")
    parser.add_argument("--mixing", type=float, default=0.5, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default="./checkpoint/EXE_GAN_model.pt", help="psp model pretrained model")

    # if masked image is not provided, the mask will be generated automatically
    parser.add_argument("--mask_root", type=str, default="",
                        help="example:/home/k/Data/mask/mask root for the irregualr masks")
    parser.add_argument("--mask_file", type=str, default="", help="file names for the irregualr masks")

    args = parser.parse_args()

    delete_dirs(args.eval_dir)
    mkdirs(args.eval_dir)

    generator = get_model(args.arch, model_path=args.ckpt, psp_path=args.psp_checkpoint_path)
    eval_(args, generator, device, args.eval_dir)


if __name__ == "__main__":
    eval_all()

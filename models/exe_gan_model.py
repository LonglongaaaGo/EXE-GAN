
import torch
from Loss.psp_embedding import Psp_Embedding,embedding_loss
from op.utils import get_completion
from models.exe_gan_nets import Generator
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
        if size== 512:
            num_psp_latent = 12
        self.latent = latent
        self.mixing = mixing
        self.device = device
        self.generator = Generator(
            size, latent, n_mlp, channel_multiplier=channel_multiplier,
            psp_start_latent=psp_start_latent, num_psp_latent=num_psp_latent
        ).to(device)
        print(self.generator)
        
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

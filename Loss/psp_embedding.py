import torch
from torch import nn

from models.encoders import psp_encoders
from argparse import Namespace
from models.svgl import SVGL_layer

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def embedding_loss(z_id_X, z_id_Y,tags=None):
    l2 = nn.MSELoss()

    if tags == None:
        return l2(z_id_X, z_id_Y).mean()
    else:
        loss = 0
        count =0
        for i in range(min(len(tags),z_id_X.shape[2])):
            if tags[i] == 1:
                loss = loss + l2(z_id_X[:,i,:], z_id_Y[:,i,:]).mean()
                count+=1
        if count == 0:
            return torch.zeros((1)).mean()
        loss = loss / float(count)
        return loss


class Psp_Embedding(nn.Module):
    def __init__(self,psp_encoder_path,start_latent=None,n_psp_latent=None,all_psp_latent=14):
        super(Psp_Embedding, self).__init__()
        print('Loading ResNet ArcFace')

        self.psp_encoder = None
        if psp_encoder_path is not None:
            self.psp_encoder = self.get_psp_encoder(psp_encoder_path=psp_encoder_path).eval().cuda()
        if start_latent == None: self.start_latent = 0
        else:  self.start_latent = start_latent
        if n_psp_latent == None: self.n_psp_latent = self.psp_opts.n_styles
        else: self.n_psp_latent = n_psp_latent
        self.all_psp_latent = all_psp_latent


    def get_keys(self, d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

    def get_psp_latents(self, img):
        # with torch.no_grad():
        codes = self.psp_encoder(img, self.n_psp_latent, self.start_latent)
        # normalize with respect to the center of an average face
        if self.psp_opts.start_from_latent_avg:
            if self.psp_opts.learn_in_w:
                codes = codes + self.latent_avg[self.latent_avg.shape[0] - codes.shape[1]:, :].repeat(
                    codes.shape[0], 1)
            else:
                # codes = codes + self.latent_avg[:codes.shape[1],:].repeat(codes.shape[0], 1, 1)
                # codes = codes + self.latent_avg[self.latent_avg.shape[0]-codes.shape[1]:,:].repeat(codes.shape[0], 1, 1)
                codes = codes + self.latent_avg[self.start_latent:self.start_latent + self.n_psp_latent, :].repeat(
                    codes.shape[0], 1, 1)

        return codes

    def get_psp_all_latents(self, img):
        # with torch.no_grad():
        codes = self.psp_encoder(img, self.all_psp_latent, 0)
        # normalize with respect to the center of an average face
        if self.psp_opts.start_from_latent_avg:
            if self.psp_opts.learn_in_w:
                codes = codes + self.latent_avg[self.latent_avg.shape[0] - codes.shape[1]:, :].repeat(
                    codes.shape[0], 1)
            else:
                # codes = codes + self.latent_avg[:codes.shape[1],:].repeat(codes.shape[0], 1, 1)
                # codes = codes + self.latent_avg[self.latent_avg.shape[0]-codes.shape[1]:,:].repeat(codes.shape[0], 1, 1)
                # codes = codes + self.latent_avg[0:self.start_latent + self.n_psp_latent, :].repeat(
                #     codes.shape[0], 1, 1)
                codes = codes + self.latent_avg[0:self.all_psp_latent, :].repeat(
                    codes.shape[0], 1, 1)

        return codes

    def get_psp_verse_latents(self, img):
        #得到剩余的其他latent
        codes = self.psp_encoder(img, self.n_psp_latent, self.start_latent,label="adverse")
        # normalize with respect to the center of an average face
        if self.psp_opts.start_from_latent_avg:
            if self.psp_opts.learn_in_w:
                codes = codes + self.latent_avg[self.latent_avg.shape[0] - codes.shape[1]:, :].repeat(
                    codes.shape[0], 1)
            else:
                # codes = codes + self.latent_avg[:codes.shape[1],:].repeat(codes.shape[0], 1, 1)
                # codes = codes + self.latent_avg[self.latent_avg.shape[0]-codes.shape[1]:,:].repeat(codes.shape[0], 1, 1)
                # codes = codes + self.latent_avg[0:self.start_latent + self.n_psp_latent, :].repeat(
                #     codes.shape[0], 1, 1)

                out_latent_avg_a =  self.latent_avg[0:self.start_latent,:]
                out_latent_avg_b =  self.latent_avg[self.start_latent + self.n_psp_latent:self.all_psp_latent,:]
                ave_latent = torch.cat([out_latent_avg_a, out_latent_avg_b], dim=0).repeat(codes.shape[0], 1, 1)

                artual_num = self.all_psp_latent - self.n_psp_latent
                codes = codes[:,0:artual_num,:] + ave_latent

        return codes

    def get_psp_encoder(self, psp_encoder_path):
        # update test options with options used during training
        ckpt = torch.load(psp_encoder_path, map_location='cpu')
        self.psp_opts = ckpt['opts']
        self.psp_opts['n_styles'] = 18
        self.psp_opts = Namespace(**self.psp_opts)

        encoder = None
        if self.psp_opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.psp_opts)
        print('Loading pSp from checkpoint: {}'.format(psp_encoder_path))
        encoder.load_state_dict(self.get_keys(ckpt, 'encoder'), strict=True)
        self.__load_latent_avg(ckpt)

        requires_grad(encoder, False)
        # self.latent_avg.requires_grad = False

        return encoder

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].cuda()
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def forward(self, target_img,label="part",weight_map=None):
        if weight_map is not None:
            target_img = SVGL_layer.ada_piexls(target_img,weight_map)

        if label == "part":
            self.psp_styles = self.get_psp_latents(target_img)
        elif label == "verse_part":
            self.psp_styles = self.get_psp_verse_latents(target_img)
        else:
            self.psp_styles = self.get_psp_all_latents(target_img)

        return self.psp_styles



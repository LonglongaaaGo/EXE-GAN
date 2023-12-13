
import torch
from torch import nn
from torchvision.models import resnet101
import torch.nn.functional as F
from models.svgl import SVGL_layer

class IDLoss(nn.Module):
    def __init__(self,model_path):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')

        self.Z = resnet101(num_classes=256).eval()
        self.Z.requires_grad_(False)
        self.Z.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.Z.cuda()
        self.l1 = nn.L1Loss()


    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def forward(self, target_img, source_img,weight_map=None):
        if weight_map is not None:
            target_img = SVGL_layer.ada_piexls(target_img, weight_map)

        z_id = self.Z(F.interpolate(source_img, size=112, mode='bilinear'))
        z_id = F.normalize(z_id)
        z_id = z_id.detach()

        output_z_id = self.Z(F.interpolate(target_img, size=112, mode='bilinear'))
        output_z_id = F.normalize(output_z_id)

        id_loss = self.id_loss(z_id, output_z_id)

        return id_loss

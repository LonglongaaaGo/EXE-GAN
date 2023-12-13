"""
SVGL layer
Spatial variant gradient layer
"""


from torch.autograd import Function

class SVGL_layer(Function):
    """
    Spatial variant gradient layer (SVGL)
    used to adjust the loss for each pixel.
    it means that pixels on the edges should be not emphasised during training.
    """
    @staticmethod
    def forward(ctx,input,loss_maps):
        """
        :param input: [batch,channel,height,width] (input image)
        :param loss_maps:  [batch,1,height,width] (corresponding weight map)
        :return: input (without any change)
        """
        ctx.save_for_backward(input,loss_maps)

        return input

    @staticmethod
    def backward(ctx,grad_output):
        input,loss_weights = ctx.saved_tensors  # 获取前面保存的参数,也可以使用self.saved_variables

        d_input = grad_output * loss_weights

        return d_input,None

    def ada_piexls(input,loss_maps):
        return SVGL_layer.apply(input,loss_maps)


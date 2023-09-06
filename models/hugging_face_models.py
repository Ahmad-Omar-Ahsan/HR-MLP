from timm import create_model
import torch
import numpy as np
from fvcore.nn import FlopCountAnalysis


def get_efficientVIT_MIT(pretrained,num_classes):
    m = create_model(
        model_name='efficientvit_b0',
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return m


def get_fastvit(pretrained,num_classes):
    m = create_model(
        model_name='fastvit_t8',
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return m

if __name__ =='__main__':
    img = torch.ones([3, 3, 32, 32])
    model = get_efficientVIT_MIT(pretrained=False, num_classes=10)
    # print(list_models())
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    flops = FlopCountAnalysis(model, img)
    print(f"Number of flops: {flops.total()}")
    # out_img = model(img)
    
    # print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
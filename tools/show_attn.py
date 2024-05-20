import sys, os
sys.path.append(os.path.abspath('.'))

from matplotlib import pyplot as plt
import numpy as np
import torch
from math import isqrt

from models.MixFormerClas import MixFormerClas

def get_attention(name, activations):
    def hook(model, input, output):
        activations[name] = output[1].detach()
    return hook

if __name__ == "__main__":

    m = MixFormerClas(use_mlp=False)
    m.load_state_dict(torch.load("pretrained/no_mlp_108e.pth", map_location="cpu"))

    activations = {}

    m.enc.block1[0].attn.register_forward_hook(get_attention('attn10', activations))
    m.enc.block1[1].attn.register_forward_hook(get_attention('attn11', activations)) # 1x 4096 = 64x64
    m.enc.block2[0].attn.register_forward_hook(get_attention('attn20', activations))
    m.enc.block2[1].attn.register_forward_hook(get_attention('attn21', activations)) # 2x 1024 = 32x32
    m.enc.block3[0].attn.register_forward_hook(get_attention('attn30', activations))
    m.enc.block3[1].attn.register_forward_hook(get_attention('attn31', activations)) # 5x  256 = 16x16
    m.enc.block4[0].attn.register_forward_hook(get_attention('attn40', activations))
    m.enc.block4[1].attn.register_forward_hook(get_attention('attn41', activations)) # 8x   64 =  8x8

    d = np.load("data/train_b64/b00000.npz")

    idx = 12
    x_rgb = d["x"][idx].swapaxes(0,1).swapaxes(1,2)
    x_torch = torch.from_numpy(d["x"][idx:idx+1]-127.5).to(dtype=torch.float32)

    f = m.enc(x_torch)

    full_res = {k: torch.nn.functional.interpolate(
                        activations[k][0].permute(0,2,1).reshape(
                            activations[k][0].shape[0],
                            activations[k][0].shape[2],
                            isqrt(activations[k][0].shape[1]),
                            isqrt(activations[k][0].shape[1])
                        ),
                        x_rgb.shape[:-1],
                        mode="bilinear",
                        align_corners=True).norm(dim=1)
                for k in activations}

    #print([full_res[k].shape for k in full_res])

    # figs = [plt.subplots(8, 8, figsize=(6,6)) for _ in range(8)]
    heads = [1,1,2,2,5,5,8,8]
    figs = [plt.subplots(1, h, figsize=(int(h*3), 3)) for h in heads]

    # for k, (fig, axs) in zip(full_res, figs):
    #     for r in range(8):
    #         for c in range(8):
    #             axs[r,c].imshow(full_res[k][0][8*r+c])
    #     fig.tight_layout()
    # plt.show()

    for i, (k, (fig, axs)) in enumerate(zip(full_res, figs)):
        if heads[i]>1:
            for h in range(heads[i]):
                axs[h].imshow(full_res[k][h].unsqueeze(-1)*x_rgb/255.)
        else:
            axs.imshow(full_res[k][0].unsqueeze(-1)*x_rgb/255.)
        fig.suptitle(k)
        fig.tight_layout()
    plt.show()
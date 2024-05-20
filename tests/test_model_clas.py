import sys, os
sys.path.append(os.path.abspath('.'))

from models.MixFormerClas import MixFormerClas

if __name__ == "__main__":
    import torch

    m = MixFormerClas()

    x = torch.zeros(1,3,512,1024)

    o = m(x)

    print(o.shape)
from models.mix_transformer_penc import mit_b0
from models.segformer_head import SegFormerHead

if __name__ == "__main__":
    import torch

    b = mit_b0()
    
    d = SegFormerHead(19)

    x = torch.zeros(1,3,512,1024)

    f = b(x)
    o = d(f)

    print([e.shape for e in f], o.shape)
import sys, os
sys.path.append(os.path.abspath('.'))

import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse
from copy import deepcopy
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from shutil import rmtree
from tensorboardX import SummaryWriter
from tqdm import tqdm
import wandb

from models.MixFormerClas import MixFormerClas
from utils.class_names_imagenet import lab_dict
from utils.prebatched_dataset import PrebatchedDataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('The input cannot be interpreted as boolean.')

def to_rgb(ts):
    ts = ts.cpu()
    return Image.fromarray(np.round((ts.transpose(0,1).transpose(1,2)+127.5).numpy()).astype(np.uint8))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=1024)
    parser.add_argument("--batch_scale", type=int, default=2)
    parser.add_argument("--dataloader_workers", type=int, default=3)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--use_prebatched_train", type=str2bool, default=False)

    parser.add_argument("--data_root", type=str, default="D:/Datasets/imagenet_full/ILSVRC/Data/CLS-LOC/")
    parser.add_argument("--logdir", type=str, default="logs/")
    parser.add_argument("--rname", type=str, default="pretrain")
    parser.add_argument("--rname_suffix", type=str, default="")

    parser.add_argument("--model_smooth_freq", type=int, default=20)
    parser.add_argument("--model_smooth_rate", type=float, default=.85)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-2)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gradient_clip", type=float, default=5.)
    parser.add_argument("--label_smoothing", type=float, default=.11)

    parser.add_argument("--model_use_mlp", type=str2bool, default=True)
    parser.add_argument("--model_penc_after_attn", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])

    args = parser.parse_args()

    # >>> WANDB INIT
    rname = args.rname+"_"+args.rname_suffix if args.rname_suffix != "" else args.rname
    args.logdir = os.path.join(args.logdir, rname).replace("\\", '/').rstrip("/")

    if os.environ.get('WANDB_API_KEY') is not None:
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
    run = wandb.init(project="depthformer", entity="barbafrank", name=rname)
    args.logdir += "_"+run.id
    wandb.config.update(args)
    # <<< WANDB INIT 

    print("Global configuration as follows:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    device = 'cuda' if (torch.cuda.is_available() and args.device == "gpu") else 'cpu'

    rmtree(args.logdir, ignore_errors=True)
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(args.logdir, flush_secs=.5)

    m = MixFormerClas(use_mlp=args.model_use_mlp, penc_after_attn=args.model_penc_after_attn)
    m.to(device)
    old_dict = deepcopy(m.state_dict())
    
    l = torch.nn.CrossEntropyLoss(label_smoothing=.11)
    l.to(device)

    optim = torch.optim.AdamW(m.get_params(args.lr, args.wd))
    gscaler = torch.cuda.amp.GradScaler()
    lrscheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=25),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs-25, eta_min=1e-5)],
        milestones=[25]
    )

    ttransforms = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((.5,.5,.5), (.5/127.5, .5/127.5, .5/127.5))])
    vtransforms = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((.5,.5,.5), (.5/127.5, .5/127.5, .5/127.5))])

    if args.use_prebatched_train:
        tset = PrebatchedDataset(os.path.join(args.data_root, "train_b64"))
        tloader = DataLoader(tset, args.batch_size//64, shuffle=True, num_workers=args.dataloader_workers, drop_last=True, pin_memory=args.pin_memory)
        len_tset = len(tset)//(args.batch_size//64)
    else:
        tset = datasets.ImageFolder(os.path.join(args.data_root, "train"), transform=ttransforms)
        tloader = DataLoader(tset, args.batch_size, shuffle=True, num_workers=args.dataloader_workers, drop_last=True, pin_memory=args.pin_memory)
        len_tset = len(tset)//args.batch_size

    vset = datasets.ImageFolder(os.path.join(args.data_root, "val"), transform=vtransforms)
    vloader = DataLoader(vset, args.val_batch_size, shuffle=False, num_workers=args.dataloader_workers, pin_memory=args.pin_memory)

    i = 0
    for e in range(args.epochs):
        m.train()
        optim.zero_grad()
        for x, y in tqdm(tloader, total=len_tset, desc="Training Epoch %d/%d"%(e+1, args.epochs)):

            if args.use_prebatched_train:
                x, y = x.flatten(0,1).to(device, dtype=torch.float32), y.flatten(0,1).to(device, dtype=torch.long)
            else:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)

            with torch.cuda.amp.autocast():
                o = m(x)
                loss = l(o,y)
                scaled_loss = loss/args.batch_scale

            gscaler.scale(scaled_loss).backward()

            i += 1

            if i%args.batch_scale == 0:
                # unscale the gradients for clipping
                gscaler.unscale_(optim)

                nc = 0
                ns = 0
                wc = 0
                ws = 0
                for p in m.parameters():
                    wc += 1
                    ws += p.detach().norm()
                    if p.grad is not None:
                        n = p.grad.norm()
                        nc += 1
                        ns += n
                        if n > args.gradient_clip:
                            s = args.gradient_clip/n
                            p.grad *= s

                if not torch.isnan(ws/wc):
                    writer.add_scalar("norms/weight", ws/wc, i)
                    wandb.log({"norms/weight": ws/wc}, step=i)

                if not torch.isnan(ns/nc):
                    writer.add_scalar("norms/grad", ns/nc, i)
                    wandb.log({"norms/grad": ns/nc}, step=i)
                
                #optim.step()
                gscaler.step(optim) # since the gradients are unscaled no scaling will occur, but still skipped in case of wrong values
                optim.zero_grad()
                gscaler.update()

            if i%args.model_smooth_freq == 0:
                new_dict = m.state_dict()
                for k in new_dict:
                    new_dict[k] = args.model_smooth_rate*old_dict[k] + (1-args.model_smooth_rate)*new_dict[k]
                m.load_state_dict(new_dict)
                old_dict = deepcopy(new_dict)

            with torch.no_grad():
                p = o.argmax(dim=1)
                acc = (y==p).float().mean()

                if not np.isnan(loss.item()):
                    writer.add_scalar("train/loss", loss.item(), i)
                    wandb.log({"train/loss": loss.item()}, step=i)
                
                if not torch.isnan(acc):
                    writer.add_scalar("train/acc", 100*acc, i)
                    wandb.log({"train/acc": 100*acc}, step=i)
                
                writer.add_scalar("train/lr", optim.param_groups[0]['lr'], i)
                wandb.log({"train/lr": optim.param_groups[0]['lr']}, step=i)

        writer.add_image("train/input", (x[0]+127.5)/255., i)
        wimage = wandb.Image(to_rgb(x[0]), caption="Prediction: %s, Label: %s"%(lab_dict[p[0].item()], lab_dict[y[0].item()]))
        wandb.log({"train/input": wimage}, step=i)
        lrscheduler.step()

        if e%10==0:
            m.eval()
            with torch.no_grad():
                conf = torch.zeros((1000,1000))

                for b, (x, y) in enumerate(tqdm(vloader, total=len(vset)//args.val_batch_size+1, desc="Evaluation Epoch %d/%d"%(e+1, args.epochs))):

                    x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)

                    o = m(x)
                    p = o.argmax(dim=1)

                    if b == i%(len(vset)//args.val_batch_size+1):
                        writer.add_image("val/input", (x[0]+127.5)/255, i)
                        wimage = wandb.Image(to_rgb(x[0]), caption="Prediction: %s, Label: %s"%(lab_dict[p[0].item()], lab_dict[y[0].item()]))
                        wandb.log({"val/input": wimage}, step=i)
                    
                    for bid in range(x.shape[0]):
                        conf[p[bid],y[bid]] += 1

                conf /= conf.sum(dim=0, keepdim=True)
                acc = torch.diag(conf).mean()

                writer.add_scalar("val/acc", 100*acc, i)
                wandb.log({"val/acc": 100*acc}, step=i)

                writer.add_image("val/conf", plt.cm.viridis(conf), i, dataformats="HWC")
                wimage = wandb.Image(plt.cm.viridis(conf))
                wandb.log({"val/conf": wimage}, step=i)

        torch.save(m.state_dict(), args.logdir+"/latest.pth")

import os
import importlib
from tqdm import tqdm
from glob import glob

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from data import create_loader 
from loss import loss as loss_module
from .common import timer, reduce_loss_dict
from icecream import ic
from utils.helpers import visualize_run, visualize_single_map

class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        # setup data set and data loader
        self.dataloader = create_loader(args)

        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        self.adv_loss = getattr(loss_module, args.gan_type)()

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        net = importlib.import_module('model.'+args.model)
        
        self.netG = net.InpaintGenerator(args).cuda()
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))

        self.netD = net.Discriminator().cuda()
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))
        
        self.load()
        if args.distributed:
            self.netG = DDP(self.netG, device_ids= [args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids= [args.local_rank], output_device=[args.local_rank])
        
        if args.tensorboard: 
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            

    def load(self):
        try: 
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'G*.pt'))))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0: 
                print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 
        
        try: 
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'D*.pt'))))[-1]
            self.netD.load_state_dict(torch.load(dpath, map_location='cuda'))
            if self.args.global_rank == 0: 
                print(f'[**] Loading discriminator network from {dpath}')
        except: 
            pass
        
        try: 
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.global_rank == 0: 
                print(f'[**] Loading optimizer from {opath}')
        except: 
            pass


    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            torch.save(self.netG.module.state_dict(), 
                os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
            torch.save(self.netD.module.state_dict(), 
                os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()}, 
                os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))
            
    def plot_iter_loss(self, iter_l1_loss, iter_adv_g_loss, iter_adv_d_loss, k=None):
        plt.close()
        plt.plot(list(range(1, len(iter_l1_loss) + 1)), iter_l1_loss, label = "Rec Loss")
        plt.plot(list(range(1, len(iter_adv_g_loss) + 1)), iter_adv_g_loss, label = "Adv G Loss")
        plt.plot(list(range(1, len(iter_adv_d_loss) + 1)), iter_adv_d_loss, label = "Adv D Loss")
        plt.legend()
        plt.title("Training Loss")
        if not os.path.exists('visualizations/losses/'):
                os.makedirs('visualizations/losses/')
        if k is None:
            plt.savefig('visualizations/losses/train_loss.png')
        else:
            plt.savefig('visualizations/losses/train_loss{}.png'.format(k))

    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        if self.args.global_rank == 0: 
            pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
            timer_data, timer_model = timer(), timer()

        if self.args.global_rank == 0:
            iter_l1_loss = []
            iter_adv_g_loss = []
            iter_adv_d_loss = []
        
        if not os.path.exists('visualizations/runs/'):
            os.makedirs('visualizations/runs/')

        for idx in pbar:
            self.iteration += 1
            images, masks, filename = next(self.dataloader)
            images, masks = images.cuda(), masks.cuda()
            images_masked = (images * (1 - masks).float()) + masks

            if self.args.global_rank == 0: 
                timer_data.hold()
                timer_model.tic()

            # in: [rgb(3) + edge(1)]
            pred_img = self.netG(images_masked, masks)
            comp_img = (1 - masks) * images + masks * pred_img

            # reconstruction losses 
            losses = {}
            for name, weight in self.args.rec_loss.items(): 
                losses[name] = weight * self.rec_loss_func[name](pred_img, images)
            
            # adversarial loss 
            dis_loss, gen_loss = self.adv_loss(self.netD, comp_img, images, masks)
            losses[f"advg"] = gen_loss * self.args.adv_weight
            
            # backforward 
            self.optimG.zero_grad()
            self.optimD.zero_grad()
            sum(losses.values()).backward()
            losses[f"advd"] = dis_loss 
            dis_loss.backward()
            self.optimG.step()
            self.optimD.step()

            losses = reduce_loss_dict(losses, self.args.world_size)
            
            if self.args.global_rank == 0:
                timer_model.hold()
                timer_data.tic()
                
                l1_loss = losses['L1'] + losses['Perceptual'] + losses['Style']
                iter_l1_loss.append(l1_loss.item())

                advg_loss = losses['advg']
                iter_adv_g_loss.append(advg_loss.item())

                advd_loss = losses['advd']
                iter_adv_d_loss.append(advd_loss.item())

            
            if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0): 
                pbar.update(self.args.print_every)
                description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                # pbar.set_description((description))
                ic(description)

                k = int(self.iteration // self.args.print_every)
                visualize_run(k, images, 1 - masks, images * (1-masks), pred_img.detach(), comp_img.detach())
                self.plot_iter_loss(iter_l1_loss, iter_adv_g_loss, iter_adv_d_loss, k)
            
            if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0: 
                self.save()
        
        if self.args.global_rank == 0:
            self.plot_iter_loss(iter_l1_loss, iter_adv_g_loss, iter_adv_d_loss)
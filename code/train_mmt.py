import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pdb

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import queue
from collections import deque

# from networks.hierarchical_vnet import VNet
# from networks.vnet_pyramid import VNet
from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
#from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from dataloaders.acdc import acdc, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/cropped_images', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='acdc_mt_16-80-4000_nocl_valloss_1211', help='model_name')
parser.add_argument('--dataset', type=str,  default='la', help='dataset to use')

parser.add_argument('--max_iterations', type=int,  default=8000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')

parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### weight
parser.add_argument('--w0', type=float,  default=0.5, help='weight of p0')
parser.add_argument('--w1', type=float,  default=0.4, help='weight of p1')
parser.add_argument('--w2', type=float,  default=0.05, help='weight of p2')
parser.add_argument('--w3', type=float,  default=0.05, help='weight of p3')
### train
parser.add_argument('--mt', type=int,  default=0, help='mean teacher')
parser.add_argument('--mmt', type=int,  default=1, help='multi-scale mean teacher')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float,  default=1.0, help='temperature')
#maybe change temp to 0.99
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model_la/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
temperature = args.temperature
w0 = args.w0
w1 = args.w1
w2 = args.w2
w3 = args.w3
mt = args.mt
mmt = args.mmt

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def calculate_validation_loss(model, ema_model,val_loader):
        #print('here??')
        negative_valkeys= deque()
        val_losses=[]
        model.eval()  # Set the model in evaluation mode
        ema_model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            # num_batches = len(val_loader)
            #print(len(val_loader))
            for i_batch, val_batch in enumerate(val_loader):
                volume_batch, label_batch = val_batch['image'], val_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = volume_batch + noise 

                # distill: 
                # student bs=4
                student_encoder_output,outputs= model(volume_batch)
                
                # teacher bs=2
                with torch.no_grad():
                    teacher_encoder_output,ema_output = ema_model(ema_inputs)
            
                loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            
                outputs_main_soft = F.softmax(outputs, dim=1)
                loss_seg_dice = losses.dice_loss(outputs_main_soft[:labeled_bs], label_batch[:labeled_bs])

                supervised_loss = 0.5*(loss_seg+loss_seg_dice)

                consistency_weight = get_current_consistency_weight(iter_num//150)
                consistency_dist = consistency_criterion(outputs, ema_output)
                consistency_dist = torch.mean(consistency_dist)
                consistency_loss = consistency_weight * consistency_dist

                # print('cont loss is ', cont_loss)
                cont_weight=0
                if epoch_num<=150:
                    cont_weight=0.01
                # print('cont weight is ', cont_weight)
                # # print('epoch num is ', epoch_num,' weight for contrastive loss is ', cont_weight)
                # print('supervised loss is ', supervised_loss)
                cont_loss=0
            # student_encoder_output=student_encoder_output.detach()
                if len(negative_valkeys)!=0:
                    cont_loss=losses.contrastive_loss(student_encoder_output[3], teacher_encoder_output[3],negative_valkeys)
                if len(negative_valkeys)>=max_queue_size:
                    negative_valkeys.popleft()
                negative_valkeys.append(teacher_encoder_output[3])


                loss = supervised_loss + consistency_loss #+ cont_weight* cont_loss
             
                # loss = supervised_loss + consistency_loss + contrastive_loss #+contrastive_loss here
          
                total_val_loss+=loss.item()
                # val_losses.append(loss.item())

            # Calculate the average validation loss
            # avg_val_loss = total_val_loss  / num_batches

        model.train()  # Set the model back to training mode
        ema_model.train()
        return np.mean(total_val_loss)
    
if __name__ == "__main__":
    #negative_keys=queue.Queue()
    negative_keys= deque()
    average_conts=deque()
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == 'la':
        num_classes = 2
        patch_size = (112, 112, 8)
        db_train = acdc(base_dir=train_data_path,
                           split='train',
                           transform = transforms.Compose([
                              RandomRotFlip(),
                              RandomCrop(patch_size),
                              ToTensor(),
                              ]))
        db_val = acdc(base_dir=train_data_path,
                           split='val',
                           transform = transforms.Compose([
                              RandomRotFlip(),
                              RandomCrop(patch_size),
                              ToTensor(),
                              ]))
        db_test = acdc(base_dir=train_data_path,
                           split='test',
                           transform = transforms.Compose([
                               CenterCrop(patch_size),
                               ToTensor()
                           ]))
        #print('get passed here')
        labeled_idxs = list(range(37))
        # unlabeled_idxs = list(range(37, 75))
        unlabeled_idxs = list()

    labeled_idxs_val = list(range(5))
    unlabeled_idxs_val = list()


    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    batch_sampler_val = TwoStreamBatchSampler(labeled_idxs_val, unlabeled_idxs_val, batch_size, batch_size-labeled_bs)

    def create_model(ema=False):
        # Network definition
        #, pyramid_has_dropout=True
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    #print('train size is ', len(trainloader))
    valloader=DataLoader(db_val, batch_sampler=batch_sampler_val, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)


    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    max_queue_size=50
    

    iter_num = 0
    cont_losses=[]
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    model.train()



    loss_vals=[]
    val_losses=[]
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        epoch_conts=[]
        epoch_loss=[]
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()   

            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise 

            # distill: 
            # student bs=4
            student_encoder_output,outputs = model(volume_batch)
   
            # teacher bs=2
            with torch.no_grad():
                teacher_encoder_output,ema_output = ema_model(ema_inputs)
            ## calculate the loss
            # 1. L_sup bs=2 (labeled)
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)

            # 2. L_con (labeled and unlabeled)
            ### hierarchical consistency
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion(outputs, ema_output)
            consistency_dist = torch.mean(consistency_dist)
            consistency_loss = consistency_weight * consistency_dist


            # contrastive loss
            cont_loss=0
            # student_encoder_output=student_encoder_output.detach()
            if len(negative_keys)!=0:
                cont_loss=losses.contrastive_loss(student_encoder_output[3], teacher_encoder_output[3],negative_keys)
            if len(negative_keys)>=max_queue_size:
                negative_keys.popleft()
            negative_keys.append(teacher_encoder_output[3])

            # total loss
        
            cont_weight=0
            if epoch_num<=150:
                cont_weight=0.01

            loss = supervised_loss + consistency_loss #+ cont_weight* cont_loss
            if iter_num==0:
                epoch_conts.append(cont_loss)
            else:
                epoch_conts.append(cont_loss.item())
        
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            # print(type((sum(average_conts) / len(average_conts))), (sum(average_conts) / len(average_conts)))
            

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/cont_loss', cont_loss, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
       
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        loss_vals.append(np.mean(epoch_loss))
        average_conts.append(np.mean(epoch_conts))
        if len(average_conts)>=50:
                average_conts.popleft()
        cont_losses.append((sum(average_conts) / len(average_conts)))
        val_losses.append(calculate_validation_loss(model, ema_model,valloader))
        if iter_num >= max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

    #loss_values = losses.cpu().detach().numpy()
    num_ep=len(loss_vals)
    iterations = np.linspace(1, num_ep, num_ep, dtype=int)
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, loss_vals, linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    # Save the plot as an image file (e.g., PNG or PDF)
    plt.savefig('training_loss_plot_acdc_mt_16-80-4000_nocl_valloss_1211.png')


    num_ep=len(val_losses)
    iterations = np.linspace(1, num_ep, num_ep, dtype=int)
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, val_losses, linestyle='-')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    # Save the plot as an image file (e.g., PNG or PDF)
    plt.savefig('validation_loss_plot_acdc_mt_16-80-4000_nocl_valloss_1211.png')

    
    # cont_losses = [tensor.cpu().numpy() for tensor in cont_losses]
    # cont_losses = [float(val) for val in cont_losses]
    cont_losses = [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in cont_losses]
    
    
    # print(cont_losses)
    num_ep=len(loss_vals)
    iterations = np.linspace(1, num_ep, num_ep, dtype=int)
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, cont_losses, linestyle='-')
    plt.title('Average Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    # Save the plot as an image file (e.g., PNG or PDF)
    plt.savefig('average_cont_loss_perepoch_acdc_mt_16-80-4000_nocl_valloss_1211.png')



    # plt.figure(figsize=(8, 6))
    # plt.plot(iterations, val_losses, linestyle='-')
    # plt.title('Validation Loss Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.savefig('validation_loss_plot_acdc_mt_16-80-4000_cl_25_30_0.01_scalefactors.png')
    # writer.flush()

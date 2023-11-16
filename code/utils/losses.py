import torch
from torch.nn import functional as F
import numpy as np

# def contrastive_loss(fs,ft_plus,negative_keys):
#     fs=[F.normalize(tensor,dim=1) for tensor in fs]
#     ft_plus=[F.normalize(tensor,dim=1) for tensor in ft_plus]
#     epsilon = 1e-8
#     positive_dot=np.float64(0)
#     t=0.99
    
#     # positive_dot = positive_dot.to(torch.float64)
#     for tensor1, tensor2 in zip(fs, ft_plus):
#         tensor1 = tensor1
#         tensor2 = tensor2
#         tensor1 = tensor1.view(-1)  # Flattening the tensor into a 1D tensor
#         tensor2 = tensor2.view(-1)
#         # print('dot is: ',torch.dot(tensor1,tensor2))
#         positive_dot+=torch.dot(tensor1,tensor2)
#     # print('positive dot is ', positive_dot)
#     positive_dot=0.000001*positive_dot
#     positive_exp=torch.exp(positive_dot)
    
#     # print('positive exp is ', positive_exp.item(),' ', type(positive_exp.item()))
#     negative_sum=0
#     for negative_key in negative_keys:
#         negative_key=[F.normalize(tensor,dim=1) for tensor in negative_key]
#         negative_dot=0
#         for tensor1, tensor2 in zip(fs, negative_key):
#             tensor1 = tensor1
#             tensor2 = tensor2
#             tensor1 = tensor1.view(-1)  # Flattening the tensor into a 1D tensor
#             tensor2 = tensor2.view(-1)
#             # print(torch.dot(tensor1,tensor2))
#             negative_dot += torch.dot(tensor1, tensor2)
#         # print('negative dot is ', negative_dot.item(),' ', type(negative_dot.item()))
#         negative_dot=0.000001*negative_dot
#         negative_exp=torch.exp(negative_dot)
#         negative_sum+=negative_exp
#     # print('positive exp is ', positive_exp)
#     # print('negative sum is ', negative_sum)
#     bit=(positive_exp/(negative_sum+epsilon))
#     print('problem is ', bit )
#     answer=torch.log(bit)
#     print('so loss is ', answer)
#     # print('answer would be, ', answerwould)
#     # answer=0
#     return answer


def contrastive_loss(fs,ft_plus,negative_keys):
    fs=F.normalize(fs,dim=1)
    ft_plus= F.normalize(ft_plus,dim=1)

    fs=fs.view(4, -1)
    ft_plus=ft_plus.view(4, -1)

    positive_dot = 0
    for batch_idx in range(fs.shape[0]):
        tensor1 = fs[batch_idx]
        tensor2= ft_plus[batch_idx]
        positive_dot+=torch.dot(tensor1,tensor2)

    positive_exp=torch.exp(torch.tensor(positive_dot, dtype=torch.float64))

    negative_sum=0
    for negative_key in negative_keys:
        negative_key=F.normalize(negative_key,dim=1)
        negative_key=negative_key.view(4, -1)
        negative_dot=0
        for batch_idx in range(fs.shape[0]):
            tensor1 = fs[batch_idx]
            tensor2= negative_key[batch_idx]
            negative_dot+=torch.dot(tensor1,tensor2)
        negative_exp=torch.exp(torch.tensor(negative_dot, dtype=torch.float64))
        negative_sum+=negative_exp
    answer=-1*torch.log(positive_exp/negative_sum)
    return answer

# def contrastive_loss(fs,ft_plus,negative_keys):
#     #according to multi scale, multi view global local contratsive learning
#     #for semi-supervised cardiac image segmentation
#     length=len(negative_keys)
#     temp=0.1
    
#     fs=F.normalize(fs,dim=1)
#     ft_plus= F.normalize(ft_plus,dim=1)

#     positive_sim=(F.cosine_similarity(fs,ft_plus)/temp).mean()
#     positive_exp=torch.exp(torch.tensor(positive_sim, dtype=torch.float64))

#     negative_sum=0
#     for negative_key in negative_keys:
#         negative_key=F.normalize(negative_key,dim=1)
#         negative_sim=(F.cosine_similarity(fs,negative_key)/temp).mean()
#         negative_exp=torch.exp(torch.tensor(negative_sim, dtype=torch.float64))
#         negative_sum+=negative_exp
    
#     answer=-(1/length)*torch.log(positive_exp/negative_sum)
#     # print('answer is ', answer)
#     return answer


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

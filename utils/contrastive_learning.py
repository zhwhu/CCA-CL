import pdb

import torch
import torch.distributed as dist
import torch.nn as nn
import diffdist.functional as distops
import  torch.nn.functional as F

# def Supervised_NT_xent_n_with_fisher(sim_matrix, labels, fisher_matrix, embedding=None, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
#     '''
#     Compute NT_xent loss with Fisher orthogonal weight adjustment
#     - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
#     - fisher_matrix: (B, B) tensor containing Fisher similarity or orthogonality between samples
#     '''

#     device = sim_matrix.device
#     labels1 = labels.repeat(2)

#     # Normalize sim_matrix for numerical stability
#     logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
#     sim_matrix = sim_matrix - logits_max.detach()

#     B = sim_matrix.size(0) // chunk  # B = B' / chunk

#     # Remove diagonal terms
#     eye = torch.eye(B * chunk).to(device)
#     sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)

#     # Calculate the denominator (normalization term)
#     denom = torch.sum(sim_matrix, dim=1, keepdim=True)

#     # Normalize similarity matrix
#     sim_matrix = sim_matrix / (denom + eps)

#     # Compute Fisher-based weights for each sample pair
#     fisher_weights = torch.zeros_like(sim_matrix)
#     for i in range(B * chunk):
#         for j in range(B * chunk):
#             if labels1[i] == labels1[j]:  # Positive pair
#                 fisher_weights[i, j] = torch.exp(fisher_matrix[i % B, j % B])  # Adjust by Fisher orthogonality
#             else:  # Negative pair
#                 fisher_weights[i, j] = torch.exp(-fisher_matrix[i % B, j % B])  # Adjust by Fisher similarity
#     # for i in range(B * chunk):
#     #     for j in range(B * chunk):
#     #         if labels1[i] == labels1[j]:  # Positive pair
#     #             fisher_weights[i, j] = torch.exp(fisher_matrix[i, j ])  # Adjust by Fisher orthogonality
#     #         else:  # Negative pair
#     #             fisher_weights[i, j] = torch.exp(-fisher_matrix[i , j])  # Adjust by Fisher similarity
#     #             # fisher_weights[i, j] = 1  # Adjust by Fisher similarity

#     # Apply Fisher weights to the similarity matrix
#     weighted_sim_matrix = sim_matrix * fisher_weights

#     # Calculate loss
#     weighted_sim_matrix = -torch.log(weighted_sim_matrix + eps)  # Loss matrix
#     sim_matrix = -torch.log(sim_matrix + eps)  # Loss matrix
#     labels1 = labels1.contiguous().view(-1, 1)
#     mask = torch.eq(labels1, labels1.t()).float().to(device)

#     # Normalize the mask
#     mask = mask / (mask.sum(dim=1, keepdim=True) + eps)

#     # Compute the supervised NT-Xent loss
#     loss1 = 2 * torch.sum(mask * weighted_sim_matrix) / (2 * B)
#     # loss1 = 2 * torch.sum(mask * sim_matrix) / (2 * B)

#     # Positive pairs contribution
#     positive_pairs_loss = (
#         torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)
#     )

#     return positive_pairs_loss + loss1
def Supervised_NT_xent_n_with_fisher(sim_matrix, labels, fisher_matrix, embedding=None, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
    Compute NT_xent loss with Fisher orthogonal weight adjustment
    - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    - fisher_matrix: (B, B) tensor containing Fisher similarity or orthogonality between samples
    '''

    device = sim_matrix.device
    labels1 = labels.repeat(2)
    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    # Normalize sim_matrix for numerical stability
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    # _max, _ = torch.max(fisher_matrix, dim=1, keepdim=True)
    # fisher_matrix = (fisher_matrix - _max.detach()) / 10
    # Compute Fisher-based weights for each sample pair
    fisher_weights = torch.zeros_like(sim_matrix)
    # for i in range(B * chunk):
    #     for j in range(B * chunk):
    #         if labels1[i] == labels1[j]:  # Positive pair
    #             fisher_weights[i, j] = torch.exp(-fisher_matrix[i % B, j % B])  # Adjust by Fisher orthogonality
    #         else:  # Negative pair
    #             fisher_weights[i, j] = torch.exp(fisher_matrix[i % B, j % B])  # Adjust by Fisher similarity
    # for i in range(B * chunk):
    #     for j in range(B * chunk):
    #         fisher_weights[i, j] = torch.exp(fisher_matrix[i % B, j % B])  # Adjust by Fisher orthogonality
    # for i in range(B * chunk):
    #     for j in range(B * chunk):
    #         fisher_weights[i, j] = torch.exp(fisher_matrix[i, j])  # Adjust by Fisher orthogonality
    # for i in range(B * chunk):
    #     for j in range(B * chunk):
    #         if labels1[i] == labels1[j]:  # Positive pair
    #             fisher_weights[i, j] = torch.exp(fisher_matrix[i, j ])  # Adjust by Fisher orthogonality
    #         else:  # Negative pair
    #             fisher_weights[i, j] = torch.exp(-fisher_matrix[i , j])  # Adjust by Fisher similarity
                # fisher_weights[i, j] = 1  # Adjust by Fisher similarity

    # Apply Fisher weights to the similarity matrix
    # weighted_sim_matrix = sim_matrix * fisher_weights
    
    weighted_sim_matrix = sim_matrix + fisher_matrix.to(device)
    # Remove diagonal terms
    eye = torch.eye(B * chunk).to(device)
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)
    weighted_sim_matrix = torch.exp(weighted_sim_matrix / temperature) * (1 - eye)
    # Calculate the denominator (normalization term)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    denom_ = torch.sum(weighted_sim_matrix, dim=1, keepdim=True)
    # Normalize similarity matrix
    sim_matrix = sim_matrix / (denom + eps)
    weighted_sim_matrix = weighted_sim_matrix / (denom_ + eps)
    # Calculate loss
    weighted_sim_matrix = -torch.log(weighted_sim_matrix + eps)  # Loss matrix
    sim_matrix = -torch.log(sim_matrix + eps)  # Loss matrix

    labels1 = labels1.contiguous().view(-1, 1)
    mask = torch.eq(labels1, labels1.t()).float().to(device)
    # Normalize the mask
    mask = mask / (mask.sum(dim=1, keepdim=True) + eps)

    # Compute the supervised NT-Xent loss
    loss1 = 2 * torch.sum(mask * weighted_sim_matrix) / (2 * B)
    # loss1 = 2 * torch.sum(mask * sim_matrix) / (2 * B)

    # Positive pairs contribution
    positive_pairs_loss = (
        torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)
    )

    return positive_pairs_loss + loss1



def top_n_by_column(tensor, n):
    """
    Zero out all elements except for the top n values in each column of the 2D tensor.
    This function is intended to run on CUDA for GPU acceleration.

    Parameters:
    - tensor (Tensor): A 2D tensor on CUDA.
    - n (int): Number of values to keep in each column.

    Returns:
    - Tensor: The modified tensor with the same shape, on CUDA.
    """
    # Ensure the tensor is on CUDA
    tensor = tensor.to('cuda')
    
    # Find the indices of the top n values in each column
    _, top_n_indices = torch.topk(tensor, n, dim=1, largest=True)

    # Create a mask with the same shape as the tensor, on CUDA
    mask = torch.zeros_like(tensor, dtype=torch.bool, device='cuda')
    
    # Use the indices to mark the positions of the top n values in the mask
    # scatter_ dimensions: (dim, index, src), where src is the value to be filled
    mask.scatter_(dim=1, index=top_n_indices, src=torch.ones_like(top_n_indices, dtype=torch.bool, device='cuda'))

    # Zero out elements that are not in the top n
    result = tensor.clone()  # Clone to avoid changing the original tensor
    result[~mask] = 0
    
    return result


def JS_MI(sim_matrix, labels, temperature=0.5,eps=1e-8):
    device = sim_matrix.device
    label=labels.repeat(2)
    label = label.contiguous().view(-1, 1)
  #  logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

   # sim_matrix = sim_matrix - logits_max.detach()

    #import pdb
    #pdb.set_trace()
    Mask2 = torch.eq(label, label.t()).float().to(device)
    Mask1 = torch.ne(label, label.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    Mask2 = Mask2 / (Mask2.sum(dim=1, keepdim=True) + eps)

    B = sim_matrix.size(0) // 2
    import pdb
    #pdb.set_trace()
    #eye = torch.eye(B*2 ).to(device)  # (B', B')
    #sim_matrix = sim_matrix  * (1 - eye)
    #sim_matrix = -torch.log(sim_matrix + eps)
    loss1 = 1*torch.sum(torch.log(torch.exp(Mask1*sim_matrix+eps))) / (1*B) +torch.sum(torch.log(torch.exp(Mask2*(1-sim_matrix)+eps)))/(2*B)
    return loss1


def JS_MI_pre(sim_matrix, labels, temperature=0.5, eps=1e-8):
    device = sim_matrix.device
    label = labels#.repeat(2)
    label = label.contiguous().view(-1, 1)

    # import pdb
    # pdb.set_trace()
    Mask1 = torch.eq(label, label.t()).float().to(device)
   # Mask2 = torch.ne(label, label.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    #Mask2 = Mask2 / (Mask2.sum(dim=1, keepdim=True) + eps)

    B = sim_matrix.size(0)  # // chunk
    eye = torch.eye(B).to(device)  # (B', B')
    sim_matrix = sim_matrix * (1 - eye)
    # sim_matrix = -torch.log(sim_matrix + eps)
    loss1 = torch.sum(torch.log(torch.exp(Mask1*sim_matrix+eps))) / B
    return loss1

def Supervised_NT_xent_pp(proto, proto_t, temperature=0.07):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = proto.device
    class_num = proto.size(0)
    z = torch.cat((proto, proto_t), dim=0)

    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    pos_mask = torch.zeros((2 * class_num, 2 * class_num), dtype=torch.bool, device=device)
    pos_mask[:, class_num:].fill_diagonal_(True)
    pos_mask[class_num:, :].fill_diagonal_(True)

    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

    loss = -mean_log_prob_pos.mean()

    return loss

def InfoNce(sim_matrix,temperature=0.1):


    sim_matrix = sim_matrix / temperature        # 应用温度缩放

    # 构造标签：每个样本的正样本是自己（对角线）
    batch_size = sim_matrix.size(0)

    labels = torch.arange(batch_size, device=sim_matrix.device)

    # 使用交叉熵损失最大化对角线上的相似性
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
# def conditional_infonce(zx, zy, labels, tau=0.07, symmetrize=True):
def conditional_infonce(zx, zy, labels, tau=0.07, symmetric=True, min_pos=2):
    """
    条件 InfoNCE（类内分母，且做类大小归一化）：
      l_i = -log  exp(s_ii/tau) / ( (1/|S_i|) * sum_{k in S_i} exp(s_ik/tau) )

    zx, zy: [B, D] 两模态嵌入（会在函数内做 L2 归一化）
    labels: [B] long，同类样本共享同一 label
    tau: 温度
    symmetric: 是否加 y->x 对称项并取平均
    min_pos: 有效行的最小同类计数（含自身）。若 <2 则该行无条件内负样本，通常剔除
    返回: 标量 Tensor（可与其它 loss 直接相加）
    """
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, device=zx.device)
    labels = labels.long().view(-1)

    # 归一化 + 相似度
    zx = F.normalize(zx, dim=1)
    zy = F.normalize(zy, dim=1)
    logits_xy = (zx @ zy.t()) / tau  # [B, B]

    # 同类掩码与计数
    same = labels[:, None].eq(labels[None, :])   # [B,B] bool
    counts = same.sum(dim=1)                     # [B] 每行同类数量(含自身)
    valid = counts >= min_pos                    # 至少有类内负样本

    if not valid.any():
        # 没有有效行则返回 0（保留梯度图）
        return logits_xy.sum() * 0.0

    neg_inf = torch.finfo(logits_xy.dtype).min

    # x->y：分子 = 对角，分母 = 类内 log-mean-exp
    numer_xy = torch.diag(logits_xy)  # [B]
    # 类内 logsumexp
    lse_xy = torch.logsumexp(logits_xy.masked_fill(~same, neg_inf), dim=1)  # [B]
    # 类大小归一化：log-mean-exp = logsumexp - log(|S_i|)
    denom_xy = lse_xy - counts.clamp_min(1).to(logits_xy.dtype).log()
    loss_xy = -(numer_xy - denom_xy)[valid].mean()

    if not symmetric:
        return loss_xy

    # y->x 对称项
    logits_yx = logits_xy.t()
    numer_yx = torch.diag(logits_yx)
    lse_yx   = torch.logsumexp(logits_yx.masked_fill(~same, neg_inf), dim=1)
    denom_yx = lse_yx - counts.clamp_min(1).to(logits_yx.dtype).log()
    loss_yx = -(numer_yx - denom_yx)[valid].mean()

    return 0.5 * (loss_xy + loss_yx)
def Supervised_NT_xent_n(sim_matrix, labels, embedding=None,temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
  #  labels1 = labels
    labels1 = labels.repeat(2)


    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()


    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix

    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    loss1 = 2*torch.sum(Mask1 * sim_matrix) / (2 * B)

    return (torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)) +  loss1#+1*loss2
def Supervised_NT_xent(sim_matrix, labels, embedding=None,temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
  #  labels1 = labels
    labels1 = labels.repeat(2)


    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()


    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix

    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    loss1 = 2*torch.sum(Mask1 * sim_matrix) / (2 * B)

    return loss1#+1*loss2
def Supervised_NT_xent_simb(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, miu=0.3):
    """
        top n
    """
    device = sim_matrix.device
    labels1 = labels.repeat(2)
    labels1 = labels1.contiguous().view(-1, 1)
    mask_yeq = torch.eq(labels1, labels1.t()).float().to(device)

    B = sim_matrix.size(0) // chunk
    eye = torch.zeros((B * chunk, B * chunk), dtype=torch.bool, device=device)
    eye[:, :].fill_diagonal_(True)
    sim_eye = sim_matrix# * (~eye)
    sim_m = (sim_eye + sim_eye.t()) / 2
    sim_yeq = sim_m * mask_yeq
    ele_max, _ = torch.max(sim_m, dim=1, keepdim=True)
    # ele_mean = torch.sum(sim_yeq, dim=1, keepdim=True) / torch.sum(mask_yeq*(~eye), dim=1, keepdim=True)
    ele_mean = torch.sum(sim_yeq, dim=1, keepdim=True) / torch.sum(mask_yeq, dim=1, keepdim=True)
    sim_thresh = (ele_max - miu*(ele_max - ele_mean*0.9)).expand(B * chunk, B * chunk)
    return_thresh = (ele_max - 0.2*(ele_max - ele_mean*0.9)).expand(B * chunk, B * chunk)

    mask_thr = torch.where(sim_m > sim_thresh, 1, 0)
    sim_mask = mask_thr - mask_yeq
    sim_mask = torch.where(sim_mask <= 0, 0, 1)
    simsum =sim_mask.sum(dim=1)

    sim_nonyed = sim_m * (1-mask_yeq)
    sim_nonyed_topk = top_n_by_column(sim_nonyed, n=5)
    return_sim_topk = top_n_by_column(sim_nonyed, n=5)
    return_mask = torch.where(return_sim_topk > return_thresh, 1, 0)
    sim_mask = torch.where(sim_nonyed_topk > sim_thresh, 1, 0)
    b =sim_mask.sum(dim=1)
    mask = sim_mask + mask_yeq

    mask =  torch.where(mask >= 1, 1, 0)
    mask = mask / (mask.sum(dim=1, keepdim=True) + eps)
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    sim_matrix = torch.exp(sim_matrix / temperature) #* (~eye)

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)

    loss2 =  torch.sum(mask * sim_matrix) / (2 * B)
    # loss1 = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)
    # return loss1 + 2 * loss2
    return loss2, return_mask

def Supervised_NT_xent_uni(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
   # labels1 = labels
    labels1 = labels.repeat(2)


    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()


    B = sim_matrix.size(0) // chunk  # B = B' / chunk

   # eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature)# * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix

    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)



    a = 1

    return torch.sum(Mask1 * sim_matrix) / (4 * B)

  #  Loss =  loss1#+1*loss2



   # return Loss
def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)
def Supervised_NT_xent_pre(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
  #  labels1 = labels
    labels1 = labels#.repeat(2)


    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()


    B = sim_matrix.size(0) // chunk  # B = B' / chunk

  #  eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) #* (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix

    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    return torch.sum(Mask1 * sim_matrix) / (2 * B)
def Supervised_NT_xent_proto(sim_matrix, labels1, labels2, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device


    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    logits = sim_matrix - logits_max.detach()


    labels1 = labels1.contiguous().view(-1, 1)
    labels2 = labels2.contiguous().view(-1, 1)
    pos_mask = torch.eq(labels1, labels2.t()).float().to(device)

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

    loss = -mean_log_prob_pos.mean()

    return loss
   # return Loss
def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)
    #sim_matrix = F.cosine_similarity(outputs.unsqueeze(1), outputs.unsqueeze(0), dim=-1)
    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')#这里是sim(z(x),z(x'))

    return sim_matrix

def Sup(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
    labels1 = labels

    if multi_gpu:
        gather_t = [torch.empty_like(labels1) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels1))
    labels1 = labels1.repeat(2)
    #labels2 = labels1.repeat(2)
    print("0",sim_matrix)
    #logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    #print("lll", logits_max.shape, sim_matrix.shape)
    #sim_matrix = sim_matrix - logits_max.detach()
    #I=torch.zeros([sim_matrix.shape[0],sim_matrix.shape[0]])+1
    #I=I.cuda()
    #print("ee1",sim_matrix.shape)

    B = sim_matrix.size(0) // chunk  # B = B' / chunk
    #print("BBB",B,chunk)
    eye = torch.eye(B * chunk).to(device)  # (B', B')
    #sim_matrix = sim_matrix  * (1 - eye)  # remove diagonal
    #print("ee2", sim_matrix.shape)
    #denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    print("1",sim_matrix)
    sim_matrix = -torch.log(torch.max(sim_matrix,eps)[0])*(1-eye)  # loss matrix
    print("2",sim_matrix)
    #print("ee3", sim_matrix.shape)
    labels1 = labels1.contiguous().view(-1, 1)
    #labels2 = labels2.contiguous().view(-1, 1)
    #print("LLLL",labels)
    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    #Mask2 = torch.eq(labels1, labels1.t()).float().to(device)
    #print("mmm",Mask)
    #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    #Mask2 = Mask2 / (Mask2.sum(dim=1, keepdim=True) + eps)
   # print("M",Mask1)
    #print("MMMM", Mask.shape, Mask.shape)

   # loss = torch.sum(Mask * sim_matrix) / (2 * B)
    a = 1
    #b = 1  # all is 1 means 2:1,-0.5&1 is1:2 no，all 1 is 1+1/n:n-1/n
   # print(a,b)
    #print("Ma",Mask.shape,sim_matrix.shape)
    loss1 = torch.sum(Mask1 * sim_matrix) / (2 * B)
    #print(loss1)
    #loss2 = torch.sum(Mask2 * sim_matrix) / (2 * B)
    #print(sim_matrix)
   # print(torch.sum(sim_matrix[:B, :B].diag() + sim_matrix[B:, B:].diag()) / (2 * B),"loss")
    # and balance question
    Loss = a* loss1#+1*loss2

    # loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return Loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)#和我们的不一样

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
           # labels = labels.repeat(2)

            labels = labels.contiguous().view(-1, 1)

    #        print("L",labels.shape,batch_size)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)#对每个数据，将它和所有数据从第一个到尾比较，标签同则1否则0获得对同label的mask标签
         #   print("mask",mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]#2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)#从第一个维度view全部切开，然后再拼起来
       # print("c_f",contrast_feature.shape)#还原
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature#是否考虑不同view，其实我们的就是考虑不同view的版本
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)#计算内积
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()#为了数值稳定性减去值

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)#repeat是扩增操作，横扩增2倍纵扩增两倍几个views repeat几次
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )#生成一个logits_mask，其实就是一个对角元素0其余是1的矩阵
        #print("log",logits_mask.shape)
        mask = mask * logits_mask#去掉和自己的比较

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask#去掉和自己的比较
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))#构建分子除分母
        #print(log_prob.shape,mask.sum(0),mask.sum(1))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#同类别的pair取平均
        #print(mean_log_prob_pos.shape)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()#单个view

        return loss

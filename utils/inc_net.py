import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, CosineLinear_RanPAC
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
import os
from convs.linears import Adapter,MLP_Adapter
import json
import torchvision.transforms as transforms
from utils.toolkit import get_attribute
import difflib
from PIL import Image
import random
random.seed(1993)

def get_convnet(args, pretrained=False):

    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()

    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            # tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='/home/zh/zh/CLPM/ENGINE-main/open_clip_pytorch_model.bin')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )



class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True



class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)
        
    def forward(self, x):
        x = self.convnet.encode_image(x)
        out = self.fc(x)
        return out



class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.class_name='SimpleClipNet'
        self.args=args


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):

        image_features, text_features, logit_scale=self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(self.args, True)


class CCA_CL(BaseNet):
    def __init__(self, args, pretrained=None):
        super().__init__(args, pretrained)
        self.model, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.args=args
        self.freeze(self.model)

        self.beta = 1
        self.decay = 1

        self.fc = None
    
    @property
    def feature_dim(self):
        return self.model.out_dim
        
    def freeze_text(self, model):
        for param in model.text.parameters():
            param.requires_grad = False
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
    def unfreeze(self, model):
        for param in model.parameters():
            param.requires_grad = True


class RunningCovMI:
    def __init__(self, d, device, dtype=torch.float32, lam=1.0):
        self.d = d
        self.device, self.dtype = device, dtype
        self.lam = lam  # 遗忘因子(<=1). 1.0=无遗忘
        self.n = 0.0
        self.mz = torch.zeros(d, device=device, dtype=dtype)
        self.mt = torch.zeros(d, device=device, dtype=dtype)
        self.Szz = torch.zeros(d, d, device=device, dtype=dtype)
        self.Stt = torch.zeros(d, d, device=device, dtype=dtype)
        self.Szt = torch.zeros(d, d, device=device, dtype=dtype)
        self.w = torch.randn(d,10000).to(device='cuda')
    @torch.no_grad()
    def update(self, z_batch: torch.Tensor, t_batch: torch.Tensor):
        # z_batch/t_batch: [B, D], 已在外部 L2 归一更稳
        B = float(z_batch.size(0))


        # 批均值
        mz_b = z_batch.mean(0)
        mt_b = t_batch.mean(0)

        # 中心化
        zc = z_batch - mz_b
        tc = t_batch - mt_b

        # 批内协方差累加（未除自由度）
        self.Szz += zc.t() @ zc
        self.Stt += tc.t() @ tc
        self.Szt += zc.t() @ tc

        # 均值累加（加权）
        self.mz = (self.mz * self.n + mz_b * B) / (self.n + B)
        self.mt = (self.mt * self.n + mt_b * B) / (self.n + B)
        self.n += B

    @torch.no_grad()
    def finalize(self, eps=1e-4):
        dof = max(self.n - 1.0, 1.0)
        Sz = self.Szz / dof
        St = self.Stt / dof
        Szt = self.Szt / dof
        # 岭稳定
        Sz = Sz + eps * torch.eye(self.d, device=self.device, dtype=self.dtype)
        St = St + eps * torch.eye(self.d, device=self.device, dtype=self.dtype)
        return Sz, St, Szt, self.mz.clone(), self.mt.clone()
    def update_stats(self):
        # 指数遗忘
        if self.n > 0:
            self.Szz.mul_(self.lam); self.Stt.mul_(self.lam); self.Szt.mul_(self.lam)
            self.mz.mul_(self.lam);  self.mt.mul_(self.lam)
            self.n *= self.lam
# ============= 2) CCA-MI 头（闭式求解 + 推理） =============
class CCAMIOnline(nn.Module):
    def __init__(self, d, device='cuda', tau=0.01, energy=0.99, rff=None):
        super().__init__()
        self.rff= rff
        self.d = d
        self.center_mu_cc = None
        self.device = device
        self.register_buffer('mu_z', torch.zeros(d))
        self.register_buffer('mu_t', torch.zeros(d))
        self.Wz = None; self.Wt = None
        self.Vr = None; self.Ur = None
        self.rho = None; self.r = None
        self.energy = energy
        # self.log_tau = nn.Parameter(torch.log(torch.tensor(tau)))
        self.tau = tau
        self.class_bias = None  
        self.rlist = []
    @torch.no_grad()
    def fit_from_stats(self, Sz, St, Szt, mu_z, mu_t, eps=1e-4):
        self.mu_z.copy_(mu_z); self.mu_t.copy_(mu_t)
        # 白化
        Lz, Uz = torch.linalg.eigh(Sz); Lt, Ut = torch.linalg.eigh(St)
        Lz = torch.clamp(Lz, min=eps); Lt = torch.clamp(Lt, min=eps)
        Wz = Uz @ torch.diag(Lz.rsqrt()) @ Uz.t()  # Σ^{-1/2}
        Wt = Ut @ torch.diag(Lt.rsqrt()) @ Ut.t()
        # C = Σzz^{-1/2} Σzt Σtt^{-1/2}
        C = Wz @ Szt @ Wt
        K = C @ C.t()              # [D,D]
        evals, V = torch.linalg.eigh(K)
        vals = torch.clamp(evals, min=0.0)
        idx = torch.argsort(vals, descending=True)
        vals = vals[idx]; V = V[:, idx]
        # 能量选 r 
        cum = vals.cumsum(0) / (vals.sum() + 1e-12)
        r = int((cum <= self.energy).sum().item())
        if self.energy == 1:
            r = len(vals)
        # r = 512
        # r= min(r, 64)
        self.rlist.append(r)
        logging.info(self.rlist)
        # print(r, K.shape)
        rho = vals[:r].sqrt()
        Vr = V[:, :r]
        # 文本侧向量
        Ur = (Wt @ (C.t() @ Vr))  # = Σtt^{-1/2} Σtz Σzz^{-1/2} Vr
        rho_safe = rho.clone(); rho_safe[rho_safe < 1e-6] = 1e-6
        Ur = Ur / rho_safe.unsqueeze(0)

        self.Wz, self.Wt = Wz, Wt
        self.Vr, self.Ur = Vr, Ur
        self.rho, self.r = rho, r
        # self.mi_w = (-torch.log1p(-self.rho**2 + 1e-12)).detach()
        # print(rho)
        # print(self.mi_w)

    @torch.no_grad()
    def update_logits_bias_from_priors(self, priors: torch.Tensor, scale=1.0, eps=1e-6):
        self.class_bias = scale * torch.log(priors.to(self.device) + eps)

    @torch.no_grad()
    def predict_batch(self, images, E_all, ret_feat = False):
        # 视觉特征
        z = F.normalize(images, dim=1)  # [B,D][94.63, 88.64, 84.69, 82.55, 79.7, 79.29, 78.07, 77.46, 76.72, 76.6]
        # 文本原型（行归一）
        E = F.normalize(E_all, dim=1)                       # [C,D]
        if self.rff is not None:
            z = self.rff.map(z)
            E = self.rff.map(E)

        # 映射
        zc = z - self.mu_z; ec = E - self.mu_t
        z_hat = zc @ self.Wz; e_hat = ec @ self.Wt
        z_cc = z_hat @ self.Vr; e_cc = e_hat @ self.Ur
        

        z_p = F.normalize(z_cc * self.rho, dim=1)
        E_p = F.normalize(e_cc * self.rho, dim=1)
        # z_p = F.normalize(z_cc * self.mi_w, dim=1)
        # E_p = F.normalize(e_cc * self.mi_w, dim=1)
        # tau = torch.exp(self.log_tau).clamp(1e-4, 1.0)
        # print(self.log_tau, tau)
        # tau = self.log_tau 
        if ret_feat:
            return z_p, E_p
        logits = (z_p @ E_p.t()) / self.tau
        if self.class_bias is not None:
            logits = logits + self.class_bias
        preds = logits.argmax(dim=1)
        return preds, logits


        




class RFF:
    def __init__(self, d, D_out=2048, sigma=None, device='cuda', seed=0):
        """
        D_in: 原始输入维度（CLIP 的 D）
        D_out: RFF 输出维度 (D')
        sigma: RBF 带宽；若 None 则后续调用 map_with_median 会估计
        device: 'cuda' 或 'cpu'
        seed: 随机种子（固定以保证可复现）
        """
        import math
        self.D_in = d
        self.D_out = D_out
        self.device = torch.device(device)
        self.sigma = sigma
        self.seed = seed

        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        # W: [D_in, D_out], 每列对应一个频率向量
        # 注意：采样为 N(0, 1/sigma^2) => W = N(0,1) / sigma
        # 若 sigma=None 先初始化 W_norm 后再 scale
        self.W = torch.randn(d, D_out, generator=g, device=self.device)
        self.b = (2 * math.pi * torch.rand(D_out, device=self.device, generator=g))

        # 若 sigma 已知，立刻缩放 W
        if sigma is not None:
            self.W = self.W / float(sigma)

    def estimate_sigma_median(self, X, subsample=1000):
        """
        基于 X [N, D_in] 估计带宽 sigma（median heuristic）
        取样本子集避免 O(N^2) 计算
        """
        N = X.shape[0]
        if N > subsample:
            idx = torch.randperm(N, device=X.device)[:subsample]
            Y = X[idx]
        else:
            Y = X
        # 计算 pairwise squared dist between Y and X (小子集)
        with torch.no_grad():
            XX = (X**2).sum(dim=1, keepdim=True)  # [N,1]
            YY = (Y**2).sum(dim=1, keepdim=True)  # [m,1]
            D2 = XX - 2.0 * (X @ Y.t()) + YY.t()  # [N, m]
            D2 = D2.view(-1)
            D2 = D2[D2 > 0]
            if D2.numel() == 0:
                med = torch.tensor(1.0, device=X.device)
            else:
                med = torch.median(D2)
            sigma = float(torch.sqrt(med + 1e-12))
        self.sigma = sigma
        # scale W
        self.W = self.W / float(self.sigma)
        return self.sigma

    def map(self, X):
        """
        X: [B, D_in]
        返回: phi_X [B, D_out]
        """
        if X.device != self.device:
            X = X.to(self.device)
        # 如果 sigma 未设置（W 未缩放），报错或需先调用 estimate_sigma_median
        # 这里容忍：若 self.sigma is None，我们就用当前 W 原样（effectively sigma=1）
        proj = X @ self.W  # [B, D_out]
        # proj = torch.nn.functional.relu(X@ self.W)
        proj = proj + self.b  # broadcasting
        phi = (2.0 / self.D_out) ** 0.5 * torch.cos(proj)

        return phi

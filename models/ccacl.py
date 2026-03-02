import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CCAMIOnline, RunningCovMI, RFF, CCA_CL
from models.base import BaseLearner
from utils.toolkit import tensor2numpy,get_attribute,ClipLoss
from utils.data_manager import LaionData
import math
import time
import matplotlib.pyplot as plt
import os
import json
from utils.contrastive_learning import Supervised_NT_xent_pp, get_similarity_matrix, Supervised_NT_xent,Supervised_NT_xent_n
import random
from utils.loss import CosineSimilarityLoss, InfoNCELoss, contrastive_loss
random.seed(1993)
np.random.seed(1993)

num_workers = 8
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args=args
        self.gap_list = []
        self._train_transformer=False
        self._network = CCA_CL(args)
        self.d = 512
        self.batch_size= get_attribute(args,"batch_size", 48)
        self.cca_batch_size= get_attribute(args,"cca_batch_size", 4)
        self.init_lr= get_attribute(args,"init_lr", 0.01)
        self.weight_decay=  get_attribute(args,"weight_decay", 0.0005)
        self.min_lr=  get_attribute(args,"min_lr", 0)
        self.tuned_epoch =  get_attribute(args,"tuned_epoch", 5)
        self._known_classes = 0
        self.rff_d = get_attribute(args,"rff_d", 5)
        D_out = self.rff_d
        self.rff = RFF(d=512, D_out=D_out, sigma=1)
        
        self.stats = RunningCovMI(d=D_out, device='cuda', lam=1)
        self.ccami = CCAMIOnline(d=D_out, device='cuda', tau=0.07, energy=0.999
                                 , rff=self.rff).to('cuda')

        
        self.ccami._network = self._network 
    def after_task(self):
        self._known_classes = self._total_classes

    def _get_text_des(self,dataname='cifar224'):
        des_path = "chat/"+dataname+'_des.json'
        with open(des_path, 'r') as f:
            des_dict = json.load(f)
        self.des_dict = des_dict
        new_des_dict = {}
        for key, value in des_dict.items():
            new_key_value = []
            for k, v in value.items():
                new_key_value.extend(v)
            new_des_dict[key] = new_key_value
        return new_des_dict
    
    def _get_batch_des(self, des_file, classnames):
        batch_des = []
        for classname in classnames:
            batch_des.append(classname + ' with ' + random.choice(des_file[classname]).lower())
        return batch_des

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager=data_manager
        self._network.to(self._device)
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
            source="train", mode="train")
        self.train_dataset=train_dataset
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        if self._cur_task ==0 and self.tuned_epoch > 0:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

            self._network.unfreeze(self._network.model)
            self.first_train(self.train_loader, self.test_loader, train_dataset)
            self._network.freeze(self._network.model)

        self.test_loader = DataLoader(test_dataset, batch_size=self.cca_batch_size, shuffle=False, num_workers=num_workers)
        self.train_loader = DataLoader(train_dataset, batch_size=self.cca_batch_size, shuffle=True, num_workers=num_workers)
        self.stats.update_stats()
        self.train(self.train_loader, self.test_loader, train_dataset)
        
    @torch.no_grad()
    def _ridge(self, S: torch.Tensor, lam: float = 1e-3, relative: bool = False):
        D = S.size(0)
        if relative:
            lam = lam * (torch.trace(S) / D)
        I = torch.eye(D, device=S.device, dtype=S.dtype)
        return S + lam * I
    def first_train(self, train_loader, test_loader, train_dataset):
        self._network.to(self._device)
        factor_lr = 1e-4
        if self.args['optimizer']=='sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr*factor_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer']=='adam': 
            optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        # if self._cur_task == 0:
        #     self.tuned_epoch=6
        # else:
        #     self.tuned_epoch=1
        # # self.tuned_epoch=1
        class_to_label=self.data_manager._class_to_label
        prog_bar =  range(self.tuned_epoch)
        total_labels=class_to_label[:self._total_classes] 
        templates=self.data_manager._data_to_prompt[0]
        
        from utils.toolkit import ClipLoss
        cliploss=ClipLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            correct_clip = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                
                labels=[class_to_label[y] for y in targets]
                texts=[templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features=self._network.model.encode_text(texts)
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features=self._network.model.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)

                logit_scale = self._network.model.logit_scale
                logits = img_feas @ text_feas.T

                texts_clip=[templates.format(inst) for inst in labels]
                clip_text_feas=self._network.model.encode_text(self._network.tokenizer(texts_clip).to(self._device))
                clip_text_norm=clip_text_feas.norm(dim=-1, keepdim=True)
                clip_text_feas = clip_text_feas / clip_text_norm
                clip_loss=cliploss(img_feas, clip_text_feas, logit_scale)
                
                loss = clip_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                correct_clip.append(preds.eq(targets).cpu())
                total += len(targets)
                for p in self._network.parameters():
                    p.grad = None
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader, epoch)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task,epoch + 1,self.args['tuned_epoch'],losses / len(train_loader),0.00, test_acc,  )
            # prog_bar.set_description(info)
            logging.info(info)
        self._network.eval()
    def train(self, train_loader, test_loader, train_dataset):
        self._network.eval()
        self._network.to(self._device)

        E_all = self.build_text_prototypes()  # 你已有函数

        all_feats = []
        all_lable = []
        if self._cur_task >=0:
            for _, images, targets in train_loader:
                with torch.no_grad():
                    z = F.normalize(self._network.model.encode_image(images.cuda()), dim=1)  # [B,D]
                    t = E_all[targets.cuda()]                                       # [B,D]
                    all_feats.append(z)
                    all_lable.append(t)
                    if self.rff != None:
                        z = self.rff.map(z)
                        t = self.rff.map(t)
                self.stats.update(z, t)

            Sz, St, Szt, mu_z, mu_t = self.stats.finalize(eps=1e-4)

            # Sz = self._ridge(Sz, lam=0, relative=False)
            # St = self._ridge(St, lam=0, relative=False)
            
            self.ccami.fit_from_stats(Sz, St, Szt, mu_z, mu_t)
        test_acc = 0

        acc, tot = 0, 0
        for _, images, targets in test_loader:
            z = self._network.model.encode_image(images.cuda())
            preds, logits = self.ccami.predict_batch(z, E_all)
            acc += (preds.cpu() == targets).sum().item()
            tot += targets.size(0)
        test_acc = acc/tot
        
        self.gap_list.append(self._eval_modality_gap(test_loader))
        logging.info("Task {}, Test_acc {:.4f}, gap {}".format(self._cur_task, test_acc, self.gap_list))


    def build_text_prototypes(self):
        text_features = []
        class_to_label=self.data_manager._class_to_label
        templates=self.data_manager._data_to_prompt
        total_labels=class_to_label[:self._total_classes]
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)

            text_features = torch.stack(text_features, dim=0)
        return text_features  

    def _compute_accuracy(self, model, loader, epoch=0):
        self._network.eval()
        class_to_label=self.data_manager._class_to_label
        templates=self.data_manager._data_to_prompt
        total_labels=class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features = self._network.model.encode_image(inputs)
                transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
                transf_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                outputs = (transf_image_features @ transf_text_features.T)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    def _eval_cnn(self, loader):
        self._network.to(self._device)
        self._network.eval()

        E_all = self.build_text_prototypes()

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            z = self._network.model.encode_image(inputs.cuda())
            preds, logits = self.ccami.predict_batch(z, E_all)  # logits: [B, C]

            # 从 logits 取 top-k 索引
            topk_idx = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True).indices  # [B, topk]

            y_pred.append(topk_idx.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_modality_gap(self, loader):
        """
        计算整个 loader 上的平均 Modality Gap:
            gap = 1 - mean( cos( f_img, f_txt(class) ) )
        """
        self._network.to(self._device)
        self._network.eval()

        E_all = self.build_text_prototypes()

        sims_all = []  

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            z = self._network.model.encode_image(inputs)  # [B, D]
            img_feats, e_ = self.ccami.predict_batch(z, E_all, ret_feat=True)
            txt_feats = e_[targets]  # [B, D]

            sim_batch = (img_feats * txt_feats).sum(dim=-1)  # [B]

            sims_all.append(sim_batch)

        sims_all = torch.cat(sims_all, dim=0)  # [N_total]

        mean_sim = sims_all.mean()  # scalar tensor

        gap = 1.0 - mean_sim  # scalar tensor
        gap_value = gap.item()
        
        return round(gap_value, 3)
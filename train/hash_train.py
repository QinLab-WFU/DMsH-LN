from torch.nn.modules import loss
from model.hash_model import DCMHT as DCMHT
from model.hash_model import Hier_Model
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as scio
from New_Loss import *

from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from dataset.dataloader import dataloader
# from hier_loss import HIERLoss
from Triplet.triplet_loss import TripletLoss
from MSLOSS import MultiSimilarityLoss

class Trainer(TrainBase):

    def __init__(self,
                 rank=1):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        # bit = args.output_dim
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        linear = False
        if self.args.hash_layer == "linear":
            linear = True

        self.logger.info("ViT+GPT!")
        HashModel = DCMHT
        # self.supervision = Hier_Model(enmb=512).to(self.rank)
        self.MSL = MultiSimilarityLoss()
        # self.triplet = TripletLoss(margin=0.6)
        # self.supervision = Hier_Model(enmb=self.args.output_dim).to(self.rank)
        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                               writer=self.writer, logger=self.logger, is_train=self.args.is_train, linear=linear).to(
            self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))



        self.model.float()
        # self.supervision.float()
        self.ClassLen = 0
        if self.args.dataset == 'nuswide':
            self.ClassLen = 21
        elif self.args.dataset == 'flickr25k':
            self.ClassLen = 24
        else:
            self.ClassLen = 80
        # self.new_loss = Proxy_Anchor(nb_classes=self.ClassLen,sz_embed=self.args.output_dim).to(self.rank)
        # self.proxy_opt = torch.optim.Adam(self.new_loss.parameters(),lr=1e-5,weight_decay=1e-4)
        # self.hier_loss = HIERLoss(nb_proxies=512, sz_embed=self.args.output_dim)
        # self.supervision_opt = torch.optim.AdamW([
        #     {"params": self.supervision.parameters(), "lr_scale": 1, "weight_decay": 1e-4},
        #     {"params": self.hier_loss.parameters(), "lr_scale": 5e1, "weight_decay": 1e-4},
        # ],
        #     lr=1e-4, eps=1e-4, weight_decay=1e-5
        # )

        self.optimizer = BertAdam([
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
            # {'params': self.new_loss.parameters(), 'lr':self.args.lr}
        ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        # print(self.model)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
                                                            indexFile=self.args.index_file,
                                                            labelFile=self.args.label_file,
                                                            maxWords=self.args.max_words,
                                                            imageResolution=self.args.resolution,
                                                            query_num=self.args.query_num,
                                                            train_num=self.args.train_num,
                                                            seed=self.args.seed)
        self.train_labels = train_data.get_all_label()
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def train_epoch(self, epoch):

        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        times = 0
        for image, text, label, index in self.train_loader:
            if image.shape[0] != 128:  #
                continue
            self.global_step += 1
            times += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            hash_img, hash_text = self.model(image, text)

            # s_i = self.supervision(hash_img)
            # s_t = self.supervision(hash_text)
            # hier_loss_img = self.hier_loss(s_i, label)
            # hier_loss_tex = self.hier_loss(s_t, label)
            # print(hash_text.shape,hash_img.shape)
            # exit()
            img_loss1 = self.MSL(hash_img,label)
            # img_loss2 = self.triplet(label,hash_img,hash_img)
            # text_loss = self.MSL(hash_text,label)
            # text_loss1 = self.triplet(label, hash_img, hash_img)
            text_loss1 = self.MSL(hash_text,label)
            i_t_loss1 = self.MSL(hash_img,label,feat2=hash_text)
            # text_loss2 = self.triplet(label, hash_text, hash_img)
            # img_to_text_loss = self.new_loss(hash_img, label, Cross=True, text=hash_text)

            all_loss += img_loss1 + i_t_loss1 + text_loss1

            # self.supervision_opt.zero_grad()
            self.optimizer.zero_grad()

            tot_hier =  img_loss1 + text_loss1 + i_t_loss1
            tot_hier.backward()

            self.optimizer.step()
            # self.supervision_opt.step()

        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def train(self):
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            self.save_model(epoch)

        self.logger.info(
            f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    def test(self, model='i2t'):
        self.logger.info("test")
        self.change_state(mode="valid")
        query_img, query_txt = super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt = super().get_code(self.retrieval_loader, self.args.retrieval_num,
                                                        )
        # print("get all code")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)


    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt = super().get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt =  super().get_code(self.retrieval_loader, self.args.retrieval_num,)
        # print("get all code")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", map=mAPi2t)
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i", map=mAPt2i)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", map=0):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir,
                                  str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + f'{map:.4f}_.mat'),
                     result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")



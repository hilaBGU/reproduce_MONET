import math
import random
import sys
from time import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
from Models import MONET
from utility.batch_test import data_generator, test_torch
from utility.parser import parse_args


class Trainer(object):
    def __init__(self, data_config, args):
        # argument settings
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]

        self.feat_embed_dim = args.feat_embed_dim
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers
        self.has_norm = args.has_norm
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.lamb = self.regs[1]
        self.alpha = args.alpha # 1.0
        self.beta = args.beta # 0.3
        self.gamma = 1.0
        self.delta = 1.0
        self.omega = args.omega

        self.learn_alpha = args.learn_alpha
        self.learn_beta = args.learn_beta
        self.learn_gamma = args.learn_gamma
        self.learn_delta = args.learn_delta

        ### our code ###
        if self.learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(self.alpha))  # Learnable alpha
        if self.learn_beta:
            self.beta = nn.Parameter(torch.tensor(self.beta))  # Learnable beta
        if self.learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(self.gamma))  # Learnable gamma
        if self.learn_delta:
            self.delta = nn.Parameter(torch.tensor(self.delta))  # Learnable delta
        # if use_omega:
        #     self.omega = nn.Parameter(torch.tensor(self.omega))  # Learnable omega


        ### our code ###
        if args.dataset == "WomenClothing":
            self.user_transform = nn.Linear(128, 14596)
        else:
            self.user_transform = nn.Linear(128, 5028)  # to match the attention size with the user size 128->5028


        self.dataset = args.dataset
        self.model_name = args.model_name
        self.agg = args.agg
        self.target_aware = args.target_aware
        self.cf = args.cf
        self.cf_gcn = args.cf_gcn
        self.lightgcn = args.lightgcn

        self.nonzero_idx = data_config["nonzero_idx"]

        self.image_feats = np.load("data/{}/image_feat.npy".format(self.dataset))
        self.text_feats = np.load("data/{}/text_feat.npy".format(self.dataset))

        self.model = MONET(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.n_layers,
            self.alpha,
            self.beta,
            self.agg,
            self.cf,
            self.cf_gcn,
            self.lightgcn,
            self.gamma,
            self.delta,
            self.omega,
            self.user_transform,
        )

        self.model = self.model.cuda()
        print("Model is on GPU:", next(self.model.parameters()).is_cuda)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model()
        result = test_torch(
            ua_embeddings,
            ia_embeddings,
            users_to_test,
            is_val,
            self.adj,
            self.beta,
            self.target_aware,
        )
        return result

    def train(self):
        nonzero_idx = torch.tensor(self.nonzero_idx).cuda().long().T
        self.adj = (
            torch.sparse.FloatTensor(
                nonzero_idx,
                torch.ones((nonzero_idx.size(1))).cuda(),
                (self.n_users, self.n_items),
            )
            .to_dense()
            .cuda()
        )
        print("Model is on GPU:", next(self.model.parameters()).is_cuda)
        stopping_step = 0

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0.0, 0.0, 0.0, 0.0
            n_batch = data_generator.n_train // args.batch_size + 1
            for _ in range(n_batch):
                self.model.train()
                self.optimizer.zero_grad()

                user_emb, item_emb = self.model()
                users, pos_items, neg_items = data_generator.sample()

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model.bpr_loss(
                    user_emb, item_emb, users, pos_items, neg_items, self.target_aware
                )

                batch_emb_loss = self.decay * batch_emb_loss
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

                del user_emb, item_emb
                torch.cuda.empty_cache()

            self.lr_scheduler.step()

            if math.isnan(loss):
                print("ERROR: loss is nan.")
                sys.exit()

            perf_str = "Pre_Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]" % (
                epoch,
                time() - t1,
                loss,
                mf_loss,
                emb_loss,
                reg_loss,
            )
            print(perf_str)

            if epoch % args.verbose != 0:
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)

            t3 = time()

            if args.verbose > 0:
                perf_str = (
                    "Pre_Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], "
                    "precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]"
                    % (
                        epoch,
                        t2 - t1,
                        t3 - t2,
                        loss,
                        mf_loss,
                        emb_loss,
                        reg_loss,
                        ret["recall"][0],
                        ret["recall"][-1],
                        ret["precision"][0],
                        ret["precision"][-1],
                        ret["hit_ratio"][0],
                        ret["hit_ratio"][-1],
                        ret["ndcg"][0],
                        ret["ndcg"][-1],
                    )
                )
                print(perf_str)

            if ret["recall"][1] > best_recall:
                best_recall = ret["recall"][1]
                stopping_step = 0

                model_dir = "./models"
                model_filename = f"{self.dataset}_{self.model_name}_alpha_{self.learn_alpha}_beta_{self.learn_beta}_gamma_{self.learn_gamma}_delta_{self.learn_delta}_omega_{self.omega}.pth"
                #model_path = os.path.join(model_dir, f"{self.dataset}_{self.model_name}")
                model_path = os.path.join(model_dir, model_filename)
                try:
                    torch.save({self.model_name: self.model.state_dict()}, model_path)
                except FileNotFoundError:
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save({self.model_name: self.model.state_dict()}, model_path)

            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print("#####Early stopping steps: %d #####" % stopping_step)
            else:
                print("#####Early stop! #####")
                break

        self.model = MONET(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.n_layers,
            self.alpha,
            self.beta,
            self.agg,
            self.cf,
            self.cf_gcn,
            self.lightgcn,
            self.gamma,
            self.delta,
            self.omega,
            self.user_transform,
        )
        load_path = os.path.join(model_dir,
                                 f"{self.dataset}_{self.model_name}_alpha_{self.learn_alpha}_beta_{self.learn_beta}_gamma_{self.learn_gamma}_delta_{self.learn_delta}_omega_{self.omega}.pth")
        self.model.load_state_dict(
            torch.load(
                #"./models/" + self.dataset + "_" + self.model_name,
                load_path,
                map_location=torch.device("cpu"),
            )[self.model_name]
        )
        self.model.cuda()
        print("Model is on GPU:", next(self.model.parameters()).is_cuda)
        test_ret = self.test(users_to_test, is_val=False)
        print("Final ", test_ret)
        print("Alpha value: ", self.alpha)
        print("Beta value: ", self.beta)
        print("Gamma value: ", self.gamma)
        print("Delta value: ", self.delta)
        print("Omega value: ", self.omega)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu


if __name__ == "__main__":

      args = parse_args(True)
      set_seed(args.seed)

      config = dict()
      config["n_users"] = data_generator.n_users
      config["n_items"] = data_generator.n_items

      nonzero_idx = data_generator.nonzero_idx()
      config["nonzero_idx"] = nonzero_idx

      print(f"Starting run with Alpha {args.learn_alpha}, Beta {args.learn_beta}, Gamma {args.learn_gamma}, Delta {args.learn_delta}, Omega {args.omega}")
      trainer = Trainer(config, args)
      trainer.train()

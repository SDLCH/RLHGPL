import copy
import torch
import time
import os
from torch import optim, nn
import numpy as np
from pretrain.module.meow import MEOW
from pretrain.utils import set_params
from pretrain.module.preprocess import *
from .tasknet import build_tasknet
from .agent import Agent, extract_states_from_embs
from .reward import *
from sklearn.metrics import f1_score
from pretrain.module.cluster import run_kmeans


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class BaseTask:
    def __init__(self, args, device, sub_num, feats_dim_list, nei_index, adjs, feat):
        self.args = args
        self.device = device
        self.sub_num = sub_num
        self.feats_dim_list = feats_dim_list
        self.nei_index = nei_index

        self.meow_args = set_params(args.dataset)
        self.embed_dim = self.meow_args.embed_dim
        self.hidden_dim = self.meow_args.hidden_dim
        self.output_dim = self.args.n_labels
        self.total_nodes = sum(self.args.type_num)
        self.num_types = len(self.args.type_num)

        adjs = pathsim(adjs, self.meow_args.nei_max)
        self.mask_feat = mask_features(feat, self.meow_args.feat_mask)
        self.adjs_norm = [normalize_adj(adj) for adj in adjs]
        self.mask_adjs = mask_edges(adjs, sub_num, self.meow_args.adj_mask)

    def initialize_optimizer(self):
        param_group = []
        param_group.append({"name": "actor", "params": self.policy.actor.parameters(), "lr": self.args.actor_lr,
                            "weight_decay": self.args.actor_wd})
        param_group.append({"name": "critic", "params": self.policy.critic.parameters(), "lr": self.args.critic_lr,
                            "weight_decay": self.args.critic_wd})
        self.policy_optim = optim.Adam(param_group, eps=1e-5)
        self.tasknet_optim = optim.Adam(self.tasknet.parameters(), lr=self.args.tasknet_lr, weight_decay=self.args.tasknet_wd)

    def initialize_policy(self):
        self.k_per_type = self.args.k_per_type
        self.policy = Agent(self.hidden_dim, self.embed_dim, self.k_per_type, self.num_types, self.args.max_z).to(self.device)

    def initialize_model(self):
        self.model = MEOW(self.feats_dim_list, self.sub_num, self.meow_args.hidden_dim, self.meow_args.embed_dim,
                          self.meow_args.tau, self.adjs_norm, self.meow_args.lam_proto, self.meow_args.dropout,
                          self.nei_index, self.args.dataset).to(self.device)
        model_path = f'./pre_trained_models/{self.args.dataset}.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def initialize_tasknet(self):
        layer = nn.Linear(self.embed_dim, self.output_dim)
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
        self.tasknet = nn.Sequential(layer).to(self.device)


class NodeTask(BaseTask):
    def __init__(self, feats, label, idx_val, idx_test, *args, **kwargs):
        super(NodeTask, self).__init__(*args, **kwargs)
        self.feats = feats
        self.label = label
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.criterion = nn.CrossEntropyLoss()

    def generate_reward(self, prompt=None):
        with torch.no_grad():
            _ = self.model(self.feats, self.mask_feat, self.mask_adjs, self.adjs_norm, self.meow_args.num_cluster,
                           prompt)
            all_type_embs_next = self.model.get_all_state()
            s = extract_states_from_embs(all_type_embs_next,self.all_cluster_labels,self.k_per_type)
            embeds = self.model.get_embeds()
            logits = self.tasknet(embeds[self.idx_train])
            loss = self.criterion(logits, self.train_labels)
        return loss.item(), s

    def train_prompt(self, s, prompt):
        with torch.no_grad():
            action, logprob = self.policy.get_action(s)
            value = self.policy.get_value(s)
            action_3d = action.squeeze(0)
            for i in range(self.num_types):
                prompt[i] += action_3d[i]
        return action_3d, logprob, value, prompt

    def run(self):
        mi_f1s = []
        ma_f1s = []
        best_loss = []

        for i in range(1, self.args.task_num + 1):
            set_random_seed(self.args.seed)
            self.reward_normalizer = RewardNormalizer(self.device)
            self.initialize_model()
            self.initialize_policy()
            self.tasknet = build_tasknet(self.embed_dim, self.output_dim).to(self.device)
            self.initialize_optimizer()

            data_folder_path = f"./data/{self.args.dataset}/{self.args.shot_num}_shot/"
            self.idx_train = torch.load(f"{data_folder_path}/train_{i}_idx.pt").type(torch.long).to(self.device)
            self.train_labels = torch.argmax(self.label[self.idx_train], dim=-1)

            with torch.no_grad():
                _ = self.model(self.feats, self.mask_feat, self.mask_adjs, self.adjs_norm, self.meow_args.num_cluster)
                init_all_embs = self.model.get_all_state()
            self.all_cluster_labels = []
            for emb in init_all_embs:
                cluster_result = run_kmeans(emb, [self.k_per_type], tau=1.0)
                self.all_cluster_labels.append(cluster_result['im2cluster'][0].to(self.device))

            self.model.all_cluster_labels = self.all_cluster_labels

            task_best_val_loss = float('inf')
            patience_cnt = 0
            best_prompt_snapshot = None
            best_tasknet_state = None

            num_episodes = self.args.num_steps // self.args.prompt_steps
            states = torch.zeros([num_episodes, self.args.prompt_steps, self.num_types, self.k_per_type, self.embed_dim]).to(self.device)
            actions = torch.zeros([num_episodes, self.args.prompt_steps, self.num_types, self.k_per_type, self.hidden_dim]).to(self.device)
            logprobs = torch.zeros([num_episodes, self.args.prompt_steps]).to(self.device)
            rewards = torch.zeros([num_episodes, self.args.prompt_steps + 1]).to(self.device)
            values = torch.zeros([num_episodes, self.args.prompt_steps]).to(self.device)
            dones = torch.zeros([num_episodes, self.args.prompt_steps]).to(self.device)

            for epoch in range(1, self.args.total_epochs + 1):
                self.tasknet.eval()
                self.policy.train()
                for ep in range(num_episodes):
                    prompt = [torch.zeros(self.k_per_type, self.hidden_dim).to(self.device) for _ in range(self.num_types)]
                    init_loss, s = self.generate_reward(prompt)
                    rewards[ep, 0] = init_loss
                    for step in range(self.args.prompt_steps):
                        if step == self.args.prompt_steps - 1:
                            dones[ep, step] = 1.0
                        a, lp,  v, prompt = self.train_prompt(s, prompt)
                        states[ep, step] = s
                        actions[ep, step] = a
                        logprobs[ep, step] = lp
                        values[ep, step] = v
                        r_loss, s = self.generate_reward(prompt)
                        rewards[ep, step + 1] = r_loss

                actual_rewards = rewards[:, :-1] - rewards[:, 1:]
                actual_rewards = self.reward_normalizer(actual_rewards)
                advantages, returns = compute_gae_and_return(actual_rewards, values, dones, self.args.gamma,
                                                             self.args.gae_lambda)
                b_obs = states.view(-1, self.num_types, self.k_per_type, self.embed_dim)
                b_actions = actions.view(-1, self.num_types, self.k_per_type, self.hidden_dim)
                b_logprobs = logprobs.view(-1)
                b_advantages = advantages.view(-1)
                b_returns = returns.view(-1)
                b_values = values.view(-1)

                self.policy.update_policy(self.args, self.policy_optim, b_obs, b_actions,
                                          b_logprobs, b_advantages, b_returns, b_values)

                self.policy.eval()
                self.tasknet.train()

                prompt_adapt = [torch.zeros(self.k_per_type, self.hidden_dim).to(self.device) for _ in range(self.num_types)]

                with torch.no_grad():
                    for step in range(self.args.prompt_steps):
                        _ = self.model(self.feats, self.mask_feat, self.mask_adjs, self.adjs_norm,
                                       self.meow_args.num_cluster, prompt_adapt)
                        all_type_embs = self.model.get_all_state()

                        s = extract_states_from_embs(all_type_embs,self.all_cluster_labels,self.k_per_type)
                        action_adapt, _ = self.policy.get_action(s, deterministic=True)
                        action_3d_adapt = action_adapt.squeeze(0)
                        for t_i in range(self.num_types):
                            prompt_adapt[t_i] += action_3d_adapt[t_i]

                    _ = self.model(self.feats, self.mask_feat, self.mask_adjs, self.adjs_norm,
                                   self.meow_args.num_cluster, prompt_adapt)
                    final_embs_adapt = self.model.get_embeds()

                epoch_min_train_loss = float('inf')

                self.tasknet_optim.zero_grad()
                logits_adapt = self.tasknet(final_embs_adapt[self.idx_train])
                t_loss = self.criterion(logits_adapt, self.train_labels)
                t_loss.backward()
                self.tasknet_optim.step()

                if t_loss.item() < epoch_min_train_loss:
                    epoch_min_train_loss = t_loss.item()

                self.tasknet.eval()
                with torch.no_grad():
                    val_logits = self.tasknet(final_embs_adapt[self.idx_val])
                    val_labels = torch.argmax(self.label[self.idx_val], dim=-1)
                    val_loss = self.criterion(val_logits, val_labels).item()

                if val_loss < task_best_val_loss:
                    task_best_val_loss = val_loss
                    patience_cnt = 0
                    best_prompt_snapshot = [p.clone().detach() for p in prompt_adapt]
                    best_tasknet_state = copy.deepcopy(self.tasknet.state_dict())
                else:
                    patience_cnt += 1

                print(f"Epoch {epoch:03d}/{self.args.total_epochs} "
                      f"| Train Loss: {epoch_min_train_loss:.4f} "
                      f"| Val Loss: {val_loss:.4f} | Patience: {patience_cnt}/{self.args.patience}")

                if patience_cnt >= self.args.patience:
                    print(f"  >>> Early stopping triggered at epoch {epoch}.")
                    break

            self.tasknet.load_state_dict(best_tasknet_state)
            test_ma_f1, test_mi_f1 = self.evaluate(best_prompt_snapshot)
            print(f"\nTask {i} Test Ma-F1: {test_ma_f1 * 100:.2f}, Test Mi-F1: {test_mi_f1 * 100:.2f}")

            ma_f1s.append(test_ma_f1)
            mi_f1s.append(test_mi_f1)
            best_loss.append(task_best_val_loss)

        ma_f1s = np.array(ma_f1s) * 100
        mi_f1s = np.array(mi_f1s) * 100

        print(f"Micro-F1: {mi_f1s.mean():.2f}±{mi_f1s.std():.2f}")
        print(f"Macro-F1: {ma_f1s.mean():.2f}±{ma_f1s.std():.2f}")

    def evaluate(self, prompt):
        self.tasknet.eval()
        with torch.no_grad():
            _ = self.model(self.feats, self.mask_feat, self.mask_adjs, self.adjs_norm,
                           self.meow_args.num_cluster, prompt)
            best_embeds = self.model.get_embeds()
            test_logits = self.tasknet(best_embeds[self.idx_test])
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
            test_labels = torch.argmax(self.label[self.idx_test], dim=-1).cpu().numpy()
            test_ma_f1 = f1_score(test_labels, test_preds, average='macro')
            test_mi_f1 = f1_score(test_labels, test_preds, average='micro')
            print(f"Test Ma-F1: {test_ma_f1 * 100:.2f}, Test Mi-F1: {test_mi_f1 * 100:.2f}")

        return test_ma_f1.item(), test_mi_f1.item()


import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def extract_states_from_embs(embs_list, all_cluster_labels, k_per_type):
    state_components = []

    for i, emb in enumerate(embs_list):
        labels = all_cluster_labels[i]

        centroids = torch.zeros(k_per_type, emb.size(1), device=emb.device)
        for k in range(k_per_type):
            mask = (labels == k)
            if mask.any():
                centroids[k] = emb[mask].mean(dim=0)

        state_components.append(centroids)

    summary_state = torch.stack(state_components, dim=0)

    return summary_state


class Agent(nn.Module):
    def __init__(self, hidden_dim, embed_dim, k_per_type, num_types, max_z=0.1):
        super().__init__()
        self.max_z = max_z
        self.num_types = num_types
        self.k_per_type = k_per_type

        self.obs_dim = num_types * k_per_type * embed_dim
        self.action_dim = num_types * k_per_type * hidden_dim
        self.hidden_dim = hidden_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(embed_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, hidden_dim), std=0.01)
        )

        init_log_std = np.log(max_z) if max_z > 0 else 0.0
        self.register_buffer("actor_logstd", torch.ones(num_types, k_per_type, hidden_dim) * init_log_std)

    def get_value(self, x):
        flat_x = x.reshape(-1, self.obs_dim)
        state_value = self.critic(flat_x).squeeze(-1)
        return state_value

    def get_action(self, x, action=None,  deterministic=False):
        raw_mean = self.actor(x)
        action_mean = torch.tanh(raw_mean) * self.max_z

        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = action_mean if deterministic else probs.sample()

        action = torch.clamp(action, -self.max_z, self.max_z)
        log_prob_matrix = probs.log_prob(action)
        logprob = log_prob_matrix.sum(dim=[-3, -2, -1])

        return action, logprob

    def update_policy(self, args, optimizer, b_obs, b_actions, b_logprobs,
                      b_advantages, b_returns, b_values):

        b_inds = np.arange(args.batch_size)

        for update_epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _,  new_logprob = self.get_action(b_obs[mb_inds],  b_actions[mb_inds])
                new_value = self.get_value(b_obs[mb_inds])
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advs = b_advantages[mb_inds]
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                pg_loss1 = -mb_advs * ratio
                pg_loss2 = -mb_advs * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_value - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                loss = pg_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    cf = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

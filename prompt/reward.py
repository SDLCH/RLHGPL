import numpy as np
import torch


class RewardNormalizer:
    def __init__(self, device):
        self.running_mean = torch.tensor(0.0).to(device)
        self.running_var = torch.tensor(1.0).to(device)
        self.alpha = 0.99

    def __call__(self, x):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)

        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * batch_mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * batch_var

        return x / (torch.sqrt(self.running_var) + 1e-8)


def compute_gae_and_return(rewards, values, dones, gamma, gae_lambda):
    device = values.device
    num_episodes, prompt_steps = values.shape
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = torch.zeros(num_episodes).to(device)

    with torch.no_grad():
        for t in reversed(range(prompt_steps)):
            if t == prompt_steps - 1:

                nextvalues = torch.zeros(num_episodes).to(device)
            else:
                nextvalues = values[:, t + 1]
            delta = rewards[:, t] + gamma * nextvalues * (1.0 - dones[:, t]) - values[:, t]
            advantages[:, t] = lastgaelam = delta + gamma * gae_lambda * (1.0 - dones[:, t]) * lastgaelam

        returns = advantages + values

    return advantages, returns


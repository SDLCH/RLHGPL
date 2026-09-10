import argparse


def get_args():
    parser = argparse.ArgumentParser(description="RL-based Heterogeneous Graph Prompt Tuning")

    parser.add_argument("--dataset", type=str, default="acm", help="acm, dblp, aminer, freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--total_epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    parser.add_argument("--num_steps", type=int, default=16, help="Steps per rollout")
    parser.add_argument("--prompt_steps", type=int, default=16, help="Steps per rollout episode")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Minibatches per update")
    parser.add_argument("--update_epochs", type=int, default=1, help="PPO update epochs per iteration")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=1e-4, help="Critic learning rate")
    parser.add_argument("--actor_wd", type=float, default=0, help="Actor weight decay")
    parser.add_argument("--critic_wd", type=float, default=0, help="Critic weight decay")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping norm")
    parser.add_argument("--target_kl", type=float, default=0.01, help="KL divergence early stopping threshold")

    parser.add_argument("--k_per_type", type=int, default=5, help="Number of clusters")
    parser.add_argument("--max_z", type=float, default=0.1, help="Max prompt delta magnitude")
    parser.add_argument("--shot_num", type=int, default=3, help="Few-shot samples per class")
    parser.add_argument("--task_num", type=int, default=5, help="Number of evaluation tasks")

    parser.add_argument("--tasknet_lr", type=float, default=0.01, help="Tasknet learning rate")
    parser.add_argument("--tasknet_wd", type=float, default=0, help="Tasknet weight decay")

    args, _ = parser.parse_known_args()
    for key, value in datasets_args[args.dataset].items():
        setattr(args, key, value)

    return args


datasets_args = {
    "dblp": {
        "type_num": [4057, 14328, 7723, 20],
        "nei_num": 1,
        "n_labels": 4,
    },
    "aminer": {
        "type_num": [6564, 13329, 35890],
        "nei_num": 2,
        "n_labels": 4,
    },
    "freebase": {
        "type_num": [3492, 2502, 33401, 4459],
        "nei_num": 3,
        "n_labels": 3,
    },
    "acm": {
        "type_num": [4019, 7167, 60],
        "nei_num": 2,
        "n_labels": 3,
    },
}

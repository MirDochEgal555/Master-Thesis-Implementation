from dataclasses import dataclass

@dataclass
class TrainConfig:
    device: str = "cpu"
    T: int = 256
    updates: int = 50
    window_size: int = 32
    gamma: float = 0.99
    lam: float = 0.1

    # speed/dev toggles
    episodes_per_batch: int = 1
    use_dyn: bool = False
    use_kf: bool = True
    print_every: int = 10

    # reward shaping
    turn_coef: float = 0.0
    turn_target: float = 0.2
    cost_coef: float = 0.0
    kappa_unc: float = 0.0

    # optimization
    lr_actor: float = 1e-3
    actor_weight: float = 1.0

    lr_critic: float = 1e-3
    critic_weight = 1.0
    use_critic: bool = True
    use_baseline: bool = True


    lr_kf: float = 1e-3
    max_grad_norm_act: float = 1e9
    max_grad_norm_crit: float = 1e6
    kf_weight: float = 1.0

    normalize_adv: bool = False

    dyn_enabled: bool = True
    lr_dyn: float = 1e-4
    dyn_hidden: int = 128
    dyn_weight: float = 1.0

    # optional later
    dyn_use_sim: bool = True
    dyn_sim_M: int = 5          # number of simulated episodes per real rollout window
    dyn_sim_pl_weight: float = 0.01  # how much to weight sim loss vs real loss
    dyn_sim_vl_weight: float = 1.0
    dyn_sim_deterministic: bool = True  # sample next_state or use mean


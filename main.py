import os, sys, time
import importlib
from collections import namedtuple

import env
from models import ACModel, Discriminator

import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str,
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--note", nargs="+", type=str, default=None,
    help="Note file.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--device", type=str, default="0",
    help="ID of the target GPU device for model running or CPU.")

parser.add_argument("--silent", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)

settings = parser.parse_args()

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
    )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None and torch.cuda.is_available and "cuda" in str(device):
        torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
    
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

TRAINING_PARAMS = dict(
    horizon = 8,
    num_envs = 512,
    batch_size = 256,
    opt_epochs = 5,
    actor_lr = 5e-6,
    critic_lr = 1e-4,
    gamma = 0.95,
    lambda_ = 0.95,
    disc_lr = 1e-5,
    max_epochs = 10000,
    save_interval = None,
    log_interval = 50,
    terminate_reward = -1,
    control_mode="position"
)

def test(env, model):
    torch.autograd.set_grad_enabled(False)
    model.eval()
    env.eval()
    env.reset()
    while not env.request_quit:
        obs, info = env.reset_done()
        seq_len = info["ob_seq_lens"]
        pi1 = model1.act(obs, seq_len-1)
        pi2 = model2.act(obs, seq_len-1)

        actions = (pi1.mean, pi2.mean)
        env.step(actions)


def train(env, model, ckpt_dir, training_params, ckpt=None):
    if ckpt_dir is not None:
        logger = SummaryWriter(ckpt_dir)
    else:
        logger = None
    
    model1 = model[0]
    model2 = model[1]

    BATCH_SIZE = training_params.batch_size
    HORIZON = training_params.horizon
    GAMMA = training_params.gamma
    GAMMA_LAMBDA = training_params.gamma * training_params.lambda_
    OPT_EPOCHS = training_params.opt_epochs
    LOG_INTERVAL = training_params.log_interval
    OB_HORIZON = env.ob_horizon
    NOT_SILENT = not settings.silent
    N_ENVS = env.n_envs
    LOG = logger is not None
    SAVE_CKPT = ckpt_dir is not None


    optimizer = torch.optim.Adam([
        {"params": list(model1.actor.parameters())+list(model2.actor.parameters()), "lr": training_params.actor_lr},
        {"params": list(model1.critic.parameters())+list(model2.critic.parameters()), "lr": training_params.critic_lr}
    ])
    ac1_parameters = list(model1.actor.parameters()) + list(model1.critic.parameters())
    ac2_parameters = list(model2.actor.parameters()) + list(model2.critic.parameters())
    disc_optimizer = {name: torch.optim.Adam(disc.parameters(), training_params.disc_lr) for name, disc in model.discriminators.items()}

    buffer = dict(
        s=[], a1=[], a2=[], v1=[], v2=[], lp1=[], lp2=[], v1_=[], v2_=[], not_done=[], terminate=[],
        ob_seq_len=[]
    )
    multi_critics = env.reward_weights is not None and env.reward_weights.size(-1) > 1
    reward_weights = env.reward_weights[0]*2
    reward_weights1 = reward_weights[[0,1,4]]
    reward_weights2 = reward_weights[[2,3,5]]
    if multi_critics:
        rewards = torch.empty((env.n_envs*HORIZON, reward_weights.size(-1)), dtype=torch.float32, device=env.device)
    has_goal_reward = env.rew_dim > 0
    if has_goal_reward:
        buffer["r"] = []

    buffer_disc = {
        name: dict(fake=[], real=[]) for name in env.discriminators.keys()
    }
    real_losses, fake_losses = {n:[] for n in buffer_disc.keys()}, {n:[] for n in buffer_disc.keys()}
    

    epoch = 0
    if ckpt is not None and os.path.exists(settings.ckpt):
        if os.path.isdir(settings.ckpt):
            ckpt = os.path.join(settings.ckpt, "ckpt")
        else:
            ckpt = settings.ckpt
            settings.ckpt = os.path.dirname(ckpt)
        if os.path.exists(ckpt):
            print("Load model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])
            for n, opt in disc_optimizer.items():
                opt.load_state_dict(state_dict["disc_optimizer"][n])
            if "rng_state" in state_dict:
                set_rng_state(torch.load(ckpt, map_location="cpu", weights_only=False)["rng_state"], device)
            epoch = state_dict["epoch"]
            if "average_reward" in state_dict:
                env.average_reward = state_dict["average_reward"]
            # if "tracking_motion_samples" in state_dict:
            #     env.tracking_motion_samples = state_dict["tracking_motion_samples"]
            # if "track_note_samples" in state_dict:
            #     env.track_note_samples = state_dict["track_note_samples"]
    
    performance = {k:[] for k in env.info["log"].keys()}
    
    model.eval()
    env.train()
    env.reset()

    tic = time.time()
    while not env.request_quit:
        with torch.no_grad():
            obs, info = env.reset_done()
            seq_len = info["ob_seq_lens"]

            pi1, values1 = model1.act(obs, seq_len-1, with_value=True)
            pi2, values2 = model2.act(obs, seq_len-1, with_value=True)
            actions1 = pi1.sample()
            actions2 = pi2.sample()
            log_probs1 = pi1.log_prob(actions1).sum(-1, keepdim=True)
            log_probs2 = pi2.log_prob(actions2).sum(-1, keepdim=True)

            obs_, rews, dones, info = env.step((actions1, actions2))

            terminate = info["terminate"]

            masked = False
            mask = torch.any(torch.isnan(obs_), -1)
            if torch.any(mask):
                assert not torch.all(mask).item()
                print("NaN simulation results at step {}".format(env.simulation_step))
                env.done[mask] = True
                terminate[mask] = True
                obs[mask] = 0
                masked = True

            not_done = (~dones).unsqueeze_(-1)
            
            if env.discriminators:
                fakes = info["disc_obs"]
                reals = info["disc_obs_expert"]
                if masked:
                    for n, d in fakes.items():
                        d[mask, -1] = 0

            values1_ = model1.evaluate(obs_, seq_len)
            values2_ = model2.evaluate(obs_, seq_len)

            info_log = info["log"]
            for k, v in performance.items():
                v.append(info_log[k].mean().item())
                  
        buffer["s"].append(obs)
        buffer["a1"].append(actions1)
        buffer["lp1"].append(log_probs1)
        buffer["a2"].append(actions2)
        buffer["lp2"].append(log_probs2)
        buffer["v1"].append(values1)
        buffer["v1_"].append(values1_)
        buffer["v2"].append(values2)
        buffer["v2_"].append(values2_)
        buffer["not_done"].append(not_done)
        buffer["terminate"].append(terminate)
        buffer["ob_seq_len"].append(seq_len)
        if has_goal_reward:
            buffer["r"].append(rews)
        if env.discriminators:
            for name, fake in fakes.items():
                buffer_disc[name]["fake"].append(fake)
                buffer_disc[name]["real"].append(reals[name])

        if len(buffer["s"]) == HORIZON:
            disc_data = []
            ob_seq_lens = torch.cat(buffer["ob_seq_len"])
            ob_seq_end_frames = ob_seq_lens - 1
            if env.discriminators:
                with torch.no_grad():
                    for name, data in buffer_disc.items():
                        disc = model.discriminators[name]
                        fake = torch.cat(data["fake"])
                        real_ = torch.cat(data["real"])
                        end_frame = ob_seq_lens # N

                        length = torch.arange(fake.size(1), 
                            dtype=end_frame.dtype, device=end_frame.device
                        ).unsqueeze_(0)         # 1 x L
                        mask = length <= end_frame.unsqueeze(1)     # N x L
                        mask_ = length >= fake.size(1)-1 - end_frame.unsqueeze(1)

                        real = torch.zeros_like(real_)
                        real[mask] = real_[mask_]
                        disc.ob_normalizer.update(fake[mask])
                        disc.ob_normalizer.update(real[mask])
                        ob = disc.ob_normalizer(fake)
                        ref = disc.ob_normalizer(real)
                        disc_data.append((name, disc, ref, ob, end_frame))

                model.train()
                n_samples = 0
                for name, disc, ref, ob, seq_end_frame_ in disc_data:
                    real_loss = real_losses[name]
                    fake_loss = fake_losses[name]
                    opt = disc_optimizer[name]
                    if len(ob) != n_samples:
                        n_samples = len(ob)
                        idx = torch.randperm(n_samples)
                    for batch in range(n_samples//BATCH_SIZE):
                        sample = idx[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]

                        score_r = disc(r, seq_end_frame, normalize=False)
                        score_f = disc(f, seq_end_frame, normalize=False)
                    
                        loss_r = torch.nn.functional.relu(1-score_r).mean()
                        loss_f = torch.nn.functional.relu(1+score_f).mean()

                        with torch.no_grad():
                            alpha = torch.rand(r.size(0), dtype=r.dtype, device=r.device)
                            alpha = alpha.view(-1, *([1]*(r.ndim-1)))
                            interp = alpha*r+(1-alpha)*f
                        interp.requires_grad = True
                        with torch.backends.cudnn.flags(enabled=False):
                            score_interp = disc(interp, seq_end_frame, normalize=False)
                        grad = torch.autograd.grad(
                            score_interp, interp, torch.ones_like(score_interp),
                            retain_graph=True, create_graph=True, only_inputs=True
                        )[0]
                        gp = grad.reshape(grad.size(0), -1).norm(2, dim=1).sub(1).square().mean()
                        l = loss_f + loss_r + 10*gp
                        l.backward()
                        opt.step()
                        opt.zero_grad()

                        real_loss.append(score_r.mean().item())
                        fake_loss.append(score_f.mean().item())


            model.eval()
            with torch.no_grad():
                terminate = torch.cat(buffer["terminate"])
                if multi_critics:
                    # reward_weights = torch.cat(buffer["reward_weights"])
                    # rewards = torch.empty_like(reward_weights)
                    pass
                else:
                    # reward_weights = None
                    rewards = None
                for name, disc, _, ob, seq_end_frame in disc_data:
                    r = (disc(ob, seq_end_frame, normalize=False).clamp_(-1, 1)
                            .mean(-1, keepdim=True))
                    if rewards is None:
                        rewards = r
                    else:
                        rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                if has_goal_reward:
                    rewards_task = torch.cat(buffer["r"])
                    if rewards is None:
                        rewards = rewards_task
                    else:
                        rewards[:, -rewards_task.size(-1):] = rewards_task
                else:
                    rewards_task = None
                rewards[terminate] = training_params.terminate_reward
                rewards_ = rewards.view(HORIZON, -1, rewards.size(-1))
                not_done = buffer["not_done"]

                values1 = torch.cat(buffer["v1"])
                values1_ = torch.cat(buffer["v1_"])
                if model1.value_normalizer is not None:
                    values1 = model1.value_normalizer(values1, unnorm=True)
                    values1_ = model1.value_normalizer(values1_, unnorm=True)
                values1_[terminate] = 0
                values1 = values1.view(HORIZON, -1, values1.size(-1))
                values1_ = values1_.view(HORIZON, -1, values1_.size(-1))

                advantages1 = (rewards_[...,[0,1,4]] - values1).add_(values1_, alpha=GAMMA)
                for t in reversed(range(HORIZON-1)):
                    # advantages1[t].addcmul_(advantages1[t+1], not_done[t], value=GAMMA_LAMBDA)
                    advantages1[t].add_(advantages1[t+1]*not_done[t], alpha=GAMMA_LAMBDA)
                advantages1 = advantages1.view(-1, advantages1.size(-1))
                returns1 = advantages1 + values1.view(-1, advantages1.size(-1))
                sigma, mu = torch.std_mean(advantages1, dim=0, unbiased=True)
                advantages1 = (advantages1 - mu) / (sigma + 1e-8) # (HORIZON x N_ENVS) x N_DISC


                values2 = torch.cat(buffer["v2"])
                values2_ = torch.cat(buffer["v2_"])
                if model2.value_normalizer is not None:
                    values2 = model2.value_normalizer(values2, unnorm=True)
                    values2_ = model2.value_normalizer(values2_, unnorm=True)
                values2_[terminate] = 0
                values2 = values2.view(HORIZON, -1, values2.size(-1))
                values2_ = values2_.view(HORIZON, -1, values2_.size(-1))

                advantages2 = (rewards_[...,[2,3,5]] - values2).add_(values2_, alpha=GAMMA)
                for t in reversed(range(HORIZON-1)):
                    # advantages2[t].addcmul_(advantages2[t+1], not_done[t], value=GAMMA_LAMBDA)
                    advantages2[t].add_(advantages2[t+1]*not_done[t], alpha=GAMMA_LAMBDA)
                advantages2 = advantages2.view(-1, advantages2.size(-1))
                returns2 = advantages2 + values2.view(-1, advantages2.size(-1))
                sigma, mu = torch.std_mean(advantages2, dim=0, unbiased=True)
                advantages2 = (advantages2 - mu) / (sigma + 1e-8) # (HORIZON x N_ENVS) x N_DISC

                log_probs1 = torch.cat(buffer["lp1"])
                log_probs2 = torch.cat(buffer["lp2"])
                actions1 = torch.cat(buffer["a1"])
                actions2 = torch.cat(buffer["a2"])
                states = torch.cat(buffer["s"])

                if model1.use_rnn:
                    states_raw = model1.observe(states, norm=False)[0]
                    if OB_HORIZON > 1:
                        length = torch.arange(env.ob_horizon, 
                            dtype=ob_seq_lens.dtype, device=ob_seq_lens.device)
                        mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                        states_raw = states_raw[mask]
                    model1.ob_normalizer.update(states_raw, count_scale=N_ENVS) # use count_scale to prevent normalizer converge too fast when env is large
                else:
                    model1.ob_normalizer.update(states, count_scale=N_ENVS)
                if model2.use_rnn:
                    if not model1.use_rnn:
                        states_raw = model2.observe(states, norm=False)[0]
                        if OB_HORIZON > 1:
                            length = torch.arange(env.ob_horizon, 
                                dtype=ob_seq_lens.dtype, device=ob_seq_lens.device)
                            mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                            states_raw = states_raw[mask]
                    model2.ob_normalizer.update(states_raw, count_scale=N_ENVS) # use count_scale to prevent normalizer converge too fast when env is large
                else:
                    model2.ob_normalizer.update(states, count_scale=N_ENVS)

                if model1.value_normalizer is not None:
                    model1.value_normalizer.update(returns1)
                    returns1 = model1.value_normalizer(returns1)
                if model2.value_normalizer is not None:
                    model2.value_normalizer.update(returns2)
                    returns2 = model2.value_normalizer(returns2)
                if multi_critics:
                    advantages1 = advantages1.mul_(reward_weights1)
                    advantages2 = advantages2.mul_(reward_weights2)

            n_samples = states.size(0)
            policy_loss, value_loss = [], []
            model.train()
            for _ in range(OPT_EPOCHS):
                idx = torch.randperm(n_samples)
                for batch in range(n_samples // BATCH_SIZE):
                    sample = idx[BATCH_SIZE * batch: BATCH_SIZE *(batch+1)]
                    s = states[sample]
                    a1 = actions1[sample]
                    a2 = actions2[sample]
                    lp1 = log_probs1[sample]
                    lp2 = log_probs2[sample]
                    adv1 = advantages1[sample]
                    adv2 = advantages2[sample]
                    v_t1 = returns1[sample]
                    v_t2 = returns2[sample]
                    end_frame = ob_seq_end_frames[sample]

                    pi1_, v1_ = model1(s, end_frame)
                    pi2_, v2_ = model2(s, end_frame)

                    lp1_ = pi1_.log_prob(a1)
                    lp2_ = pi2_.log_prob(a2)

                    lp1_ = lp1_.sum(-1, keepdim=True)
                    lp2_ = lp2_.sum(-1, keepdim=True)

                    ratio = torch.exp(lp1_ - lp1)
                    clipped_ratio = torch.clamp(ratio, 1.0-0.2, 1.0+0.2)
                    pg_loss1 = -torch.min(adv1*ratio, adv1*clipped_ratio).sum(-1).mean()
                    vf_loss1 = (v1_ - v_t1).square().mean()
                    ratio = torch.exp(lp2_ - lp2)
                    clipped_ratio = torch.clamp(ratio, 1.0-0.2, 1.0+0.2)
                    pg_loss2 = -torch.min(adv2*ratio, adv2*clipped_ratio).sum(-1).mean()
                    vf_loss2 = (v2_ - v_t2).square().mean()

                    pg_loss = pg_loss1 + pg_loss2
                    vf_loss = vf_loss1 + vf_loss2
                    loss = pg_loss + 0.5*vf_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac1_parameters, 1.0)
                    torch.nn.utils.clip_grad_norm_(ac2_parameters, 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    policy_loss.append(pg_loss.item())
                    value_loss.append(vf_loss.item())
            model.eval()
            epoch += 1
            for v in buffer.values(): v.clear()
            for buf in buffer_disc.values():
                for v in buf.values(): v.clear()

            if epoch % LOG_INTERVAL == 1:
                lifetime = env.lifetime.to(torch.float32).mean().item()
                policy_loss, value_loss = np.mean(policy_loss), np.mean(value_loss)
                r = rewards.mean(0).cpu().tolist()
                terminate = terminate.sum().cpu().item()
                if NOT_SILENT:
                    print("Epoch: {}, Loss: {:.4f}/{:.4f}, Reward: {}, Lifetime: {:.4f}/{} -- {:.4f}s".format(
                        epoch, policy_loss, value_loss, "/".join(list(map("{:.4f}".format, r))), lifetime, terminate, time.time()-tic
                    ))
                if LOG:
                    logger.add_scalar("train/lifetime", lifetime, epoch)
                    logger.add_scalar("train/terminate", terminate, epoch)
                    logger.add_scalar("train/reward", np.mean(r), epoch)
                    logger.add_scalar("train/loss_policy", policy_loss, epoch)
                    logger.add_scalar("train/loss_value", value_loss, epoch)
                    for name, r_loss in real_losses.items():
                        if r_loss: logger.add_scalar("score_real/{}".format(name), sum(r_loss)/len(r_loss), epoch)
                    for name, f_loss in fake_losses.items():
                        if f_loss: logger.add_scalar("score_fake/{}".format(name), sum(f_loss)/len(f_loss), epoch)
                    if rewards_task is not None and rewards.size(-1) > 1:
                        rewards_task = rewards_task.mean(0).cpu().tolist()
                        for i in range(len(rewards_task)):
                            logger.add_scalar("train/task_reward_{}".format(i), rewards_task[i], epoch)
                    
                    for k, v in performance.items():
                        if v: logger.add_scalar("perform/{}".format(k), np.nanmean(v), epoch)
            
            for v in performance.values(): v.clear()
            for v in real_losses.values(): v.clear()
            for v in fake_losses.values(): v.clear()
            
            if SAVE_CKPT:
                state = None
                if epoch % 100 == 0:
                    state = dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        disc_optimizer={n: disc_opt.state_dict() for n, disc_opt in disc_optimizer.items()},
                        rng_state=get_rng_state(device),
                        epoch=epoch
                    )
                    if hasattr(env, "average_reward"):
                        state["average_reward"] = env.average_reward
                    if hasattr(env, "tracking_motion_samples"):
                        state["tracking_motion_samples"] = env.tracking_motion_samples
                    if hasattr(env, "track_note_samples"):
                        state["track_note_samples"] = env.track_note_samples

                    torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                if epoch % training_params.save_interval == 0:
                    if state is None:
                        state = dict(
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            disc_optimizer={n: disc_opt.state_dict() for n, disc_opt in disc_optimizer.items()},
                            rng_state=get_rng_state(device),
                            epoch=epoch
                        )
                        if hasattr(env, "average_reward"):
                            state["average_reward"] = env.average_reward
                        if hasattr(env, "tracking_motion_samples"):
                            state["tracking_motion_samples"] = env.tracking_motion_samples
                        if hasattr(env, "track_note_samples"):
                            state["track_note_samples"] = env.track_note_samples
                    torch.save(state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch)))
            if epoch > training_params.max_epochs: exit()
            tic = time.time()

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if settings.note is not None:
        config.env_params["note_file"] = settings.note

    if settings.device.lower() == "cpu":
        settings.device = "cpu"
    else:
        settings.device = int(settings.device)

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    training_params = namedtuple('training_params', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())
    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }

    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        env_cls = env.ICCGANHumanoid
        if settings.ckpt and not settings.resume:
            if os.path.isfile(settings.ckpt) or os.path.exists(os.path.join(settings.ckpt, "ckpt")):
                raise ValueError("Checkpoint folder {} exists. Add `--test` option to run test with an existing checkpoint file".format(settings.ckpt))
            import shutil, inspect
            os.makedirs(settings.ckpt, exist_ok=True)
            shutil.copy(settings.config, settings.ckpt)
            shutil.copy(__file__, settings.ckpt)
            shutil.copy(env.__file__, settings.ckpt)
            shutil.copy("utils.py", settings.ckpt)
            shutil.copy(inspect.getfile(ACModel), settings.ckpt)
            shutil.copy("ref_motion.py", settings.ckpt)
            with open(os.path.join(settings.ckpt, "command_{}.txt".format(time.time())), "w") as f:
                f.write(" ".join(sys.argv))
    if settings.test:
        num_envs = 1
    else:
        num_envs = training_params.num_envs
    print(training_params)
    print(env_cls, config.env_params)

    env = env_cls(num_envs,
        discriminators=discriminators,
        compute_device=settings.device, 
        **config.env_params
    )
    if settings.test:
        env.episode_length = 500000

    use_rnn = env.ob_horizon > 1
    value_dim = len(env.discriminators)+env.rew_dim
    state_dim, goal_dim = env.state_dim, env.goal_dim
    
    assert env.multi_objective_reward
    model1 = ACModel(use_rnn, state_dim, env.act_dim//2, goal_dim, value_dim//2, **config.model_params)
    model2 = ACModel(use_rnn, state_dim, env.act_dim//2, goal_dim, value_dim//2, **config.model_params)
    model = torch.nn.ModuleList([model1, model2])

    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    if settings.test:
        if settings.ckpt is not None and os.path.exists(settings.ckpt):
            if os.path.isdir(settings.ckpt):
                ckpt = os.path.join(settings.ckpt, "ckpt")
            else:
                ckpt = settings.ckpt
                settings.ckpt = os.path.dirname(ckpt)
            if os.path.exists(ckpt):
                print("Load model from {}".format(ckpt))
                print(model)
                state_dict = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state_dict["model"])
        env.render()
        test(env, model)
    else:
        train(env, model, settings.ckpt, training_params, settings.ckpt if settings.resume else None)

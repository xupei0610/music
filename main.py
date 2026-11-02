import os, sys, time
import importlib
from collections import namedtuple

import env
from models import ACModel, Discriminator
from fast_learner import GraphedStep, disc_step, ppo_step

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
parser.add_argument("--character", nargs="+", type=str, default=None,
    help="One or more MJCF paths used as the character model. Overrides "
         "the config's `character_model`. Several paths are merged into "
         "one model, for example a hand model and the include-files of "
         "scene objects.")
parser.add_argument("--motion", type=str, default=None,
    help="Motion file path. Overrides the config's `motion_file`.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--backend", choices=["warp", "mujoco"], default=None,
    help="Simulator: warp (mujoco_warp, default) or mujoco (native MuJoCo on the CPU, "
         "for tests and the viewer). Overrides the config's `backend`.")
parser.add_argument("--device", type=str, default="0",
    help="ID of the target GPU device for model running or CPU.")


parser.add_argument("--silent", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--normalize-value", action="store_true", default=False,
    help="Normalize the critic's value targets with PopArt (sets model_params "
         "normalize_value=True). A checkpoint trained with it has value_normalizer "
         "entries, so --resume and --test with that checkpoint also need this option.")

parser.add_argument("--cuda-graph", action="store_true", default=False,
    help="Replay each PPO and discriminator minibatch step as one CUDA graph "
         "(removes the CPU kernel-launch overhead). Needs a CUDA device.")

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
    control_mode="position",
    # Weight of the penalty on action stds at the max_sigma cap. For an action
    # dimension whose raw log_sigma is at or above the cap, the policy gradient
    # of its log-probability is stopped and reg * ||sigma|| is added to the
    # loss. 0 disables it. Needs model_params max_sigma.
    reg = 0,
)

def policy_models(env, model):
    """The per-hand policies: [model] for one hand, [left, right] for two."""
    if getattr(env, "two_hands", False):
        return [model[0], model[1]]
    return [model]


def value_columns(env):
    """Reward/value columns each policy is trained on, or [None] (all columns).

    Reward layout (env.reward_weights): one column per discriminator, ordered by
    DiscriminatorConfig.id, followed by env.rew_dim task rewards. With two
    hands the first half of the discriminators and task reward 0 belong to the
    left policy, the second half and task reward 1 to the right one.
    """
    if not getattr(env, "two_hands", False):
        return [None]
    n_disc = len(env.discriminators)
    assert n_disc % 2 == 0 and env.rew_dim == 2, \
        "two-hand training needs an even number of discriminators and 2 task rewards, got {} and {}".format(n_disc, env.rew_dim)
    for disc in env.discriminators.values():
        prefix = "L:" if disc.id < n_disc // 2 else "R:"
        names = [env.mj_model.body(lid + 1).name for lid in (disc.key_links or [])]
        assert all(n.startswith(prefix) for n in names), \
            "discriminator {} (id {}) must observe only {} links: {}".format(disc.name, disc.id, prefix, names)
    return [list(range(n_disc // 2)) + [n_disc], list(range(n_disc // 2, n_disc)) + [n_disc + 1]]


def test(env, model):
    torch.autograd.set_grad_enabled(False)
    model.eval()
    env.eval()
    env.reset()
    models = policy_models(env, model)
    while not env.request_quit:
        obs, info = env.reset_done()
        seq_len = info["ob_seq_lens"]
        actions = tuple(m.act(obs, seq_len-1).mean for m in models)
        env.step(actions if len(actions) > 1 else actions[0])


def train(env, model, ckpt_dir, training_params, ckpt=None):
    if ckpt_dir is not None:
        logger = SummaryWriter(ckpt_dir)
    else:
        logger = None

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

    REG = training_params.reg

    models = policy_models(env, model)
    cols = value_columns(env)
    N_POLICIES = len(models)

    CUDA_GRAPH = settings.cuda_graph
    if CUDA_GRAPH:
        # Normal's arg validation syncs the GPU, which breaks graph capture
        torch.distributions.Distribution.set_default_validate_args(False)
    optimizer = torch.optim.Adam([
        {"params": [p for m in models for p in m.actor.parameters()], "lr": training_params.actor_lr},
        {"params": [p for m in models for p in m.critic.parameters()], "lr": training_params.critic_lr}
    ], capturable=CUDA_GRAPH)
    ac_parameters = [list(m.actor.parameters()) + list(m.critic.parameters()) for m in models]

    disc_optimizer = {name: torch.optim.Adam(disc.parameters(), training_params.disc_lr, capturable=CUDA_GRAPH) for name, disc in model.discriminators.items()}
    ppo_graph, disc_graphs = None, {}
    if CUDA_GRAPH:
        # graphs bake in the address of all_inst; size it for the largest batch so it is never reallocated
        for mod in model.modules():
            if hasattr(mod, "all_inst"):
                mod.all_inst = torch.arange(N_ENVS*HORIZON, dtype=env.lifetime.dtype, device=env.device)

    buffer = dict(s=[], not_done=[], terminate=[], ob_seq_len=[])
    for k in ("a", "v", "lp", "v_"):
        buffer[k] = [[] for _ in models]
    multi_critics = env.reward_weights is not None and env.reward_weights.size(-1) > 1
    if multi_critics:
        # each policy's weights sum to the same total as a single policy's
        reward_weights = env.reward_weights[0] * N_POLICIES
        # every column is written each horizon: discriminator columns by
        # DiscriminatorConfig.id, task rewards in the last env.rew_dim columns
        rewards = torch.empty((env.n_envs*HORIZON, reward_weights.size(-1)), dtype=torch.float32, device=env.device)
        assert len(env.discriminators) + env.rew_dim == rewards.size(-1)
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

    performance = {k:[] for k in env.info["log"].keys()}

    model.eval()
    env.train()
    env.reset()

    tic = time.time()
    while not env.request_quit:
        with torch.no_grad():
            obs, info = env.reset_done()
            seq_len = info["ob_seq_lens"]

            step_actions, step_values, step_log_probs = [], [], []
            for m in models:
                pi, values = m.act(obs, seq_len-1, with_value=True)
                a = pi.sample()
                step_actions.append(a)
                step_values.append(values)
                step_log_probs.append(pi.log_prob(a).sum(-1, keepdim=True))
            actions = tuple(step_actions) if N_POLICIES > 1 else step_actions[0]

            obs_, rews, dones, info = env.step(actions)

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

            step_values_ = [m.evaluate(obs_, seq_len) for m in models]

            info_log = info["log"]
            for k, v in performance.items():
                v.append(info_log[k].mean().item())

        buffer["s"].append(obs)
        for i in range(N_POLICIES):
            buffer["a"][i].append(step_actions[i])
            buffer["lp"][i].append(step_log_probs[i])
            buffer["v"][i].append(step_values[i])
            buffer["v_"][i].append(step_values_[i])
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
                    if CUDA_GRAPH and name not in disc_graphs:
                        disc_graphs[name] = GraphedStep(
                            lambda r, f, e, disc=disc, opt=opt: disc_step(disc, opt, r, f, e),
                            [disc], [opt])
                    scores = []
                    for batch in range(n_samples//BATCH_SIZE):
                        sample = idx[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]

                        if CUDA_GRAPH:
                            scores.append(torch.stack(disc_graphs[name](r, f, seq_end_frame)))
                            continue

                        score_r, score_f = disc_step(disc, opt, r, f, seq_end_frame)
                        opt.zero_grad()

                        real_loss.append(score_r.item())
                        fake_loss.append(score_f.item())
                    if scores:
                        scores = torch.stack(scores).tolist()
                        real_loss.extend(x[0] for x in scores)
                        fake_loss.extend(x[1] for x in scores)

            model.eval()
            with torch.no_grad():
                terminate = torch.cat(buffer["terminate"])
                if not multi_critics:
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
                states = torch.cat(buffer["s"])

                batches = []
                for i, m in enumerate(models):
                    r_ = rewards_ if cols[i] is None else rewards_[..., cols[i]]
                    values = torch.cat(buffer["v"][i])
                    values_ = torch.cat(buffer["v_"][i])
                    if m.value_normalizer is not None:
                        values = m.value_normalizer(values, unnorm=True)
                        values_ = m.value_normalizer(values_, unnorm=True)
                    values_[terminate] = 0
                    values = values.view(HORIZON, -1, values.size(-1))
                    values_ = values_.view(HORIZON, -1, values_.size(-1))

                    advantages = (r_ - values).add_(values_, alpha=GAMMA)
                    for t in reversed(range(HORIZON-1)):
                        advantages[t].add_(advantages[t+1]*not_done[t], alpha=GAMMA_LAMBDA)
                    advantages = advantages.view(-1, advantages.size(-1))
                    returns = advantages + values.view(-1, advantages.size(-1))
                    sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
                    advantages = (advantages - mu) / (sigma + 1e-8)

                    if m.use_rnn:
                        states_raw = m.observe(states, norm=False)[0]
                        if OB_HORIZON > 1:
                            length = torch.arange(env.ob_horizon,
                                dtype=ob_seq_lens.dtype, device=ob_seq_lens.device)
                            mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                            states_raw = states_raw[mask]
                        m.ob_normalizer.update(states_raw, count_scale=N_ENVS) # count_scale keeps the normalizer from converging too fast with many envs
                    else:
                        m.ob_normalizer.update(states, count_scale=N_ENVS)
                    if m.value_normalizer is not None:
                        m.value_normalizer.update(returns)
                        returns = m.value_normalizer(returns)
                    if multi_critics:
                        advantages = advantages.mul_(reward_weights if cols[i] is None else reward_weights[cols[i]])
                    batches.append((torch.cat(buffer["a"][i]), torch.cat(buffer["lp"][i]), advantages, returns))

            n_samples = states.size(0)
            policy_loss, value_loss = [], []
            model.train()
            for _ in range(OPT_EPOCHS):
                idx = torch.randperm(n_samples)
                for batch in range(n_samples // BATCH_SIZE):
                    sample = idx[BATCH_SIZE * batch: BATCH_SIZE *(batch+1)]
                    s = states[sample]
                    end_frame = ob_seq_end_frames[sample]

                    minibatch = [x[sample] for b in batches for x in b]
                    if CUDA_GRAPH:
                        if ppo_graph is None:
                            ppo_graph = GraphedStep(
                                lambda *x: ppo_step(models, optimizer, ac_parameters, REG, *x),
                                models, [optimizer])
                        pg_loss, vf_loss = ppo_graph(s, end_frame, *minibatch)
                        policy_loss.append(torch.stack((pg_loss, vf_loss)))
                        continue

                    pg_loss, vf_loss = ppo_step(models, optimizer, ac_parameters, REG, s, end_frame, *minibatch)
                    optimizer.zero_grad()

                    policy_loss.append(pg_loss.item())
                    value_loss.append(vf_loss.item())
            if CUDA_GRAPH and policy_loss:
                losses = torch.stack(policy_loss).tolist()
                policy_loss, value_loss = [x[0] for x in losses], [x[1] for x in losses]
            model.eval()
            epoch += 1
            buffer["s"].clear(); buffer["not_done"].clear(); buffer["terminate"].clear(); buffer["ob_seq_len"].clear()
            if has_goal_reward: buffer["r"].clear()
            for k in ("a", "v", "lp", "v_"):
                for v in buffer[k]: v.clear()
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
                if epoch % 100 == 0 or epoch % training_params.save_interval == 0:
                    state = dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        disc_optimizer={n: disc_opt.state_dict() for n, disc_opt in disc_optimizer.items()},
                        rng_state=get_rng_state(device),
                        epoch=epoch
                    )
                    for k in ("average_reward", "tracking_motion_samples", "track_note_samples"):
                        if hasattr(env, k):
                            state[k] = getattr(env, k)
                if epoch % 100 == 0:
                    torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                if epoch % training_params.save_interval == 0:
                    torch.save(state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch)))
            if epoch > training_params.max_epochs: exit()
            tic = time.time()

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if settings.note is not None:
        config.env_params["note_file"] = settings.note
    if settings.character is not None:
        # nargs="+" yields a list. Single path -> string, multiple -> list
        # (Env._resolve_character_model handles both forms).
        cm = settings.character
        config.env_params["character_model"] = cm[0] if len(cm) == 1 else cm
    if settings.motion is not None:
        config.env_params["motion_file"] = settings.motion
    if settings.backend is not None:
        config.env_params["backend"] = settings.backend
    if settings.normalize_value:
        config.model_params = dict(getattr(config, "model_params", {}), normalize_value=True)

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
            if settings.normalize_value:
                # record the command-line override in the copied config
                with open(os.path.join(settings.ckpt, os.path.basename(settings.config)), "a") as f:
                    f.write("\n# set by main.py --normalize-value\nmodel_params[\"normalize_value\"] = True\n")
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

    if getattr(env, "two_hands", False):
        model1 = ACModel(use_rnn, state_dim, env.act_dim//2, goal_dim, value_dim//2, **config.model_params)
        model2 = ACModel(use_rnn, state_dim, env.act_dim//2, goal_dim, value_dim//2, **config.model_params)
        model = torch.nn.ModuleList([model1, model2])
    else:
        model = ACModel(use_rnn, state_dim, env.act_dim, goal_dim, value_dim, **config.model_params)

    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = env.device  # the native MuJoCo backend runs on the CPU
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
                # A pretrained checkpoint keeps only the actor and the observation
                # normalizer mean/var (see runs/PRETRAINED.md). Running the policy
                # needs nothing else, so allow the other entries to be missing.
                res = model.load_state_dict(state_dict["model"], strict=False)
                needed = [k for k in res.missing_keys
                    if not k.startswith("discriminators.") and (".actor." in "." + k or k.split(".")[-2:] in (["ob_normalizer", "mean"], ["ob_normalizer", "var"]))]
                if needed or res.unexpected_keys:
                    raise RuntimeError("Checkpoint {} does not match the policy. Missing: {} Unexpected: {}".format(
                        ckpt, needed[:5], res.unexpected_keys[:5]))
        env.render()
        test(env, model)
    else:
        train(env, model, settings.ckpt, training_params, settings.ckpt if settings.resume else None)

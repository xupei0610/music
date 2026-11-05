from typing import Callable, Optional, Union, List, Dict, Any, Sequence
import math
import os, pickle, sys, tempfile, time
import resource

# The discriminator sample workers send their batches as shared-memory torch
# tensors, one file descriptor per tensor (128 per batch and discriminator).
# The common default soft limit of 1024 open files is too low ("received 0
# items of ancdata"), so raise it to the hard limit, as `ulimit -n` would.
# Child processes inherit it.
_nofile_soft, _nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
_nofile_want = 65536 if _nofile_hard == resource.RLIM_INFINITY else min(_nofile_hard, 65536)
if _nofile_soft != resource.RLIM_INFINITY and _nofile_soft < _nofile_want:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_nofile_want, _nofile_hard))

import mujoco
import mujoco.viewer

import numpy as np
import torch

from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quat2expmap, quat2expmap2, xquat_to_xyzw
from sim import make_simulator, backend_device


def warp2torch(arr) -> torch.Tensor:
    """Zero-copy torch view of a warp array (for scripts that use env.mjx_data)."""
    import warp as wp
    return wp.to_torch(arr)


def episode_limit(episode_length):
    """The hard episode length limit, or None for no limit.
    None, a missing value or a non-positive number means no limit. A callable
    (simulation_step -> limit) is kept as it is."""
    if episode_length is None or callable(episode_length):
        return episode_length
    return int(episode_length) if episode_length > 0 else None


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """x[m].float().mean() without the host sync of boolean indexing; NaN if m is empty."""
    return torch.where(m, x, 0).float().sum() / m.sum()


class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, key_link_index_offset: Optional[int]=None,
        ob_horizon: Optional[int]=None,
        parent_link: Optional[str]=None,
        local_pos: Optional[bool]=None, global_pos: bool=False,
        replay_speed: Optional[str]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.key_link_index_offset = key_link_index_offset
        self.local_pos = local_pos
        self.global_pos = global_pos
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight


class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS = 0, -4.5, 2.0
    CAMERA_FOLLOWING = True

    def __init__(self,
        n_envs: int, fps: int=30, frameskip: int=8,
        episode_length: Optional[Union[Callable, int]] = None,  # None or <= 0: no hard limit
        control_mode: str = "none",
        compute_device: int = 0,
        character_model: Optional[Union[str, Sequence[str]]] = None,
        **kwargs
    ):
        self.viewer = None
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        assert control_mode in ["direct", "position", "muscle", "none"]
        self.control_mode = control_mode
        self.episode_length = episode_limit(episode_length)
        # simulator behind the task: "warp" (mujoco_warp, batched, GPU) or "mujoco" (native, CPU)
        self.backend = kwargs.get("backend", "warp")
        self.compute_device = compute_device
        self.device = backend_device(self.backend, compute_device)

        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model

        self.n_envs = n_envs
        # warp only: per-world contact / constraint buffer sizes (None: derived
        # from the MJCF) and CUDA graphs for the physics and kinematics
        self.nconmax = kwargs.get("nconmax")
        self.njmax = kwargs.get("njmax")
        self.cuda_graph = kwargs.get("cuda_graph", True)
        self.prepare_sim()
        self.setup_action_normalizer()

        self.refresh_tensors()
        self.train()
        self.viewer_pause = False
        self.viewer_advance = False

        self.simulation_step = 0
        self.lifetime = torch.zeros(self.n_envs, dtype=torch.int64, device=self.device)
        self.done = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime, log=dict())

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def render(self):
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self.viewer_key_callback)

    def viewer_key_callback(self, key):
        if key == 80: # P
            self.viewer_pause = not self.viewer_pause


    def update_viewer(self):
        self.sim.copy_to_mjdata(self.mj_data, 0)

    @property
    def mjx_model(self):
        """The mujoco_warp model (warp backend only)."""
        return self.sim.model

    @property
    def mjx_data(self):
        """The mujoco_warp data (warp backend only)."""
        return self.sim.data

    def refresh_tensors(self):
        self.sim.refresh_kinematics()
        # build a (n_envs, n_links, 13) link_tensor in xyzw quat order, excluding world body
        link_xpos = self.sim.state("xpos")
        link_xquat = xquat_to_xyzw(self.sim.state("xquat"))
        link_cvel = self.sim.state("cvel")  # spatial vector (rot:lin) shape (n_envs, n_body, 6)
        link_state = torch.cat([link_xpos, link_xquat, link_cvel], dim=-1)[:, 1:]
        self.link_tensor = link_state.contiguous()

    def _resolve_character_model(self):
        """Return a path to a single MJCF that mj_model.from_xml_path can
        load. If character_model is a string, return it as-is. If it's a
        list/tuple of paths, build a transient wrapper MJCF that
        <include>s each one in order, with paths made relative to the
        wrapper's directory so MuJoCo can resolve them. The wrapper is
        written next to the first (arm) MJCF and removed after load by
        prepare_sim. Use case: combine a character model with one or more
        include-files (for example, scene objects) without editing the
        character XML."""
        cm = self.character_model
        if isinstance(cm, (str, os.PathLike)):
            return str(cm), None
        parts = list(cm)
        if not parts:
            raise ValueError("character_model is an empty list")
        if len(parts) == 1:
            return str(parts[0]), None

        primary = os.path.abspath(str(parts[0]))
        primary_dir = os.path.dirname(primary)

        rel_paths = []
        for p in parts:
            ap = os.path.abspath(str(p))
            try:
                rel = os.path.relpath(ap, primary_dir)
            except ValueError:
                # Different drives on Windows -- fall back to absolute.
                rel = ap
            rel_paths.append(rel)

        body = "\n".join(f'  <include file="{r}"/>' for r in rel_paths)
        text = f'<mujoco model="merged_character_model">\n{body}\n</mujoco>\n'

        fd, tmp_path = tempfile.mkstemp(
            suffix=".xml",
            prefix="merged_",
            dir=primary_dir,
        )
        os.close(fd)
        with open(tmp_path, "w") as f:
            f.write(text)
        return tmp_path, tmp_path  # second value: the temp file to remove later

    def prepare_sim(self, opt={}):
        xml_path, tmp_to_remove = self._resolve_character_model()
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        finally:
            if tmp_to_remove is not None:
                try:
                    os.unlink(tmp_to_remove)
                except FileNotFoundError:
                    pass

        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_model.opt.iterations = 30
        self.mj_model.opt.tolerance = 1e-5
        self.mj_model.opt.ls_iterations = 20
        self.mj_model.opt.timestep = 1/(self.fps*self.frameskip)
        for k, v in opt.items():
            setattr(self.mj_model.opt, k, v)
        print(self.mj_model.opt)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        bodies = []
        for i in range(self.mj_model.nbody):
            b = self.mj_model.body(i)
            bodies.append((b.name, b.id))
        print(bodies, len(self.mj_data.xpos))
        actuators = []
        for i in range(self.mj_model.nu):
            a = self.mj_model.actuator(i)
            actuators.append((a.name, a.id))
        print(actuators, len(self.mj_data.ctrl))

        self._make_simulator()

        self.root_links = []
        root_link_qidx = []
        qpos_range = [[], []]
        for jnt in range(self.mj_model.njnt):
            jnt = self.mj_model.jnt(jnt)
            if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
                self.root_links.append(int(jnt.bodyid[0]-1))
                root_link_qidx.append(int(jnt.qposadr[0]))

                qpos_range[0].extend([-np.inf]*7)
                qpos_range[1].extend([np.inf]*7)
            else:
                qpos_range[0].append(jnt.range[0])
                qpos_range[1].append(jnt.range[1])
        if self.root_links:
            character_qidx = []
            if root_link_qidx[0] > 0:
                character_qidx.append((-1, root_link_qidx[0]))
            for i in range(len(root_link_qidx)-1):
                character_qidx.append((root_link_qidx[i], root_link_qidx[i+1]-root_link_qidx[i]-7))
            character_qidx.append((root_link_qidx[-1], len(self.mj_data.qpos)-root_link_qidx[-1]))
        else:
            character_qidx = [(-1, self.mj_model.nv)]

        # prevent extreme pose outside the control of muscles
        qpos_range0 = np.add(qpos_range[0], np.subtract(qpos_range[1],qpos_range[0])*0.1)
        qpos_range1 = np.subtract(qpos_range[1], np.subtract(qpos_range[1],qpos_range[0])*0.1)
        qpos_range0 = torch.tensor(np.nan_to_num(qpos_range0, nan=-np.inf), dtype=torch.float, device=self.device)
        qpos_range1 = torch.tensor(np.nan_to_num(qpos_range1, nan=np.inf), dtype=torch.float, device=self.device)
        self._qpos_range0 = qpos_range0
        self._qpos_range1 = qpos_range1
        self._character_qidx = character_qidx

    def _reset_pose(self, env_ids, ref_root, ref_root_ang_vel, ref_joint_pos, ref_joint_vel):
        qpos = self.sim.state("qpos")
        qvel = self.sim.state("qvel")
        jidx, ridx = 0, 0
        q, v = [], []
        for qidx0, ndofs in self._character_qidx:
            jidx_ = jidx+ndofs
            if qidx0 >= 0:
                # ref_root is (N, n_root, 13) in xyzw order; mujoco qpos free-joint expects (px,py,pz, qw,qx,qy,qz)
                root = ref_root[:, ridx]
                q.append(torch.cat([root[..., 0:3], root[..., [6]], root[..., 3:6]], -1))
                v.append(root[..., 7:10])
                v.append(ref_root_ang_vel[:, ridx])
                ridx += 1
            if jidx < ref_joint_pos.shape[1]:
                if jidx_ <= ref_joint_pos.shape[1]:
                    q.append(ref_joint_pos[:, jidx:jidx_])
                    v.append(ref_joint_vel[:, jidx:jidx_])
                else:
                    q.append(ref_joint_pos[:, jidx:])
                    v.append(ref_joint_vel[:, jidx:])
            jidx = jidx_

        v = torch.cat(v, -1)
        qdim = v.shape[1] + ridx
        if qpos.shape[1] > qdim:
            q.append(torch.zeros_like(qpos[env_ids, qdim:]))
            qvel[env_ids, :v.shape[1]] = v
            qvel[env_ids, v.shape[1]:] = 0
        else:
            qvel[env_ids] = v
        q = torch.cat(q, -1)
        q = torch.clamp(q, min=self._qpos_range0, max=self._qpos_range1)
        qpos[env_ids] = q

    def _make_simulator(self):
        self.sim = make_simulator(self.backend, self.mj_model, self.mj_data, self.n_envs, self.compute_device,
                                  nconmax=self.nconmax, njmax=self.njmax, cuda_graph=self.cuda_graph)

    def setup_action_normalizer(self):
        action_upper = []
        action_lower = []
        action_scale = []
        self.muscle_control = False
        for i in range(self.mj_model.nu):
            actuator = self.mj_model.actuator(i)
            if self.control_mode == "direct" or (self.control_mode == "none" and actuator.biastype[0] == 0):
                lower_limit = actuator.ctrlrange[0]
                upper_limit = actuator.ctrlrange[1]
                if actuator.ctrlrange[0] == actuator.ctrlrange[1]:
                    if actuator.ctrllimited.item():
                        print("[Warning] Find limited actuator {} with the same lower and upper bounds of control range.".format(actuator.name))
                    else:
                        lower_limit = -1
                        upper_limit = 1
                action_scale.append(1)
            elif self.control_mode == "position" or (self.control_mode == "none" and actuator.biastype[0] == 1):
                if actuator.trntype[0] == 0: # joint
                    joint = self.mj_model.joint(actuator.trnid[0])
                    lower_limit = joint.range[0]
                    upper_limit = joint.range[1]
                elif actuator.trntype[0] == 3: # tendon
                    lower_limit = actuator.lengthrange[0]
                    upper_limit = actuator.lengthrange[1]
                else:
                    raise ValueError("Unsupported actuator type {} for actuator {}.".format(actuator.trntype[0], actuator.name))
                action_scale.append(2)
            elif self.control_mode == "muscle" or (self.control_mode == "none" and actuator.biastype[0] == 2):
                lower_limit = -1
                upper_limit = 1
                self.muscle_control = True
                action_scale.append(1)
            else:
                assert self.control_mode in ["muscle", "position", "direct"] or (self.control_mode == "none" and actuator.biastype[0] in [0, 1, 2]), "Unsupported control mode"
            assert not np.isinf(lower_limit) and not np.isinf(upper_limit) and lower_limit < upper_limit, actuator
            action_upper.append(upper_limit)
            action_lower.append(lower_limit)
        action_upper = np.array(action_upper)
        action_lower = np.array(action_lower)
        self.action_offset = torch.tensor(0.5 * (action_upper + action_lower), dtype=torch.float, device=self.device)
        self.action_scale = torch.tensor(action_scale * (0.5 * (action_upper - action_lower)), dtype=torch.float, device=self.device)

        if not self.muscle_control:
            self.actuator_forcerange = torch.tensor(self.mj_model.actuator_forcerange[:-6, 1], dtype=torch.float, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions*self.action_scale + self.action_offset

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime, log=dict())
        self.request_quit = False
        self.obs = None

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == self.n_envs or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info

    def reset_envs(self, env_ids0):
        ref_link_tensor, ref_joint_tensor = self.init_state(env_ids0)

        self.lifetime.index_fill_(0, env_ids0, 0)
        env_ids = env_ids0

        if self.root_links:
            ref_root_tensor = ref_link_tensor[:, self.root_links]
            ref_root_ang_vel_tensor = rotatepoint(quatconj(ref_root_tensor[..., 3:7]), ref_root_tensor[..., 10:13])
            ref_root = ref_root_tensor
            ref_root_ang_vel = ref_root_ang_vel_tensor
        else:
            ref_root, ref_root_ang_vel = None, None
        ref_joint_pos = ref_joint_tensor[0]
        ref_joint_vel = ref_joint_tensor[1]

        self._reset_pose(env_ids, ref_root, ref_root_ang_vel, ref_joint_pos, ref_joint_vel)

        qacc_warmstart = self.sim.state("qacc_warmstart")
        # index_fill_ takes the fill value as a scalar argument; `x[ids] = 0`
        # copies a CPU tensor to the GPU, which waits for all queued GPU work.
        qacc_warmstart.index_fill_(0, env_ids, 0)
        if self.muscle_control:
            act = self.sim.state("act")
            if self.training and self.random_init:
                idx = torch.from_numpy(np.random.randint(0, self.n_envs, len(env_ids))).to(self.device)
                act[env_ids] = act[idx]
            else:
                act.index_fill_(0, env_ids, 0)
        self.refresh_tensors()

    def do_simulation(self, actions):
        # actions is (n_envs, nu)
        self.sim.state("ctrl").copy_(actions)
        self.sim.step(self.frameskip)
        self.simulation_step += 1

    def step(self, actions):
        if not self.viewer_pause or self.viewer_advance:
            actions = self.process_actions(actions)
            self.do_simulation(actions)
            self.refresh_tensors()
            self.lifetime += 1
        rewards = self.reward()
        terminate = self.termination_check()
        self.info["terminate"] = terminate
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check()
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate

        if self.viewer is not None:
            if not self.viewer.is_running(): exit()
            self.update_viewer()
            self.viewer.sync()
            # Play in real time: one control step per step_time of wall time.
            # A step that runs late does not shorten the next one.
            now = time.perf_counter()
            due = getattr(self, "_viewer_due", now)
            if due > now:
                time.sleep(due - now)
            self._viewer_due = max(due, now) + self.step_time

        self.obs = self.observe()
        return self.obs, rewards, self.done, self.info

    def init_state(self, env_ids):
        raise NotImplementedError()

    def observe(self, env_ids=None):
        raise NotImplementedError()

    def overtime_check(self):
        if self.episode_length is None:
            # no hard limit: subclasses add their own end conditions to this
            return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        if callable(self.episode_length):
            return self.lifetime >= self.episode_length(self.simulation_step).to(self.lifetime.device)
        return self.lifetime >= self.episode_length

    def termination_check(self):
        return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)

    def reward(self):
        return torch.ones((self.n_envs, 0), dtype=torch.float, device=self.device)


from ref_motion import ReferenceMotion


class ICCGANHumanoid(Env):

    CHARACTER_MODEL = os.path.join("assets", "humanoid.xml")
    CONTACTABLE_LINKS = ["right_foot", "left_foot", "right_shin", "left_shin"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4
    KEY_LINKS = None
    PARENT_LINK = None
    LOCAL_POS = None
    GLOBAL_POS = False
    INCLUDE_VEL = True


    def __init__(self, *args,
        motion_file: Optional[str]=None,
        discriminators: Optional[Dict[str, DiscriminatorConfig]]={},
    **kwargs):

        contactable_links = kwargs.get("contactable_links", self.CONTACTABLE_LINKS)
        goal_reward_weight = kwargs.get("goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = kwargs.get("enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = self.get_goal_tensor_dim()
        self.ob_horizon = kwargs.get("ob_horizon", self.OB_HORIZON)
        self.key_links = kwargs.get("key_links", self.KEY_LINKS)
        self.parent_link = kwargs.get("parent_link", self.PARENT_LINK)
        self.local_pos = kwargs.get("local_pos", self.LOCAL_POS)
        self.global_pos = kwargs.get("global_pos", self.GLOBAL_POS)
        self.include_vel = kwargs.get("include_vel", self.INCLUDE_VEL)

        super().__init__(*args, **kwargs)

        if contactable_links:
            contactable_link_idx = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link) for link in contactable_links]
            for lid, link in zip(contactable_link_idx, contactable_links):
                assert lid > 0, "Unrecognized contactable link {}".format(link)

            contactable_geoms = []
            for i in range(self.mj_model.ngeom):
                geom = self.mj_model.geom(i)
                body_id = geom.bodyid[0]
                if body_id in contactable_link_idx:
                    contactable_geoms.append(i)
            self._contactable_geoms = torch.tensor(contactable_geoms, dtype=torch.int32, device=self.device)
            self._ground_contact_test = self._ground_contact_test_sim
        else:
            self._contactable_geoms = None
            self._ground_contact_test = None


        if goal_reward_weight is not None:
            reward_weights = torch.empty((self.n_envs, self.rew_dim), dtype=torch.float, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[:, i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight))

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((self.n_envs, n_comp), dtype=torch.float, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            if config.key_links is None:
                key_links = None
            else:
                key_links = []
                for link in config.key_links:
                    lid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link) - 1
                    assert lid >= 0, "Unrecognized key link {}".format(link)
                    key_links.append(lid)
                key_links = sorted(key_links)
            if config.parent_link is None:
                parent_link = None
            else:
                parent_link = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, config.parent_link) - 1
                assert parent_link >= 0, "Unrecognized parent link {}".format(config.parent_link)
            assert key_links is None or all(lid >= 0 for lid in key_links)
            assert parent_link is None or parent_link >= 0
            config.parent_link = parent_link
            config.key_links = key_links

            assert config.motion_file is not None or motion_file is not None
            if config.motion_file is None:
                config.motion_file = motion_file
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            config.id = i
            config.name = id
            self.discriminators[id] = config
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.zeros((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.state_hist.dtype, device=self.device)
        if self.reward_weights is None:
            self.reward_weights = torch.ones((self.n_envs, 1), dtype=torch.float, device=self.device)
        elif self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights

        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)
        self.goal_dim = self.get_goal_dim()
        self.state_dim = (self.ob_dim-self.goal_dim)//self.ob_horizon
        if self.discriminators:
            self.info["disc_obs"] = self.observe_disc(self.state_hist)
            self.info["disc_obs_expert"] = self.info["disc_obs"]
            self.disc_dim = {
                name: ob.size(-1)
                for name, ob in self.info["disc_obs"].items()
            }
        else:
            self.disc_dim = {}

        if motion_file:
            self.ref_motion = self.build_motion_lib(motion_file)
        else:
            self.ref_motion = None
        self.sampling_workers = []
        self.real_samples = []

    def get_goal_dim(self):
        return self.GOAL_DIM

    def get_goal_tensor_dim(self):
        return self.GOAL_TENSOR_DIM

    def build_motion_lib(self, motion_file: Union[str, Sequence[str]]):
        # ReferenceMotion takes character_model only as opaque metadata; if
        # we were given a list of MJCFs we hand it the primary one (the arm).
        cm = self.character_model
        if not isinstance(cm, (str, os.PathLike)):
            cm = str(list(cm)[0])
        return ReferenceMotion(motion_file=motion_file, character_model=cm, device=self.device)

    def __del__(self):
        if hasattr(self, "sampling_workers"):
            for p in self.sampling_workers:
                p.terminate()
            for p in self.sampling_workers:
                p.join()
        super().__del__()

    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info

    def reset(self):
        if self.goal_tensor is not None:
            if torch.is_tensor(self.goal_tensor):
                self.goal_tensor.zero_()
            else:
                for t in self.goal_tensor: t.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)

    def reset_goal(self, env_ids):
        pass

    def step(self, actions):
        obs, rews, dones, info = super().step(actions)
        if self.discriminators and self.training:
            info["disc_obs"] = self.observe_disc(self.state_hist)
            info["disc_obs_expert"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        overtime = super().overtime_check()
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero((self.goal_timer <= 0).logical_and_(~overtime).logical_and_(~self.info["terminate"])).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return overtime

    def _ground_contact_test_sim(self):
        # Any contact between the ground (geom 0) and a geom outside the
        # contactable links within 1cm. The scatter keeps the test on the
        # device without a host sync.
        geom, dist, worldid, valid = self.sim.contacts()
        g1, g2 = geom[:, 0], geom[:, 1]
        contact1 = (g1 == 0).logical_and_(~torch.isin(g2, self._contactable_geoms))
        contact2 = (g2 == 0).logical_and_(~torch.isin(g1, self._contactable_geoms))
        hit = contact1.logical_or_(contact2).logical_and_(dist < 0.01).logical_and_(valid)
        terminate = torch.zeros(self.n_envs, dtype=torch.int32, device=self.device)
        terminate.scatter_reduce_(0, worldid.to(torch.int64).clip_(0, self.n_envs-1), hit.to(torch.int32), reduce="amax")
        return terminate.to(torch.bool)

    def termination_check(self):
        if self._ground_contact_test is None:
            return torch.zeros_like(self.done)
        terminate = self._ground_contact_test()
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times, with_joint_tensor=True)
        return ref_link_tensor, ref_joint_tensor

    def prepare_sim(self, opt={}):
        super().prepare_sim(opt)
        n_links = self.mj_model.nbody - 1
        self.state_hist = torch.empty((self.ob_horizon+1, self.n_envs, n_links*13),
            dtype=torch.float, device=self.device)

        if self.key_links is not None:
            key_links = []
            for link in self.key_links:
                lid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link)-1
                assert lid >= 0, "Unrecognized key link {}".format(link)
                key_links.append(lid)
            self.key_links = sorted(key_links)
        if self.parent_link is not None:
            parent_link = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.parent_link)-1
            assert parent_link >= 0, "Unrecognized parent link {}".format(self.parent_link)
            self.parent_link = parent_link

        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((self.n_envs, dim), dtype=torch.float, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((self.n_envs, self.goal_tensor_dim), dtype=torch.float, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((self.n_envs, ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None

    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1
        n_envs = self.n_envs
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[:-1] = self.state_hist[1:].clone()
            self.state_hist[-1] = self.link_tensor.reshape(n_envs, -1)
            env_ids = None
        else:
            self.state_hist[:-1, env_ids] = self.state_hist[1:, env_ids].clone()
            self.state_hist[-1, env_ids] = self.link_tensor[env_ids].reshape(len(env_ids), -1)
        return self._observe(env_ids)

    def _observe(self, env_ids):
        if env_ids is None:
            ob = observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link,
                include_vel=self.include_vel, local_pos=self.local_pos, global_pos=self.global_pos,
                key_link_index=self._key_link_index(self.key_links)
            )
        else:
            ob = observe_iccgan(
                self.state_hist[-self.ob_horizon:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link,
                include_vel=self.include_vel, local_pos=self.local_pos, global_pos=self.global_pos,
                key_link_index=self._key_link_index(self.key_links)
            )
        return ob.flatten(start_dim=1)

    def _key_link_index(self, key_links):
        """key_links as a device tensor, built once per list."""
        if key_links is None:
            return None
        cache = self.__dict__.setdefault("_key_link_index_cache", {})
        k = tuple(key_links)
        if k not in cache:
            cache[k] = torch.tensor(key_links, dtype=torch.int64, device=self.device)
        return cache[k]

    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        if torch.is_tensor(state):
            for id, disc in self.discriminators.items():
                res[id] = observe_iccgan(state[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link,
                    include_vel=False, local_pos=disc.local_pos, global_pos=disc.global_pos,
                    key_link_index=self._key_link_index(disc.key_links))
            return res
        else:
            assert False

    def fetch_real_samples(self):
        if not self.real_samples:
            if not self.sampling_workers:
                self.disc_ref_motion = {}
                import torch.multiprocessing as mp
                if mp.get_start_method(allow_none=True) != "spawn":
                    mp.set_start_method("spawn", force=True)
                # Workers and the queue manager exit on their own when this
                # process dies without running its cleanup (SIGTERM/SIGKILL,
                # os._exit); otherwise they would keep their GPU memory.
                from multiprocessing.managers import SyncManager
                manager = SyncManager(ctx=mp.get_context("spawn"))
                manager.start(_sampling_manager_init, (os.getpid(),))
                self._sampling_manager = manager
                seed = np.random.get_state()[1][0]
                # ReferenceMotion takes character_model only as opaque metadata (see build_motion_lib)
                character_model = self.character_model
                if not isinstance(character_model, (str, os.PathLike)):
                    character_model = str(list(character_model)[0])
                for n, config in self.discriminators.items():
                    q = manager.Queue(maxsize=1)
                    self.disc_ref_motion[n] = q
                    key_links = config.key_links
                    if key_links is None:  # all links are key links and observable
                        parent_link_index = config.parent_link
                        key_links_index = None
                    if config.global_pos:
                        assert config.parent_link is None
                        parent_link_index = None
                        key_links_index = None
                    elif config.parent_link is None: # parent link is the root, ensure it appears as the first in the key link list
                        parent_link_index = None
                        if 0 in key_links:
                            key_links = [0] + [_ for _ in key_links if _ != 0] # root link is the first key links
                            key_links_index = None # all links in the key link list are key links for observation
                        else:
                            key_links_index = list(range(1, len(key_links)+1))
                            key_links = [0] + key_links # the root link in the key link list but not for observation
                    else:
                        if config.parent_link in key_links:
                            key_links_index = None
                        else:
                            key_links_index = list(range(1, len(key_links)+1))
                            key_links = [config.parent_link] + key_links
                        parent_link_index = key_links.index(config.parent_link)
                    if config.key_link_index_offset is not None:
                        key_links = [_+config.key_link_index_offset for _ in key_links]

                    p = mp.Process(target=self.__class__.ref_motion_sample, daemon=True, args=(os.getpid(), q,
                        seed+1+config.id, self.step_time, self.n_envs, config.ob_horizon, key_links_index, parent_link_index, config.local_pos, config.global_pos, config.replay_speed,
                        dict(motion_file=config.motion_file, character_model=character_model,
                            key_links=key_links, device=self.device
                        )
                    ))
                    p.start()
                    self.sampling_workers.append(p)

            self.real_samples = [{n: None for n in self.disc_ref_motion.keys()} for _ in range(128)]
            for n, q in self.disc_ref_motion.items():
                try:
                    batch = q.get()
                except (OSError, EOFError) as e:  # BrokenPipeError and ConnectionError are OSError
                    self._report_sampling_failure(n, e)
                    raise
                for i, v in enumerate(batch):
                    self.real_samples[i][n] = v.to(self.device)
        return self.real_samples.pop()

    def _report_sampling_failure(self, name, error):
        # Exit codes: None = still alive, 0 = clean exit (for example the
        # parent watchdog), 1 = uncaught exception, -N = killed by signal N.
        lines = ["Reference motion queue '{}' failed: {!r}".format(name, error)]
        proc = getattr(getattr(self, "_sampling_manager", None), "_process", None)
        if proc is not None:
            lines.append("  queue manager pid {} exitcode {}".format(proc.pid, proc.exitcode))
        for n, p in zip(self.disc_ref_motion.keys(), self.sampling_workers):
            lines.append("  sampling worker '{}' pid {} exitcode {}".format(n, p.pid, p.exitcode))
        print("\n".join(lines), file=sys.stderr, flush=True)

    @staticmethod
    def ref_motion_sample(parent_pid, queue, seed, step_time, n_inst, ob_horizon, key_links, parent_link, local_pos, global_pos, replay_speed, kwargs):
        import faulthandler
        faulthandler.enable()
        _exit_with_parent(parent_pid, name="sampling worker")
        np.random.seed(seed)
        torch.set_num_threads(1)
        lib = ReferenceMotion(**kwargs)
        if replay_speed is not None:
            replay_speed = eval(replay_speed)
        while True:
            obs = []
            for _ in range(128):
                if replay_speed is None:
                    dt = step_time
                else:
                    dt = step_time * replay_speed(n_inst)
                motion_ids, motion_times0 = lib.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                motion_ids = np.tile(motion_ids, ob_horizon)
                motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
                link_tensor = lib.state(motion_ids, motion_times, with_joint_tensor=False)
                samples = link_tensor.view(ob_horizon, n_inst, -1)
                ob = observe_iccgan(samples, None, key_links, parent_link, include_vel=False, local_pos=local_pos, global_pos=global_pos)
                obs.append(ob.cpu())
            queue.put(obs)


def _sampling_manager_init(parent_pid):
    """Initializer of the reference motion queue manager process."""
    import faulthandler, logging
    import multiprocessing.util
    faulthandler.enable()  # print a traceback on SIGSEGV/SIGABRT
    # managers.py reports a failed client connection only through this logger
    multiprocessing.util.log_to_stderr(logging.INFO)
    _exit_with_parent(parent_pid, name="queue manager")


def _exit_with_parent(parent_pid, interval=5.0, name=None):
    """In a child process: exit as soon as the parent process is gone."""
    import threading
    def watch():
        while True:
            time.sleep(interval)
            if os.getppid() != parent_pid:
                if name is not None:
                    print("{} (pid {}): parent {} is gone, exit".format(name, os.getpid(), parent_pid), file=sys.stderr, flush=True)
                os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: Optional[torch.Tensor]=None,
    key_links: Optional[List[int]]=None, parent_link: Optional[int]=None,
    include_vel: bool=True, local_pos: Optional[bool]=None, global_pos: bool=False,
    ground_height:Optional[torch.Tensor]=None, key_link_index: Optional[torch.Tensor]=None
):
    # key_link_index: key_links as a tensor on the state's device. Selecting with
    # it avoids copying the Python list to the GPU (a host sync) on every call.
    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    link_tensor = state_hist.view(n_hist, n_inst, -1, 13)
    if key_links is None:
        key_link_tensor = link_tensor
    elif key_link_index is not None:
        key_link_tensor = link_tensor.index_select(2, key_link_index)
    else:
        key_link_tensor = link_tensor[:,:,key_links]
    if global_pos:
        if include_vel:
            ob = key_link_tensor
        else:
            ob = key_link_tensor[...,:7]
    else:
        link_pos, link_orient = key_link_tensor[...,:3], key_link_tensor[...,3:7]
        if parent_link is None:
            root_tensor = state_hist[..., :13]
            if local_pos is True:
                origin = root_tensor[:,:, :3]
                orient = root_tensor[:,:,3:7]
            else:
                origin = root_tensor[-1,:, :3]
                orient = root_tensor[-1,:,3:7]

            heading = heading_zup(orient)
            up_dir = torch.zeros_like(origin)
            up_dir[..., UP_AXIS] = 1
            orient_inv = axang2quat(up_dir, -heading)
            orient_inv = orient_inv.view(-1, n_inst, 1, 4)

            origin = origin.clone()
            if ground_height is None:
                origin[..., UP_AXIS] = 0
            else:
                origin[..., UP_AXIS] = ground_height
            origin.unsqueeze_(-2)
        else:
            if local_pos is False:
                origin = link_tensor[-1,:, parent_link, :3]
                orient = link_tensor[-1,:, parent_link,3:7]
            else:
                origin = link_tensor[:,:, parent_link, :3]
                orient = link_tensor[:,:, parent_link,3:7]
            orient_inv = quatconj(orient)
            orient_inv = orient_inv.view(-1, n_inst, 1, 4)
            origin = origin.unsqueeze(-2)

        ob_link_pos = link_pos - origin
        ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
        ob_link_orient = quatmultiply(orient_inv, link_orient)

        if include_vel:
            link_lin_vel, link_ang_vel = key_link_tensor[...,7:10], key_link_tensor[...,10:13]
            ob_link_lin_vel = rotatepoint(orient_inv, link_lin_vel)
            ob_link_ang_vel = rotatepoint(orient_inv, link_ang_vel)
            ob = torch.cat((ob_link_pos, ob_link_orient,
                ob_link_lin_vel, ob_link_ang_vel), -1)
        else:
            ob = torch.cat((ob_link_pos, ob_link_orient), -1)
    ob = torch.reshape(ob, (n_hist, n_inst, -1))
    ob1 = ob.permute(1, 0, 2)
    if seq_len is None: return ob1

    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    # Move the last min(seq_len, n_hist) frames of each instance to the front and
    # zero the rest. A gather, unlike boolean-mask indexing, needs no host sync.
    seq_len_ = seq_len.clamp(max=n_hist).unsqueeze(1)
    src = (arange + (n_hist - seq_len_)).clamp_(min=0, max=n_hist-1)
    ob2 = torch.gather(ob1, 1, src.unsqueeze(-1).expand(-1, -1, ob1.size(-1)))
    ob2 = ob2.masked_fill((arange >= seq_len_).unsqueeze(-1), 0.)
    return ob2


class PianoBase(ICCGANHumanoid):
    GOAL_REWARD_WEIGHT = 1
    ENABLE_GOAL_TIMER = False

    N_KEYS = 88

    def __init__(self, *args, **kwargs):

        self.note_file = kwargs.get("note_file")
        max_note_t = kwargs.get("max_note_t", 20) # maximal duration (in frames) for one note during training
        assert max_note_t >= 2
        self.max_note_t2 = max_note_t//2

        self.n_keys = kwargs.get("n_keys", self.N_KEYS)
        self.enable_audio = kwargs.get("enable_audio", True)
        self.audio_gain = kwargs.get("audio_gain", 0.16)
        self.audio_backend = kwargs.get("audio_backend", "auto")
        self.audio_reverb = kwargs.get("audio_reverb", 0.3)
        self._piano_audio = None
        self._audio_key_state = None
        self._audio_key_position = None
        self._audio_warning_printed = False
        
        self.random_init_note = kwargs.get("random_init_note", True)
        self.random_init = False # for self.reset_envs()

        # take n consecutive frames as the goal input
        self.goal_horizon =kwargs.get("goal_horizon", 5)
        # number of empty frames before resampling new target notes
        self.grace_period = kwargs.get("grace_period", 10)

        self.importance_sampling = False
        # None: no hard limit, an episode ends with the last note of the track
        self.episode_length_default = episode_limit(kwargs.get("episode_length"))

        goal_reward_weight = kwargs.get("goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.multi_objective_reward = hasattr(goal_reward_weight, "__len__") and len(goal_reward_weight) > 1

        super().__init__(*args, **kwargs)

        # How many sampling episodes on average each environment will handle
        # The whole dataset will be roughly splited into N_envs-by-N_importance_sampling chunks at most.
        # When N_importance_sampling = 0, the wholde dataset will not splited, i.e. N_chunks == N_frames.
        # The more chunks there are, the more sampling steps are needed to update the (performance) importance w.r.t each chunk.
        # The training epochs that are neeeded to update the performance of all chunks:
        #       N_chunks/N_envs * (episode_length/rollout_length)
        # where int(N_chunks/N_envs) ~= N_importance_sampling
        importance_sampling = kwargs.get("importance_sampling", -1)
        self.importance_discount = kwargs.get("importance_discount", 0.99)
        self.importance_scale = kwargs.get("importance_scale", 6)
        self.importance_decay = kwargs.get("importance_decay", 0.5)
        self.importance_sampling = importance_sampling > -1
        if self.importance_sampling:
            self.random_init_note = True

            n_envs = self.n_envs
            n_tracks = len(self.track_note_tensor)
            n_notes = torch.sum(self.track_n_notes_tensor).item()

            if importance_sampling:
                tail = int(np.ceil(n_notes / (n_envs*importance_sampling+n_tracks)))
            else:
                tail = 0
            samples_ = []
            for start, length in zip(self.track_tensor_offset, self.track_n_notes_tensor):
                n = max(1, length.item()-max(tail, 1))
                samples_.append((start.item(), start.item()+n))
            
            tot = sum(e-s for s, e in samples_)
            if importance_sampling:
                gap = max(1, tot//(n_envs*importance_sampling))
            else:
                gap = 1
            self.sampling_gap = int(gap)

            samples = []
            for s, e in samples_:
                cnt = 0
                for i in np.arange(s, e, self.sampling_gap):
                    cnt += self.track_t_tensor[i]
                    if torch.all(self.track_note_tensor[i]==0).item():
                        if self.sampling_gap > 1:
                            samples.append(i+1)
                    else:
                        samples.append(i)

            samples = np.sort(np.unique(samples))

            self.track_note_samples = torch.tensor(samples, device=self.device, dtype=torch.int64)
            self.average_reward = 0.1*torch.ones(self.track_note_samples.shape, dtype=torch.float32, device=self.device)
            print("IMPORTANCE SAMPLING: {} chunks with sampling gap of {} notes".format(self.average_reward.size(0), self.sampling_gap))
            
            self.cumulative_reward = torch.zeros((n_envs,), dtype=torch.float, device=self.device)
            self.cumulative_reward_discount = torch.full((n_envs,), 1, dtype=torch.float, device=self.device)
            self.sampling_idx = torch.empty((n_envs,), dtype=torch.long, device=self.device)

            self.cumulative_reward_buffer = torch.empty_like(self.average_reward)

            track_id = []
            for i in range(len(self.track_tensor_offset)):
                s = self.track_tensor_offset[i]
                e = len(self.track_note_tensor) if i==len(self.track_tensor_offset)-1 else self.track_tensor_offset[i+1]
                track_id.extend([i]*(e-s))
            self.track_id_tensor = torch.tensor(track_id, dtype=torch.int32, device=self.device)
            
            if self.enable_goal_timer:
                max_track_frames = 0
                for i in range(len(self.track_tensor_offset)):
                    s = self.track_tensor_offset[i]
                    e = len(self.track_note_tensor) if i==len(self.track_tensor_offset)-1 else self.track_tensor_offset[i+1]
                    dt = self.track_t_tensor[s:e]
                    for j in range(len(dt)-self.note_sampling_range[1]):
                        max_track_frames = max(max_track_frames, torch.sum(dt[j:j+self.note_sampling_range[1]]).item())
                max_steps = max_track_frames + self.grace_period
            elif self.episode_length_default is not None:
                max_steps = self.episode_length_default
            else:
                # no hard limit: the longest episode plays a whole track
                max_steps = int(max(self.track_t_tensor[s:e].sum().item()
                    for s, e in zip(self.track_tensor_offset.tolist(), self.track_tensor_offset.tolist()[1:] + [len(self.track_t_tensor)]))) + self.grace_period + 1

            discount_rew = [0]*self.grace_period+[self.importance_discount**i for i in range(max_steps-self.grace_period)]+[0]
            boostrap_reward = np.flip(np.cumsum(np.flip(discount_rew)))
            self.boostrap_reward = torch.tensor(boostrap_reward.copy(), dtype=torch.float, device=self.device)
            self.max_rew = self.boostrap_reward[0].item()
            self.goal_reset_stamp = torch.empty((n_envs,), dtype=torch.long, device=self.device)
            self.importance_resampling_at = torch.empty((n_envs,), dtype=torch.int32, device=self.device)
            self.info["log"]["acc_rew0"] = self.average_reward[0]
            self.info["log"]["acc_rew_low"] = torch.min(self.average_reward)

    def __del__(self):
        audio = getattr(self, "_piano_audio", None)
        if audio is not None:
            audio.close()
        super().__del__()

    def _start_audio(self):
        if not self.enable_audio or self._piano_audio is not None:
            return
        try:
            from piano_audio import PianoAudio

            audio = PianoAudio(
                gain=self.audio_gain,
                backend=self.audio_backend,
                reverb=self.audio_reverb,
            )
            if audio.available:
                self._piano_audio = audio
                print("Piano audio backend: {}.".format(audio.backend))
            elif not self._audio_warning_printed:
                print("Piano audio disabled: {}".format(audio.error))
                self._audio_warning_printed = True
                self.enable_audio = False
        except Exception as exc:
            if not self._audio_warning_printed:
                print("Piano audio disabled: {}".format(exc))
                self._audio_warning_printed = True
            self.enable_audio = False

    def _play_audio_for_pressed_keys(self, key_activated):
        if not self.enable_audio:
            return
        key_pos = self.key_pos[0].detach().cpu().numpy()
        if self._audio_key_state is None:
            self._audio_key_state = key_activated.copy()
            self._audio_key_position = key_pos.copy()
            return
        self._start_audio()
        audio = self._piano_audio
        if audio is None:
            self._audio_key_state = key_activated.copy()
            self._audio_key_position = key_pos.copy()
            return

        # The audio follows the env's own press state (key_activated). Each
        # release that the env counts, even for one frame, gives a new note on
        # the next press.
        audio_key_state = np.asarray(key_activated, dtype=bool).copy()
        pressed = np.logical_and(audio_key_state, np.logical_not(self._audio_key_state))
        released = np.logical_and(self._audio_key_state, np.logical_not(audio_key_state))
        for key in np.flatnonzero(released):
            audio.release_key(int(key))

        if np.any(pressed):
            previous_pos = self._audio_key_position
            if previous_pos is None:
                previous_pos = key_pos
            press_speed = np.maximum(key_pos - previous_pos, 0) * self.fps
            for key in np.flatnonzero(pressed):
                velocity = 0.18 + 0.82 * (
                    1.0 - math.exp(-float(press_speed[key]) / 8.0)
                )
                audio.play_key(int(key), velocity=velocity)
        self._audio_key_state = audio_key_state
        self._audio_key_position = key_pos.copy()

    def _make_simulator(self):
        self.goal_note_tensor = torch.zeros((self.n_envs, self.goal_horizon, self.n_keys), dtype=torch.int32, device=self.device)
        self.goal_t_tensor = torch.empty((self.n_envs, self.goal_horizon), dtype=torch.int32, device=self.device)
        self.sampling_track_id = torch.empty((self.n_envs,), dtype=torch.int64, device=self.device)
        self.sampling_note_id = torch.empty((self.n_envs,), dtype=torch.int64, device=self.device)

        # 0 left pinky, ..., 4 left thumb, 5 right thumb, ..., 9 right pinky
        rigid_body = {self.mj_model.body(i).name: i-1 for i in range(1, self.mj_model.nbody)} # exclude the world link
        # NOTE finger_tips is ordered from left to right
        finger_tips = [rigid_body[n] if n in rigid_body else 0 for n in
            ("L:LFtip", "L:RFtip", "L:MFtip", "L:IFtip", "L:THtip", "R:THtip", "R:IFtip", "R:MFtip", "R:RFtip", "R:LFtip")]
        assert not all(fid==0 for fid in finger_tips)
        self.finger_tips = torch.tensor(finger_tips, dtype=torch.int, device=self.device)
        self.finger_tips_valid = torch.tensor([_ for _ in finger_tips if _ != 0], dtype=torch.int, device=self.device)

        self.right_hand = finger_tips[0] == 0
        self.two_hands = all(_!=0 for _ in finger_tips)
        self.right_hand_only = not self.two_hands and self.right_hand

        self.load_note(self.note_file)

        # The MJX upstream additionally (1) disabled the collision of piano keys
        # farther than two keys from any key in the loaded notes and
        # (2) replaced the MJX collision driver with a custom one that culls
        # finger-key pairs to the keys under each hand before running the
        # narrow phase. Both only cut the cost of MJX's static, all-pairs
        # contact enumeration. mujoco_warp's broadphase prunes the pairs per
        # step, so the model collides as specified by its MJCF and steps
        # through the standard mujoco_warp pipeline.
        super()._make_simulator()


    def prepare_sim(self, opt={}):
        opt["disableflags"] = int(mujoco.mjtDisableBit.mjDSBL_GRAVITY) | int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        opt["gravity"] = np.array([0, 0, 0])
        super().prepare_sim(opt)
        self.origin = torch.empty((self.n_envs, 1, 3), dtype=torch.float, device=self.device)
        self.orient = torch.empty((self.n_envs, 1, 4), dtype=torch.float, device=self.device)
        self.orient_inv = torch.empty((self.n_envs, 1, 4), dtype=torch.float, device=self.device)

        self.has_piano = any("P:" in self.mj_model.body(i).name for i in range(self.mj_model.nbody))
        self.n_char_links = min(i for i in range(self.mj_model.nbody) if "P:" in self.mj_model.body(i).name) - 1 # exclude world body

        # we assume piano is placed along x axis (left to right) and key is along y axis
        piano_key0_link = min(i for i in range(1, self.mj_model.nbody) if "P:" in self.mj_model.body(i).name and "key" in self.mj_model.body(i).name)
        piano_key0_geom = min(i for i in range(self.mj_model.ngeom) if "P:" in self.mj_model.geom(i).name and "key" in self.mj_model.geom(i).name)
        piano_key_pos, piano_key_edge0, piano_key_edge1 = [], [], []
        piano_key_edge01, piano_key_height = [], []
        for i in range(self.n_keys):
            g = self.mj_model.geom(piano_key0_geom+i)
            p, s = g.pos, g.size
            c = p + self.mj_model.body(piano_key0_link+i).pos
            # extend left and right by 5mm and cut the bottom by 5mm for finger tip placement
            piano_key_edge0.append(c[:2]-s[:2]-0.005*(p[:2]==0))
            piano_key_edge01.append(c[:2]+s[:2]+0.005*(p[:2]==0))
            piano_key_edge1.append(c[:2]+s[:2]+0.005*(p[:2]==0)-0.005*(p[:2]!=0))
            piano_key_height.append(c[2]+s[2])
            p = c + (s*(0.8))*(p!=0)  # 0.6 for 80% position, 0.7 for 85%, 0.8 for 90%
            p[2] += s[2]
            piano_key_pos.append(p)
        self.piano_key_pos = torch.tensor(np.array(piano_key_pos), dtype=torch.float, device=self.device)
        piano_key_edge0 = np.array(piano_key_edge0)
        piano_key_edge1 = np.array(piano_key_edge1)
        piano_key_edge01 = np.array(piano_key_edge01)
        piano_key_height = np.array(piano_key_height)
        self.piano_key0_geom = piano_key0_geom

        # # let white keys's effective edge to be below the black keys' area
        self.piano_key_edge0 = torch.tensor(piano_key_edge0, dtype=torch.float, device=self.device)
        self.piano_key_edge1 = torch.tensor(piano_key_edge1, dtype=torch.float, device=self.device)
        self.piano_key_edge01 = torch.tensor(piano_key_edge01, dtype=torch.float, device=self.device)
        self.piano_key_height = torch.tensor(piano_key_height, dtype=torch.float, device=self.device)

        piano_key0_joint = min(i for i in range(self.mj_model.njnt) if "P:" in self.mj_model.joint(i).name and "joint" in self.mj_model.joint(i).name)
        self.piano_key_joints = [self.mj_model.joint(piano_key0_joint).qposadr.item()+i for i in range(self.n_keys)]
        self.piano_key_joints_tensor = torch.tensor(self.piano_key_joints, dtype=torch.int64, device=self.device)
        self.piano_key_depth = torch.tensor([self.mj_model.joint(piano_key0_joint+i).range[0].item() for i in range(self.n_keys)], dtype=torch.float, device=self.device)

        self.arange_tensor_n_envs = torch.arange(self.n_envs, device=self.device)
        self.arange_tensor_n_fingers = torch.arange(len(self.finger_tips_valid), device=self.device)
        self.arange_tensor_n_keys = torch.arange(self.n_keys, device=self.device)

        self.key_activated = torch.zeros((self.n_envs, self.n_keys), dtype=torch.bool, device=self.device)
        self.key_ever_activated = torch.zeros((self.n_envs, self.n_keys), dtype=torch.bool, device=self.device)
        self.key_pos = torch.zeros((self.n_envs, self.n_keys), dtype=torch.float, device=self.device)

        n_fingers = len(self.finger_tips_valid)//2
        if n_fingers == 5:
            self.fingering_heuristic = True

            def ascend_permute(_r, _n):
                _choice = [[_] for _ in range(1, _n+1-_r+1)]
                for _ in range(1, _r):
                    _choice_ = []
                    for _c in _choice:
                        _choice_.extend([_c+[_] for _ in range(_c[-1]+1, _n+1)])
                    _choice = _choice_
                return _choice
            
            n = 1
            finger_schemes = [[[0 for _ in range(n_fingers+1)]]]
            for n in range(n_fingers):
                if n > 0:
                    choice = ascend_permute(n+1, n_fingers-1)
                    choice_ = ascend_permute(n, n_fingers-1)
                    for digit in range(n+1):
                        for c in choice_:
                            c_ = c.copy()
                            c_.insert(digit, 5)
                            choice.append(c_)
                else:
                    choice = ascend_permute(n+1, n_fingers)
                finger_schemes.append([[0] + c + [0]*(n_fingers-len(c)) for c in choice])


            n_schemes = max((len(_) for _ in finger_schemes))
            finger_schemes = [c+[c[-1] for _ in range(n_schemes-len(c))] for c in finger_schemes]
            self.finger_schemes = torch.tensor(finger_schemes, device=self.device, dtype=torch.int32)
            self.arange_tensor_n_schemes = torch.arange(n_schemes, device=self.device).unsqueeze_(-1) * self.finger_schemes.size(-1)

            black_key = False
            black_keys = []
            key_gap = []
            for i in range(self.n_keys):
                if "black" in self.mj_model.geom(self.piano_key0_geom + i).name:
                    key_gap.append(1)
                    black_key = True
                    black_keys.append(True)
                else:
                    if black_key:
                        key_gap.append(1)
                    else:
                        key_gap.append(2)
                    black_key = False
                    black_keys.append(False)
            # key_gap = [0.5 if "black" in self.mj_model.geom(self.piano_key0_geom + i).name else 1 for i in range(self.n_keys)]
            self.key_gap = torch.tensor(np.cumsum(key_gap), dtype=torch.int32, device=self.device)

            scheme_gap = []
            thumb_last = []
            for sch in finger_schemes:
                g = []
                for c in sch:
                    c = np.array(c)
                    activated = c!=0
                    tot = np.sum(activated)
                    if tot > 1:
                        activated_ = np.sort(np.nonzero(c[activated])[0])
                        min_ = c[activated][activated_[0]]
                        max_ = c[activated][activated_[-1]]
                        if min_ == 1:
                            if max_ == 2:
                                gap = 3
                            elif max_ == 3:
                                gap = 4.5 #5
                            elif max_ == 4:
                                gap = 5.5 #6
                            else:
                                gap = 8 # little and thumb
                        elif min_ == 2:
                            if max_ == 3:
                                gap = 3
                            elif max_ == 4:
                                gap = 4.5 #6
                            else:
                                gap = 8
                        elif min_ == 3:
                            if max_ == 4:
                                gap = 3.5 # 4
                            else:
                                gap = 8
                        elif min_ == 4:
                            gap = 8
                        else:
                            gap = 5
                        thumb_last.append(bool(max_ == 5))
                    else:
                        gap = 0
                        min_= None
                        max_=None
                        thumb_last.append(bool(5 in c))
                    g.append(gap)
                scheme_gap.append(g)
            self.scheme_gap = torch.tensor(scheme_gap, dtype=torch.float, device=self.device).mul_(2).to(torch.int32)
            self.thumb_last = torch.tensor(thumb_last, dtype=torch.bool, device=self.device).reshape(*self.scheme_gap.shape)

            self.black_keys = torch.tensor(black_keys, dtype=torch.bool, device=self.device)
            self.white_keys = ~self.black_keys

        else:
            self.fingering_heuristic = False
            print("[Warn] Unsupport to fingering heuristic for hands with fingers less or more than five.")

    def load_note(self, note_file):
        notes = ["A0", "A#0", "B0"]
        for octave in range(1, 8):
            for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
                notes.append(note + str(octave))
        notes.append("C8")
        note2key_number = {note: i for i, note in enumerate(notes)}
        note2key_number["Bb0"] = note2key_number["A#0"]
        for octave in range(1, 8):
            note2key_number["Db"+str(octave)] = note2key_number["C#"+str(octave)]
            note2key_number["Eb"+str(octave)] = note2key_number["D#"+str(octave)]
            note2key_number["Gb"+str(octave)] = note2key_number["F#"+str(octave)]
            note2key_number["Ab"+str(octave)] = note2key_number["G#"+str(octave)]
            note2key_number["Bb"+str(octave)] = note2key_number["A#"+str(octave)]
        midi2key_number = {k + 21: k for k in range(88)}

        adjcent_keys = {}
        adjcent_keys[note2key_number["A0"]] = (note2key_number["A#0"], note2key_number["B0"])
        adjcent_keys[note2key_number["A#0"]] = (note2key_number["A0"], note2key_number["B0"])
        adjcent_keys[note2key_number["B0"]] = (note2key_number["A0"], note2key_number["A#0"], note2key_number["C1"])
        adjcent_keys[note2key_number["C8"]] = (note2key_number["B7"],)
        for octave in range(1, 8):
            adjcent_keys[note2key_number["C"+str(octave)]] = (note2key_number["B"+str(octave-1)], note2key_number["C#"+str(octave)], note2key_number["D"+str(octave)])
            adjcent_keys[note2key_number["D"+str(octave)]] = (note2key_number["C"+str(octave)], note2key_number["C#"+str(octave)], note2key_number["D#"+str(octave)], note2key_number["E"+str(octave)])
            adjcent_keys[note2key_number["E"+str(octave)]] = (note2key_number["D"+str(octave)], note2key_number["D#"+str(octave)], note2key_number["F"+str(octave)])
            adjcent_keys[note2key_number["F"+str(octave)]] = (note2key_number["E"+str(octave)], note2key_number["F#"+str(octave)], note2key_number["G"+str(octave)])
            adjcent_keys[note2key_number["G"+str(octave)]] = (note2key_number["F"+str(octave)], note2key_number["F#"+str(octave)], note2key_number["G#"+str(octave)], note2key_number["A"+str(octave)])
            adjcent_keys[note2key_number["A"+str(octave)]] = (note2key_number["G"+str(octave)], note2key_number["G#"+str(octave)], note2key_number["A#"+str(octave)], note2key_number["B"+str(octave)])
            adjcent_keys[note2key_number["B"+str(octave)]] = (note2key_number["A"+str(octave)], note2key_number["A#"+str(octave)], note2key_number["C"+str(octave+1)])
            adjcent_keys[note2key_number["C#"+str(octave)]] = (note2key_number["C"+str(octave)], note2key_number["D"+str(octave)])
            adjcent_keys[note2key_number["D#"+str(octave)]] = (note2key_number["D"+str(octave)], note2key_number["E"+str(octave)])
            adjcent_keys[note2key_number["F#"+str(octave)]] = (note2key_number["F"+str(octave)], note2key_number["G"+str(octave)])
            adjcent_keys[note2key_number["G#"+str(octave)]] = (note2key_number["G"+str(octave)], note2key_number["A"+str(octave)])
            adjcent_keys[note2key_number["A#"+str(octave)]] = (note2key_number["A"+str(octave)], note2key_number["B"+str(octave)])
        black_keys = []
        for n in notes:
            if "#" in n:
                black_keys.append(note2key_number[n])
        adjcent_keys_ = {}
        for k, n in adjcent_keys.items():
            if k in black_keys: continue
            n = tuple(_ for _ in n if _ not in black_keys)
            if len(n): adjcent_keys_[k] = n
        adjcent_keys = adjcent_keys_

        
        finger_tips = [None] + self.finger_tips.cpu().tolist() + [None, None]

        track = []
        t = []

        self.fingering = False
        self.masked_fingering = False
        self.masked_fingering_left = False
        self.masked_fingering_right = False
        single_hand = not torch.all(torch.any(self.finger_tips.view(2, -1), -1)).item()

        fps = self.fps
        for filename in note_file:
            note, dt = [], []
            # FingeringDataset
            with open(filename, "r") as f:
                midi = f.readlines()
            t_start, t_end = 1000000, -1
            items, ts = [], []
            is_midi_definition = False
            for item in midi:
                item = item.strip()
                if not item or item.startswith("//"): continue
                _ = item.split()
                rid, t0, t1, pitch, v0, v1 = _[:6]
                if len(_) > 8:
                    assert len(item.split()) in [6, 7, 8]
                elif len(_) == 6:
                    # unknown clef, unknonw finger
                    ch = 2
                    finger_id = "x"
                elif len(_) == 7:
                    # known clef, unknown finger
                    ch = _[6]
                    finger_id = "x"
                else:
                    # known clef, known finger
                    ch = _[6]
                    finger_id = _[7]
                # -5 left pinky, -1 left thumb, 1 right thumb, 5 right pinky
                try:
                    fid = int(finger_id)
                except:
                    fid = "x"
                
                if fid not in [-5,-4,-3,-2,-1,1,2,3,4,5]:
                    # unknown fingering
                    if int(ch) == 1:
                        #left hand, based on clef
                        if finger_tips[3] == 0:
                            # ignore note for undefined hand
                            continue
                        fid = 7 if single_hand else -6
                        if single_hand:
                            self.masked_fingering = True
                        else:
                            self.masked_fingering_left = True
                    elif int(ch) == 0:
                        #right hand, based on clef
                        if finger_tips[-3] == 0:
                            # ignore note for undefined hand
                            continue
                        fid = 7 if single_hand else 6
                        if single_hand:
                            self.masked_fingering = True
                        else:
                            self.masked_fingering_right = True
                    else:
                        #unknown hand, unknonw fingering
                        fid = 7
                        self.masked_fingering = True
                # convert to: 0 not assigned (left hand), 1 left pinky, 5 left thumb, 6 right thumb, 10 right pinky, 11 not assigned (right hand),12 not assigned
                fid = fid+6 if fid < 0 else fid+5
                if finger_tips[fid] == 0:
                    # ignore note for undefined hand
                    continue
                self.fingering = self.fingering or finger_tips[fid] is not None


                t0, t1 = float(t0), float(t1)
                try:
                    _ = int(pitch)
                    is_midi_definition = True
                except:
                    is_midi_definition = False
                items.append((rid, pitch, v0, v1, fid, item))
                ts.append((t0, t1))
                t_start = min(t_start, t0)
                t_end = max(t_end, t1)
            ts0 = [int(round((t0-t_start)*fps)) for t0, t1 in ts]
            ts1 = [int(round((t1-t_start)*fps)) for t0, t1 in ts]
                
            record = np.zeros((int(round((t_end-t_start)*fps)), self.n_keys, 13, 2), dtype=bool)
            for (rid, pitch, v0, v1, fid, item), t0, t1 in zip(items, ts0, ts1):
                pitch = midi2key_number[int(pitch)] if is_midi_definition else note2key_number[pitch]
                if t1 - t0 < 3:
                    print("Skip note less than 3 frames\n({}) [ignored]".format(item))
                    continue
                if fid > 0 and fid < 11:
                    if np.any(np.sum(record[t0:t1, ..., fid, 0], 1) > 0):
                        multiple_key_assigned = False
                        adjcent_key_assigned = False
                        for tid in range(t0, t1):
                            pid = np.where(record[tid, :, fid, 0])[0]
                            if len(pid) and pitch not in pid:
                                if len(pid) > 1:
                                    if pid[0] not in adjcent_keys[pitch]:
                                        multiple_key_assigned = True
                                        break
                                    else:
                                        adjcent_key_assigned = True
                        if multiple_key_assigned:
                            print("Multiple keys are assigned for finger {}\n({}) [ignored]".format(fid, item))
                            continue
                        if adjcent_key_assigned:
                            print("Adjcent keys are assigned for one finger {}\n({})".format(fid, item))
                if np.any(record[t0:t1, pitch, :, 0]):
                    print("Key {} is already set for another finger\n({})".format(pitch, item))
                record[t0:t1, pitch, :, 0] = False
                record[t0:t1, pitch, fid, 0] = True
                record[t0, pitch, fid, 1] = True
            # if two consecutive notes are the same without any interval,
            # disable the last frame of the previous note
            # to force the hand to lift and re-press rather than keeping pressing

            # check if there is any key that needs to be pressed while has not been released
            # then disable the last frame of pressed notes before the new pressing
            masked = np.logical_and(
                np.any(record[:-1, ..., 0], axis=-1),    # T x N_keys, in pressed status
                np.any(record[1:, ..., 1], axis=-1)      # T x N_keys, a new pressing
            )                                            # T x N_keys
            for i, m in enumerate(masked):
                if np.any(m):
                    for k in range(len(record[i])):
                        # disable the last frame pressing for all notes if the last frame is the ending frame of them
                        # i.e. if next timestep is a new note or no note in the next timestep
                        if np.any(record[i+1, k, :, 1]) or np.all(record[i+1, k, :, 0]==False):
                            record[i, k] = False
            record = record[:-1, ..., 0]
            
            for rid, r in enumerate(record):
                n = np.zeros((self.n_keys,))
                key, finger = np.where(r)
                if len(key):
                    # from: 0, 1, ..., 5, 6, ..., 10, 11, 12
                    # convert to: -6 not assigned, -5 left pinky, ..., -1 left thumb, 1 right thumb, ..., 5 right pinky, 6,7 not assigned
                    n[key] = (finger-6)*(finger<6) + (finger-5)*(finger>5)
                if note and np.all(n == note[-1]):
                    dt[-1] += 1
                else:
                    left_pressed = np.where(np.logical_and(n>-6,n<0))[0]
                    right_pressed = np.where(np.logical_and(n<6,n>0))[0]
                    # check two-hand overlapping
                    if len(left_pressed) and len(right_pressed) and np.max(left_pressed) > np.min(right_pressed):
                        print("Detected overlapping of hands: {} (right), {} (left)".format(right_pressed.tolist(), left_pressed.tolist()))
                    # check span
                    if len(right_pressed):
                        span =  np.max(right_pressed) - np.min(right_pressed)
                        if span > 12:
                            print(n, span, len(note))
                            print("Detected Large span ({} keys) over one octave for right hand".format(span))
                    if len(left_pressed):
                        span =  np.max(left_pressed) - np.min(left_pressed)
                        if span > 12:
                            print("Detected Large span ({} keys) over one octave for left hand".format(span))
                    note.append(n)
                    dt.append(1)
            if np.all(note[-1] == 0):
                note.pop()
                dt.pop()
            if len(note) > 1:
                track.append(note)
                t.append(dt)
            
        assert any([self.fingering, self.masked_fingering, self.masked_fingering_left, self.masked_fingering_right])
        assert not single_hand or (not self.masked_fingering_left and not self.masked_fingering_right)
        self.mixed_fingering = sum(1 if _ else 0 for _ in [self.fingering, self.masked_fingering, self.masked_fingering_left, self.masked_fingering_right]) > 1

        zero_note = 0*track[0][0]
        for tr in track:
            for _ in range(self.goal_horizon):
                tr.append(zero_note)
        for dt in t:
            for _ in range(self.goal_horizon):
                dt.append(0)
        self.track_note_tensor = torch.tensor(np.array(sum(track, [])), dtype=torch.int32, device=self.device)
        self.track_t_tensor = torch.tensor(sum(t, []), dtype=torch.int32, device=self.device)
        track_offsets = np.cumsum([0] + [len(items) for items in track[:-1]])
        self.track_tensor_offset = torch.tensor(track_offsets, dtype=torch.int32, device=self.device)
        self.track_n_notes_tensor = torch.tensor([len(_)-self.goal_horizon for _ in track], dtype=torch.int32, device=self.device)
        print("Load {} notes ({:.4f}s) from {} files of {} tracks.".format(sum(len(note) for note in track), sum(sum(dt) for dt in t)/fps, len(note_file), len(track)))
        
    def reset(self):
        audio = getattr(self, "_piano_audio", None)
        if audio is not None:
            audio.all_notes_off()
        self._audio_key_state = None
        self._audio_key_position = None
        super().reset()
        self.track_length_tensor = torch.zeros_like(self.track_t_tensor)
        for i in range(len(self.track_tensor_offset)):
            s = self.track_tensor_offset[i]
            e = len(self.track_note_tensor) if i==len(self.track_tensor_offset)-1 else self.track_tensor_offset[i+1]
            self.track_length_tensor[s+1:e] = torch.cumsum(self.track_t_tensor[s:e-1], 0)
        no_limit = torch.iinfo(torch.int32).max
        self.episode_length = torch.full((self.n_envs,), no_limit if self.episode_length_default is None else self.episode_length_default, dtype=torch.int32, device=self.device)

        if self.importance_sampling:
            self.info["log"]["acc_rew0"] = self.average_reward[0]
            self.info["log"]["acc_rew_low"] = torch.min(self.average_reward)
    
    def reset_goal(self, env_ids):
        n_envs = len(env_ids)
        if self.importance_sampling:
            weights = self.max_rew/self.average_reward.clip_(min=0.1)
            weights.pow_(self.importance_scale).add_(-0.999)
            weights.clip_(min=1e-6, max=1e20).nan_to_num_(neginf=None, posinf=1e20, nan=None)
            sid = torch.multinomial(weights, len(env_ids), replacement=True) # N_envs

            if self.sampling_gap > 1:
                frac = torch.tensor(np.random.uniform(low=0.0, high=1.0, size=(len(env_ids),)), device=self.device, dtype=torch.float)
                fid = (self.track_note_samples[sid] + self.sampling_gap*frac).to(torch.int64)
            else:
                fid = self.track_note_samples[sid]

            self.sampling_idx[env_ids] = sid.to(torch.long)
            track_id = self.track_id_tensor[fid]
            note_id = fid - self.track_tensor_offset[track_id]

            self.goal_reset_stamp.index_fill_(0, env_ids, self.simulation_step)
        else:
            track_id = torch.randint(0, len(self.track_tensor_offset), size=(len(env_ids),), device=self.device)
            phase = torch.rand((n_envs, ), dtype=torch.float, device=self.device).mul_(1.1).sub_(0.1).clip_(min=0)
            note_id = ((self.track_n_notes_tensor[track_id]-self.goal_horizon) * phase).to(dtype=track_id.dtype, device=self.device)
        note_id += torch.all(self.track_note_tensor[note_id]==0, -1)

        if not self.training or not self.random_init_note:
            note_id[:] = 0
        note_id_ = self.track_n_notes_tensor[track_id]

        note_id += self.track_tensor_offset[track_id]
        note_id_ += self.track_tensor_offset[track_id]

        if not self.enable_goal_timer or not self.training:
            note_id_ += self.goal_horizon

        note_id_ -= (self.goal_horizon-1)
        note_id_.clip_(min=note_id)

        timer = self.track_length_tensor[note_id_] - self.track_length_tensor[note_id]
        timer.add_(self.grace_period)

        # Envs under initialization (lifetime 0, called from reset_envs) get a new
        # goal window that starts with a zero note; envs already running (called
        # when their goal timer expires) shift their window and append a zero note.
        # Both windows are computed for every env in env_ids and selected per env
        # with torch.where, so no host sync is needed to split env_ids. As in the
        # original, one call holds envs of one kind only.
        m = self.lifetime[env_ids]==0
        torch._assert_async(m.all().logical_or((~m).all()))
        H = self.goal_horizon
        goal_note = self.goal_note_tensor[env_ids]
        goal_t = self.goal_t_tensor[env_ids]
        # new window: zero note for grace_period frames, then notes note_id, ..., note_id+H-2
        window_ids = note_id.unsqueeze(1) + torch.arange(H-1, dtype=note_id.dtype, device=self.device)
        new_note = torch.cat((torch.zeros_like(goal_note[:, :1]), self.track_note_tensor[window_ids]), 1)
        new_t = torch.cat((torch.full_like(goal_t[:, :1], self.grace_period), self.track_t_tensor[window_ids]), 1)
        # running window: drop the first note, append a zero note for grace_period frames
        shift_note = torch.cat((goal_note[:, 1:], torch.zeros_like(goal_note[:, :1])), 1)
        shift_t = torch.cat((goal_t[:, 1:], torch.full_like(goal_t[:, :1], self.grace_period)), 1)
        m_ = m.view(-1, 1)
        self.goal_note_tensor[env_ids] = torch.where(m_.unsqueeze(-1), new_note, shift_note)
        self.goal_t_tensor[env_ids] = torch.where(m_, new_t, shift_t)
        timer = torch.where(m, timer, (timer + (goal_t.sum(-1)-1)).to(timer.dtype))
        note_id = torch.where(m, note_id + (H-1), note_id)
        self.goal_t_tensor[env_ids, 0] += 1 # observe function will reduce it by 1 directly

        self.sampling_note_id[env_ids] = note_id
        
        if self.training:
            if self.enable_goal_timer:
                self.goal_timer[env_ids] = timer
                lifetime = self.lifetime[env_ids]
                lifespan = lifetime.to(timer.dtype).add_(timer)
                resampling = lifespan + (self.track_length_tensor[note_id_+self.goal_horizon-1] - self.track_length_tensor[note_id_])
                self.episode_length[env_ids] = lifespan if self.episode_length_default is None else lifespan.clip_(min=self.episode_length_default)
            else:
                self.episode_length[env_ids] = timer if self.episode_length_default is None else torch.clip(timer, max=self.episode_length_default)

            if self.random_init_note:
                if self.enable_goal_timer:
                    self.importance_resampling_at[env_ids] = resampling
                else:
                    self.importance_resampling_at[env_ids] = self.episode_length[env_ids]
                
                if self.simulation_step == 0:
                    self.importance_resampling_at_current = self.importance_resampling_at.clone()
                    self.sampling_idx_current = self.sampling_idx.clone()
                else:
                    m = self.lifetime==0
                    self.importance_resampling_at_current.copy_(torch.where(m, self.importance_resampling_at, self.importance_resampling_at_current))
                    self.sampling_idx_current.copy_(torch.where(m, self.sampling_idx, self.sampling_idx_current))

        else:
            self.episode_length[env_ids] = timer
            if self.enable_goal_timer:
                self.goal_timer[env_ids] = timer
        
    
    def update_note_target(self, env_ids):
        if self.viewer_pause: return
        note_id = self.sampling_note_id[env_ids]
        self.goal_note_tensor[env_ids, :-1] = self.goal_note_tensor[env_ids, 1:].clone()
        self.goal_note_tensor[env_ids, -1] = self.track_note_tensor[note_id]
        self.goal_t_tensor[env_ids, :-1] = self.goal_t_tensor[env_ids, 1:].clone()
        self.goal_t_tensor[env_ids, -1] = self.track_t_tensor[note_id]
        self.sampling_note_id[env_ids] += 1

    def _update_note_target_masked(self, m):
        """update_note_target for the envs where the (n_envs,) bool mask m is set,
        computed for all envs and selected with torch.where (no host sync)."""
        if self.viewer_pause: return
        # envs outside m may point one past the last note; their read is discarded
        note_id = self.sampling_note_id.clamp(max=self.track_note_tensor.size(0)-1)
        shift_note = torch.cat((self.goal_note_tensor[:, 1:], self.track_note_tensor[note_id].unsqueeze(1)), 1)
        shift_t = torch.cat((self.goal_t_tensor[:, 1:], self.track_t_tensor[note_id].unsqueeze(1)), 1)
        m_ = m.view(-1, 1)
        self.goal_note_tensor.copy_(torch.where(m_.unsqueeze(-1), shift_note, self.goal_note_tensor))
        self.goal_t_tensor.copy_(torch.where(m_, shift_t, self.goal_t_tensor))
        self.sampling_note_id += m

    def init_state(self, env_ids):
        if self.simulation_step < 1:
            ref_link_tensor = torch.zeros((self.n_envs, self.mj_model.nbody-1, 13), device=self.device, dtype=torch.float32)
            ref_joint_tensor = torch.zeros((self.n_envs, self.mj_model.njnt-len(self.root_links)), device=self.device, dtype=torch.float32), \
                    torch.zeros((self.n_envs, self.mj_model.njnt-len(self.root_links)), device=self.device, dtype=torch.float32)
            if self.root_links:
                # reset the reference for the root links
                # the other body links do not matter, as their pose is decided by the joint rotation
                if self.two_hands:
                    ref_link_tensor[:, self.root_links[0], 0] = 0.2 # left
                    ref_link_tensor[:, self.root_links[1], 0] = -0.2 # right
                else:
                    ref_link_tensor[..., 0] = -0.2 if self.right_hand else 0.2
                ref_link_tensor[..., 1] = 0.5
                ref_link_tensor[..., 2] = 0.02
                ref_link_tensor[..., 3:6] = 0
                ref_link_tensor[..., 6] = 1
                ref_link_tensor[..., 7:] = 0
            else:
                if self.two_hands:
                    ref_joint_tensor[0][:,0] = 0.2 # left
                    ref_joint_tensor[0][:,1] = 0.5
                    ref_joint_tensor[0][:,2] = 0.02
                    idx = (self.mj_model.njnt-(self.n_keys if self.has_piano else 0))//2
                    ref_joint_tensor[0][:,idx] = -0.2 # right
                    ref_joint_tensor[0][:,idx+1] = 0.5
                    ref_joint_tensor[0][:,idx+2] = 0.02
                else:
                    ref_joint_tensor[0][:,0] = -0.2 if self.right_hand else 0.2
                    ref_joint_tensor[0][:,1] = 0.5
                    ref_joint_tensor[0][:,2] = 0.02

            self.ref_link_tensor0, self.ref_joint_tensor0 = ref_link_tensor, ref_joint_tensor
        else:
            n_envs = len(env_ids)
            ref_link_tensor = self.ref_link_tensor0[:n_envs]
            ref_joint_tensor = self.ref_joint_tensor0[0][:n_envs], self.ref_joint_tensor0[1][:n_envs]
        self.key_pos.index_fill_(0, env_ids, 0)
        self.key_activated.index_fill_(0, env_ids, False)
        return ref_link_tensor, ref_joint_tensor

    def get_goal_dim(self):
        return (self.n_keys+1) * self.goal_horizon

    def observe(self, env_ids=None):
        if self.simulation_step > 0 and self.training and env_ids is None and self.importance_sampling:
            # Update the importance weights of the samples whose episode (or goal
            # window) ended. Computed for all envs and masked by `w`, with the
            # same arithmetic as for the selected envs only, so no host sync
            # (nonzero/unique) is needed.
            w = (self.importance_resampling_at_current == self.lifetime).logical_or_((self.lifetime>0).logical_and_(self.done))
            sid = self.sampling_idx_current
            lifetime = (self.simulation_step - self.goal_reset_stamp).clamp_(min=0, max=self.boostrap_reward.size(0)-1)
            boot = self.boostrap_reward[lifetime]
            p = self.cumulative_reward/(self.max_rew-boot).nan_to_num_(nan=1.)
            cum = self.cumulative_reward + (~self.info["terminate"])*boot*p
            cnts = torch.zeros_like(self.average_reward, dtype=torch.int64).index_put_((sid,), w.to(torch.int64), accumulate=True)
            buf = torch.zeros_like(self.cumulative_reward_buffer).index_put_((sid,), torch.where(w, cum, 0.), accumulate=True)
            touched = cnts > 0
            self.cumulative_reward_buffer.copy_(torch.where(touched, buf, self.cumulative_reward_buffer))
            average_reward = self.average_reward*(1-self.importance_decay)
            average_reward += (buf/cnts).mul_(self.importance_decay)
            self.average_reward.copy_(torch.where(touched, average_reward, self.average_reward))
            self.cumulative_reward.masked_fill_(w, 0)
            self.cumulative_reward_discount.masked_fill_(w, 1)
            self.info["log"]["acc_rew_low"] = torch.where(touched.any(), torch.min(self.average_reward), self.info["log"]["acc_rew_low"])

            self.importance_resampling_at_current.copy_(torch.where(w, self.importance_resampling_at, self.importance_resampling_at_current))
            self.sampling_idx_current.copy_(torch.where(w, self.sampling_idx, self.sampling_idx_current))
 
        ob = super().observe(env_ids)

        if env_ids is None:
            self.goal_t_tensor[:, 0] -= 1
            self._update_note_target_masked(self.goal_t_tensor[:, 0]<1)
            goal = self.goal_note_tensor
            t = self.goal_t_tensor
        else:
            self.goal_t_tensor[env_ids, 0] -= 1
            m = torch.zeros((self.n_envs,), dtype=torch.bool, device=self.device)
            m[env_ids] = self.goal_t_tensor[env_ids, 0]<1
            self._update_note_target_masked(m)
            goal = self.goal_note_tensor[env_ids]
            t = self.goal_t_tensor[env_ids]

        n_envs = goal.size(0)
        # fingering
        goal = goal.to(torch.float)
        goal /= 6
        t = t.to(torch.float).div_(self.max_note_t2).clip_(max=2).sub_(1).unsqueeze_(-1)
        g = torch.cat((goal, t), -1).view(n_envs, -1)
        
        return torch.cat((ob, g), -1)

    @torch.no_grad
    def reward(self):
        goal_ = self.goal_note_tensor[:, 0]  # N_envs x N_keys
        key_qpos = self.sim.state("qpos")[:, self.piano_key_joints_tensor]  # N_envs x N_keys

        key_press_threshold = 0.9
        key_release_threshold = 0.9

        finger_id = None
        if self.fingering:
            finger_id = (goal_<0)*(goal_+5) + (goal_>0)*(goal_+4)   # N_envs x N_keys
            if self.mixed_fingering:
                finger_id = finger_id.to(torch.int64)
            

        if self.masked_fingering:
            # nearest finger
            dp = self.link_tensor[:, self.finger_tips_valid, None, :2] - self.piano_key_pos[:, :2]   # N_envs x N_fingers_valid x N_keys x 3
            finger_id_ = dp.square().sum(-1).min(1).indices          # N_envs x N_keys
            if self.right_hand_only:
                finger_id_ += 5
            if finger_id is None:
                finger_id = finger_id_
            else:
                m = goal_ == 7
                finger_id[m] = finger_id_[m]
        if self.masked_fingering_left:
            # nearest left finger
            n_fingers = len(self.finger_tips_valid)//2
            dp = self.link_tensor[:, self.finger_tips_valid[:n_fingers], None, 0] - self.piano_key_pos[:, 0]   # N_envs x N_fingers_valid x N_keys
            finger_id_ = dp.abs_().min(1).indices          # N_envs x N_keys

            if self.fingering_heuristic:
                m = goal_ < 0 # N_envs x N_keys
                m_ = m.unsqueeze(1) # N_envs x 1 x N_keys

                tot = torch.sum(m, -1) # N_envs
                tot_ = tot.clip(max=self.finger_schemes.size(0)-1)
                scheme = self.finger_schemes[tot_] # N_envs x N_schemes x (N_fingers+1)
                order = torch.cumsum(m, dim=-1) # N_envs x N_keys
                order_mask = order[:, 1:] != order[:, :-1]
                order[:, 1:] *= order_mask  # ..0, 1, 0...0, 2, 0...0, 3, 0...
                order.clip_(max=n_fingers)
                idx = self.arange_tensor_n_schemes + order.unsqueeze(1)
                n_envs = order.size(0)
                n_schemes = scheme.size(1)
                schemes = scheme.view(n_envs, -1).gather(1, idx.view(n_envs, -1)).view(n_envs, n_schemes, -1) # N_envs x N_schemes x N_keys
                
                dp = self.link_tensor[:, None, self.finger_tips_valid[:n_fingers], 0] - self.piano_key_pos[:, :1]   # N_envs x N_fingers_valid x N_keys
                dp.abs_() # N_envs x N_keys x N_fingers

                schemes -= 1 # change finger id to the index, from little to thumb, 0, ..., 4
                schemes.clip_(min=0)
                schemes_ = (schemes + self.arange_tensor_n_keys*n_fingers).view(n_envs, -1) # N_envs x (N_schemes x N_keys)
                dp = dp.view(n_envs, -1).gather(1, schemes_).view(n_envs, n_schemes, -1) # N_envs x N_schemes x N_keys

                first = torch.argmin(order + (order==0)*1000, -1) # N_envs
                last = torch.argmax(order, -1) # N_envs
                key_gap = (self.key_gap[last]-self.key_gap[first]).unsqueeze_(-1) # N_envs
                scheme_gap = self.scheme_gap[tot_] # N_envs x N_schemes
                invalid = (key_gap > scheme_gap) # N_envs x N_schemes

                use_thumb = schemes == 4
                thumb_not_right_most = ~self.thumb_last[tot_]
                # Thumb can be used to press the black keys only if 
                # (1) there is at most one white key needs to be pressed, or
                # (2) thumb is the right most
                invalid.logical_or_(
                    torch.any(use_thumb.logical_and(self.black_keys), -1                    # N_envs x N_schemes
                    ).logical_and_(m.logical_and(self.white_keys).sum(-1, keepdim=True) > 1 # N_envs x 1
                    ).logical_and_(thumb_not_right_most)                                    # N_envs x N_schemes
                )
                # Thumb can be used to press a key between other fingers only if 
                # (1) it is a white key press, and
                # (2) it is next to an index finger pressing, and
                # (3) index is on black
                use_index = schemes == 3
                key_use_thumb = use_thumb.to(torch.int32).argmax(-1) # N_envs x N_schemes
                key_use_index = use_index.to(torch.int32).argmax(-1)  # N_envs x N_schemes
                invalid.logical_or_(
                    torch.any(use_thumb, -1
                    ).logical_and_(thumb_not_right_most
                    ).logical_and_(m.sum(-1, keepdim=True) > 2
                    ).logical_and_(
                        torch.any(use_thumb.logical_and(self.white_keys), -1).logical_and_(
                        torch.any(use_index.logical_and(self.black_keys), -1)).logical_and_(
                        key_use_index - key_use_thumb == 1).logical_not_()
                    )
                )

                assigned = m.logical_and(goal_ != -6).unsqueeze_(1) # N_envs x N_keys
                g = (goal_+5).unsqueeze_(1) # N_envs x 1 x N_keys,  covert -5->0, ..., -1->4, to keep consistent to schemes
                invalid.logical_or_(torch.any((schemes != g).logical_and_(assigned), -1))

                idx = (dp*m_).sum(-1).add_(invalid*1000).argmin(1) # N_envs
                finger_id_heuristic = schemes[(self.arange_tensor_n_envs, idx)] # N_envs x N_keys

                finger_id_ = torch.where(tot.unsqueeze_(-1) > n_fingers, finger_id_, finger_id_heuristic)

            if finger_id is None:
                finger_id = finger_id_
            else:
                m = goal_ == -6
                finger_id[m] = finger_id_[m]

        if self.masked_fingering_right:
            # nearest right finger
            n_fingers = len(self.finger_tips_valid)//2
            dp = self.link_tensor[:, self.finger_tips_valid[n_fingers:], None, 0] - self.piano_key_pos[:, 0]
            finger_id_ = dp.abs_().min(1).indices+n_fingers

            if self.fingering_heuristic:
                m = goal_ > 0 # N_envs x N_keys
                m_ = m.unsqueeze(1) # N_envs x 1 x N_keys

                tot = torch.sum(m, -1) # N_envs
                tot_ = tot.clip(max=self.finger_schemes.size(0)-1)
                scheme = self.finger_schemes[tot_] # N_envs x N_schemes x (N_fingers+1)
                # reserve such that the little finger has id 1
                order = m+torch.sum(m, dim=-1, keepdim=True) - torch.cumsum(m, dim=-1) # N_envs x N_keys
                order_mask = order[:, 1:] != order[:, :-1]
                order[:, :-1] *= order_mask
                order.clip_(max=n_fingers)

                idx = self.arange_tensor_n_schemes + order.unsqueeze(1)
                n_envs = order.size(0)
                n_schemes = scheme.size(1)
                schemes = scheme.view(n_envs, -1).gather(1, idx.view(n_envs, -1)).view(n_envs, n_schemes, -1) # N_envs x N_schemes x N_keys
                
                dp = self.link_tensor[:, None, self.finger_tips_valid[n_fingers:], 0] - self.piano_key_pos[:, :1]   # N_envs x N_fingers_valid x N_keys
                dp.abs_() # N_envs x N_keys x N_fingers

                # change finger id to the index and reverse finger id, 1->4, ..., 5->0
                schemes = (n_fingers - schemes).clip_(max=n_fingers-1) # N_envs x N_schemes x N_keys
                
                schemes_ = (schemes + self.arange_tensor_n_keys*n_fingers).view(n_envs, -1) # N_envs x (N_schemes x N_keys)
                dp = dp.view(n_envs, -1).gather(1, schemes_).view(n_envs, n_schemes, -1) # N_envs x N_schemes x N_keys

                last = torch.argmin(order + (order==0)*1000, -1) # N_envs
                first = torch.argmax(order, -1) # N_envs
                key_gap = (self.key_gap[last]-self.key_gap[first]).unsqueeze_(-1) # N_envs
                scheme_gap = self.scheme_gap[tot_] # N_envs x N_schemes
                invalid = (key_gap > scheme_gap) # N_envs x N_schemes
                # Thumb can be used to press the black keys only if 
                # (1) there is at most one white key needs to be pressed, or
                # (2) thumb is the left most
                use_thumb = schemes == 0
                thumb_not_right_most = ~self.thumb_last[tot_]
                invalid.logical_or_(
                    torch.any(use_thumb.logical_and(self.black_keys), -1              # N_envs x N_schemes
                    ).logical_and_(m.logical_and(self.white_keys).sum(-1, keepdim=True) > 1 # N_envs x 1
                    ).logical_and_(thumb_not_right_most)                                  # N_envs x N_schemes
                )
                # Thumb can be used to press a key between other fingers only if 
                # (1) it is a white key press, and
                # (2) it is next to an index finger pressing, and
                # (3) index is on black
                use_index = schemes == 1
                key_use_thumb = use_thumb.to(torch.int32).argmax(-1) # N_envs x N_schemes
                key_use_index = use_index.to(torch.int32).argmax(-1)  # N_envs x N_schemes
                invalid.logical_or_(
                    torch.any(use_thumb, -1
                    ).logical_and_(thumb_not_right_most
                    ).logical_and_(m.sum(-1, keepdim=True) > 2
                    ).logical_and_(
                        torch.any(use_thumb.logical_and(self.white_keys), -1).logical_and_(
                        torch.any(use_index.logical_and(self.black_keys), -1)).logical_and_(
                        key_use_thumb - key_use_index == 1).logical_not_()
                    )
                )
                assigned = m.logical_and(goal_ != 6).unsqueeze_(1) # N_envs x N_keys
                g = (goal_-1).unsqueeze_(1) # N_envs x 1 x N_keys
                invalid.logical_or_(torch.any((schemes != g).logical_and_(assigned), -1))
                
                idx = (dp*m_).sum(-1).add_(invalid*1000).argmin(1) # N_envs
                finger_id_heuristic = schemes[(self.arange_tensor_n_envs, idx)]+n_fingers # N_envs x N_keys

                finger_id_ = torch.where(tot.unsqueeze_(-1) > n_fingers, finger_id_, finger_id_heuristic)

            if finger_id is None:
                finger_id = finger_id_
            else:
                m = goal_ == 6
                finger_id[m] = finger_id_[m]

        finger_pos = self.link_tensor[self.arange_tensor_n_envs.unsqueeze(-1), self.finger_tips[finger_id], :3] # N_envs x N_keys x 3
        dp = finger_pos - self.piano_key_pos                # N_envs x N_keys x 3

        dp[..., 1] *= ((dp[..., 1] < 0)*((0.1)-1)).add_(1)
        self.finger_id = finger_id
        self.finger_pos = finger_pos

        overkey = torch.all((finger_pos[..., :2] > self.piano_key_edge0
            ).logical_and_(finger_pos[..., :2] < self.piano_key_edge1),-1)  # N_envs x N_keys
            
        key_p = (key_qpos/self.piano_key_depth).clip_(min=0, max=1) # N_envs x N_keys

        self.key_target_finger = goal_
        goal = goal_ != 0                                    # N_envs x N_keys

        dp[..., 2] *= (dp[..., 2] < 0).logical_and_(overkey).logical_not_()

        dist2 = dp.square().sum(-1)                           # N_envs x N_keys
        dist = dist2.sqrt()
        rew_dist0 = (-500*dist2).exp_().mul_(0.8).add((-5*dist).exp_(), alpha=0.2)
 
        # t-1 t
        # ~p  ~p p
        # p   ~p 0
        # ~p   p p
        # p    p p
        # penalty for failure to keep pressing
        # NOTE the load_note function will separate note pressing to make there at least one frame gap for hand lifting up in concecutive pressing
        not_pressed_ = self.key_pos<=key_press_threshold
        pressed = key_p > key_release_threshold
        key_p_ = key_p * (not_pressed_).logical_or_(pressed)
        
        rew_press = (overkey*key_p_).pow_(3)
        rew_dist = (rew_dist0*0.6 + 0.4*rew_press)*goal
        rew = rew_dist.sum(-1).div_(goal.sum(-1)).nan_to_num_(nan=1.)
        rew_notar = -torch.sum(key_p.pow(6)*(~goal), 1)

        if self.two_hands and self.multi_objective_reward:
            left = (finger_id < 5).logical_and_(goal)
            right = finger_id > 4
            rew_left = (rew_dist*left).sum(-1).div_(left.sum(-1)).nan_to_num_(nan=1.)
            rew_right = (rew_dist*right).sum(-1).div_(right.sum(-1)).nan_to_num_(nan=1.)
            rew = torch.stack((rew_left, rew_right), -1) + 0.2*rew_notar.unsqueeze_(-1)
        else:
            rew = rew + 0.2*rew_notar
            rew.unsqueeze_(-1)

        self.key_pos = key_p

        key_activated = self.key_activated
        key_activated[key_p>key_press_threshold] = True
        key_activated[key_p<key_release_threshold] = False
        self.key_target = goal

        # frame based f1
        pressed_goal = torch.logical_and(key_activated, goal)
        tp = torch.sum(pressed_goal, -1)
        tp_fn = torch.sum(goal, -1)
        tp_fp = torch.sum(key_activated, -1)
        recall = (tp/tp_fn).nan_to_num_(nan=1)
        precision = (tp/tp_fp).nan_to_num_(nan=1)

        m = self.lifetime>self.grace_period
        # Logged as GPU scalars: the mean over the envs in m (NaN when m is
        # empty, as .mean() of an empty selection). Selecting recall[m] first
        # would need a host sync to size the result.
        self.info["log"]["recall"] = _masked_mean(recall, m)
        self.info["log"]["precision"] = _masked_mean(precision, m)
        if self.training:
            if self.random_init_note:
                if self.simulation_step == 0:
                    self.info["log"]["recall0"] = _masked_mean(recall, m)
                else:
                    self.info["log"]["recall0"] = _masked_mean(recall, m.logical_and(self.sampling_idx_current==0))

        self.info["key_activated"] = self.key_activated

        if self.importance_sampling and self.simulation_step:
            m = self.lifetime>self.grace_period
            if self.multi_objective_reward and self.two_hands:
                left = finger_id < 5
                tp_left = torch.sum(pressed_goal.logical_and(left), -1)
                tp_right = tp - tp_left
                tp_fn_left = torch.sum(goal.logical_and(left), -1)
                tp_fn_right = tp_fn - tp_fn_left
                recall_left = (tp_left/tp_fn_left).nan_to_num_(nan=1)
                recall_right = (tp_right/tp_fn_right).nan_to_num_(nan=1)
                f1 = torch.min(recall_left, recall_right)
            else:
                f1 = (recall * precision).mul_(2).div_(recall+precision).nan_to_num_(nan=0)
            self.cumulative_reward += self.cumulative_reward_discount*f1*m
            self.cumulative_reward_discount *= m*(self.importance_discount-1) + 1
                
        return rew
    
    def termination_check(self):
        finger_pos = self.link_tensor[:, self.finger_tips_valid, :3]
        too_far = torch.stack((
            finger_pos[:,:,0] < -0.7,  finger_pos[:,:,0] > 0.7,
            finger_pos[:,:,1] < -0.1,  finger_pos[:,:,1] > 0.5,
            finger_pos[:,:,2] < -0.3,  finger_pos[:,:,2] > 0.35
        ), -1)
        too_far = torch.any(too_far.flatten(start_dim=1), -1)
        return torch.logical_and(too_far, self.lifetime>3)

    def render(self):
        super().render()

        self.viewer.cam.distance = 1.1
        self._audio_key_state = self.key_activated[0].detach().cpu().numpy().copy()
        self._audio_key_position = self.key_pos[0].detach().cpu().numpy().copy()
        self._start_audio()


        scn = self.viewer.user_scn
        self.tracker_geom_idx = scn.ngeom
        for _ in range(len(self.piano_key_pos)*2+1):
            for c in [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW, size=[0.01, 0.01, 0.01],
                    pos=[0, 0, 100], mat=[1, 0, 0, 0, 1, 0, 0, 0, 1], rgba=c
                )
                scn.ngeom += 1

    def update_viewer(self):
        super().update_viewer()

        from matplotlib import colormaps
        cm = colormaps["autumn"]
        if self.muscle_control: self.mj_model.tendon_rgba = cm(self.mj_data.act)
            
        goal = self.key_target[0].cpu().numpy()
        key_activated = self.key_activated[0].cpu().numpy()
        self._play_audio_for_pressed_keys(key_activated)

        for i, (g, k) in enumerate(zip(goal, key_activated)):
            geom = self.mj_model.geom(self.piano_key0_geom+i)
            if g:
                if k:
                    geom.rgba[0] = 0
                    geom.rgba[1] = 1
                    geom.rgba[2] = 0
                else:
                    geom.rgba[0] = 1
                    geom.rgba[1] = 1
                    geom.rgba[2] = 0
            else:
                if k:
                    geom.rgba[0] = 1
                    geom.rgba[1] = 0
                    geom.rgba[2] = 0
                elif "black" in geom.name:
                    geom.rgba[:3] = 0.1
                else:
                    geom.rgba[:3] = 0.9

        scn = self.viewer.user_scn
        pos = self.piano_key_pos.cpu().numpy()
        mat = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        rot_mat = mat.flatten()
        finger_pos = self.finger_pos[0].cpu().numpy()

        for i, g in enumerate(goal):
            g = g != 0
            p = pos[i] * g
            for j, c in enumerate([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]):
                mujoco.mjv_initGeom(scn.geoms[self.tracker_geom_idx+j+i*3],
                    type=mujoco.mjtGeom.mjGEOM_ARROW, size=[0.002, 0.002, 0.002],
                    pos=p, mat=rot_mat, rgba=c
                )
                mujoco.mjv_connector(scn.geoms[self.tracker_geom_idx+j+i*3],
                    type=mujoco.mjtGeom.mjGEOM_ARROW, width=0.001,
                    from_=p, to=p + (0.02*(mat*c[:-1])[:,j])*g
                )
            p = finger_pos[i]*g
            for j, c in enumerate([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]):
                mujoco.mjv_initGeom(scn.geoms[self.tracker_geom_idx+len(self.piano_key_pos)*3+j+i*3],
                    type=mujoco.mjtGeom.mjGEOM_ARROW, size=[0.002, 0.002, 0.002],
                    pos=p, mat=rot_mat, rgba=c
                )
                mujoco.mjv_connector(scn.geoms[self.tracker_geom_idx+len(self.piano_key_pos)*3+j+i*3],
                    type=mujoco.mjtGeom.mjGEOM_ARROW, width=0.001,
                    from_=p, to=p + (0.02*(mat*c[:-1])[:,j])*g
                )

class PianoJointPD(PianoBase):

    def reset(self):
        if self.training:
            self.track_t_tensor.clip_(max=self.max_note_t2*2) # clip long notes
        super().reset()

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if type(actions) == tuple: actions = torch.cat(actions, -1)
        return super().process_actions(actions)


from utils import quat2axang


class MuscleTracking(ICCGANHumanoid):
    GOAL_REWARD_WEIGHT = 1
    OB_HORIZON = 1

    def __init__(self, *args, **kwargs):
        self.random_init = kwargs.get("random_init", False)
        self.control_freq = kwargs.get("control_freq", 8)
        self.future_horizon = kwargs.get("future_horizon", 4)

        self.terminate_with_control_freq = kwargs.get("terminate_with_control_freq", False)

        kwargs["ob_horizon"] = 1

        self.key_link_weights_orient = kwargs["key_link_weights_orient"]
        self.key_link_weights_pos = kwargs["key_link_weights_pos"]
        assert all([_ >= 0 for _ in self.key_link_weights_orient.values()])
        assert all([_ >= 0 for _ in self.key_link_weights_pos.values()])

        class DummyRefMotion():
            def __init__(self, env):
                self.env = env
            def state(self, motion_ids, motion_times, with_joint_tensor=False):
                if with_joint_tensor:
                    q = self.env.sim.state("qpos")[:, 7:]
                    return self.env.link_tensor[:, :self.env.n_char_links if self.env.has_piano else None], (q, q)
                else:
                    return self.env.link_tensor[:, :self.env.n_char_links if self.env.has_piano else None]
        self.ref_motion = DummyRefMotion(self)

        self.importance_sampling = False
        super().__init__(*args, **kwargs)

        importance_sampling = kwargs.get("importance_sampling", -1)
        self.importance_discount = kwargs.get("importance_discount", 0.99)
        self.importance_scale = kwargs.get("importance_scale", 6)
        self.importance_decay = kwargs.get("importance_decay", 0.5)
        self.importance_sampling = importance_sampling > -1
        if self.importance_sampling:
            n_clips = self.ref_motion.motion_tensor_offset.size(0)
            n_ref_frames = self.ref_motion.motion_link_pos_tensor.size(0)

            if importance_sampling:
                tail = int(np.ceil(n_ref_frames / (self.n_envs*importance_sampling+n_clips)))
            else:
                tail = 0
            samples_ = []
            for start, dt, length in zip(self.ref_motion.motion_tensor_offset, self.ref_motion.motion_dt_tensor, self.ref_motion.motion_n_frames_tensor):
                n = max(1, length.item()-max(tail, int((0.5/dt).item())))
                samples_.append((start.item(), start.item()+n))

            tot = sum(e-s for s, e in samples_)
            if importance_sampling:
                gap = max(1, tot//(self.n_envs*importance_sampling))
            else:
                gap = 1
            self.sampling_gap = int(gap)

            samples = [np.arange(s, e, self.sampling_gap) for s, e in samples_]
            tail = []
            for b, (s, e) in zip(samples, samples_):
                if e-b[-1] > self.sampling_gap/2:
                    tail.append(e)
            samples.append(tail)
            samples = np.concatenate(samples)
            samples = np.sort(samples)
            print(samples, samples.shape, np.max(samples[1:]-samples[:-1]), self.sampling_gap, n_ref_frames-samples[-1])

            self.tracking_motion_samples = torch.tensor(samples, device=self.device, dtype=torch.int64)

            self.average_reward = 0.1*torch.ones(self.tracking_motion_samples.shape, dtype=torch.float, device=self.device)
            print("SAMPLE", self.average_reward.size(0), self.sampling_gap)

            self.cumulative_reward = torch.empty((self.n_envs,), dtype=torch.float, device=self.device)
            self.cumulative_reward_discount = torch.empty((self.n_envs,), dtype=torch.float, device=self.device)
            self.sampling_idx = torch.empty((self.n_envs,), dtype=torch.long, device=self.device)

            self.cumulative_reward_buffer = torch.empty_like(self.average_reward)
            self.ref_motion.motion_tensor_id = torch.empty((n_ref_frames, ), dtype=torch.int, device=self.device)
            for idx, offset in enumerate(self.ref_motion.motion_tensor_offset):
                self.ref_motion.motion_tensor_id[offset:] = idx

            if self.episode_length is not None:
                max_steps = (self.episode_length+1)//self.control_freq
            else:  # no hard limit: the longest episode plays a whole reference motion
                max_steps = (int(np.max(self.ref_motion.motion_length) / self.step_time) + 2)//self.control_freq + 1
            boostrap_reward = np.flip(np.cumsum(np.flip([self.importance_discount**i for i in range(max_steps)]+[0])))
            self.boostrap_reward = torch.tensor(np.repeat(boostrap_reward, self.control_freq), dtype=torch.float, device=self.device)
            self.max_rew = self.boostrap_reward[0].item()

    def get_goal_dim(self):
        return self.n_char_links*7*self.future_horizon

    def prepare_sim(self, opt={}):
        opt["disableflags"] = int(mujoco.mjtDisableBit.mjDSBL_GRAVITY) | int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        opt["gravity"] = np.array([0, 0, 0])
        super().prepare_sim(opt)
        self.tracking_motion_ids = torch.zeros((self.n_envs), dtype=torch.int, device=self.device)
        self.tracking_motion_times = torch.zeros((self.n_envs), dtype=torch.float, device=self.device)
        self.tracking_motion_length = torch.zeros((self.n_envs), dtype=torch.long, device=self.device)

        self.up_dir_tensor = torch.zeros((self.n_envs, 1, 3), dtype=torch.float, device=self.device)
        self.up_dir_tensor[..., 2] = 1

        self.arange_tensor_n_envs = torch.arange(self.n_envs, device=self.device)
        self.origin = torch.empty((self.n_envs, 1, 3), dtype=torch.float, device=self.device)
        self.orient = torch.empty((self.n_envs, 1, 4), dtype=torch.float, device=self.device)

        self.has_piano = any("P:" in self.mj_model.body(i).name for i in range(self.mj_model.nbody))
        if self.has_piano:
            self.n_char_links = min(i for i in range(self.mj_model.nbody) if "P:" in self.mj_model.body(i).name) - 1
        else:
            self.n_char_links = self.mj_model.nbody - 1

        ww = np.zeros(self.n_char_links)
        if self.key_link_weights_orient:
            for link, w in self.key_link_weights_orient.items():
                lid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link)-1
                assert lid > -1, "Unrecognized link {}".format(link)
                ww[lid] = w
            ww /= np.sum(ww)
        self.key_link_weights_orient = torch.tensor(np.nan_to_num(ww), dtype=torch.float, device=self.device)
        ww = np.zeros(self.n_char_links)
        if self.key_link_weights_pos:
            for link, w in self.key_link_weights_pos.items():
                lid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link)-1
                assert lid > -1, "Unrecognized link {}".format(link)
                ww[lid] = w
            ww /= np.sum(ww)
        self.key_link_weights_pos = torch.tensor(np.nan_to_num(ww), dtype=torch.float, device=self.device)
        if self.future_horizon > 1:
            self.tracking_target = torch.empty((self.n_envs, self.future_horizon, self.n_char_links, 13), dtype=torch.float, device=self.device)

        self.record = []

    def init_state(self, env_ids):
        if self.training and self.importance_sampling:
            if self.simulation_step > 0:
                sid = self.sampling_idx[env_ids]
                idx, cnts = torch.unique(sid, return_counts=True)
                self.cumulative_reward[env_ids] += (~self.info["terminate"][env_ids])*self.boostrap_reward[self.lifetime[env_ids]]
                self.cumulative_reward_buffer.index_fill_(0, idx, 0)
                self.cumulative_reward_buffer.index_put_((sid,), self.cumulative_reward[env_ids], accumulate=True)
                self.average_reward[idx] *= 1-self.importance_decay
                self.average_reward[idx] += (self.cumulative_reward_buffer[idx]/cnts).mul_(self.importance_decay)

            weights = self.max_rew/self.average_reward
            weights.pow_(self.importance_scale)
            weights.clip_(min=1e-6, max=1e20).nan_to_num_(neginf=None, posinf=1e20, nan=None)

            sid = torch.multinomial(weights, len(env_ids), replacement=True)
            frac = torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=(len(env_ids),))).to(device=self.device, dtype=torch.float)

            if self.sampling_gap > 1:
                sample = self.tracking_motion_samples[sid] + self.sampling_gap*frac
                fid = sample.to(torch.int64)
                frac = sample - fid
            else:
                fid = self.tracking_motion_samples[sid]

            self.sampling_idx[env_ids] = sid.to(torch.long)
            self.cumulative_reward.index_fill_(0, env_ids, 0)
            self.cumulative_reward_discount.index_fill_(0, env_ids, 1)

            motion_ids = self.ref_motion.motion_tensor_id[fid]
            motion_times = ((fid - self.ref_motion.motion_tensor_offset[motion_ids])+frac)*self.ref_motion.motion_dt_tensor[motion_ids]
            self.tracking_motion_ids[env_ids] = motion_ids
            self.tracking_motion_times[env_ids] = motion_times
            if self.control_freq > 1 and self.terminate_with_control_freq:
                self.tracking_motion_length[env_ids] = (((self.ref_motion.motion_length_tensor[motion_ids]-motion_times) // (self.step_time*self.control_freq))*self.control_freq).to(torch.long)
            else:
                self.tracking_motion_length[env_ids] = ((self.ref_motion.motion_length_tensor[motion_ids]-motion_times) // self.step_time).to(torch.long)
        else:
            motion_ids, motion_times = self.ref_motion.sample(len(env_ids), truncate_time=0.5)
            if not self.random_init:
                motion_times[:] = 0
            if not self.training:
                motion_times[:] = 0

            self.tracking_motion_ids[env_ids] = torch.tensor(motion_ids, dtype=torch.int, device=self.device)
            self.tracking_motion_times[env_ids] = torch.tensor(motion_times, dtype=torch.float, device=self.device)
            if self.control_freq > 1 and self.terminate_with_control_freq:
                self.tracking_motion_length[env_ids] = torch.tensor(((self.ref_motion.motion_length[motion_ids]-motion_times) // (self.step_time*self.control_freq))*self.control_freq,
                                                                    dtype=torch.long, device=self.device)
            else:
                self.tracking_motion_length[env_ids] = torch.tensor((self.ref_motion.motion_length[motion_ids]-motion_times) // self.step_time,
                                                                    dtype=torch.long, device=self.device)

        ref_link_tensor, ref_joint_tensor = \
            self.ref_motion.state(motion_ids, motion_times, with_joint_tensor=True)

        if self.future_horizon > 1:
            self.update_tracking_target(env_ids, n=self.future_horizon-1)
        return ref_link_tensor, ref_joint_tensor

    def update_tracking_target(self, env_ids, n=1):
        dt = self.step_time*self.control_freq
        if env_ids is None or len(env_ids) == self.n_envs:
            if n > 1:
                for i in range(n):
                    self.tracking_motion_times += dt
                    target_link_tensor = \
                        self.ref_motion.state(self.tracking_motion_ids, self.tracking_motion_times, with_joint_tensor=False)
                    self.tracking_target[:, i-n] = target_link_tensor
            else:
                self.tracking_motion_times += dt
                target_link_tensor = \
                    self.ref_motion.state(self.tracking_motion_ids, self.tracking_motion_times, with_joint_tensor=False)
                if self.future_horizon > 1:
                    self.tracking_target[:, :-1] = self.tracking_target[:, 1:].clone()
                    self.tracking_target[:, -1] = target_link_tensor
                else:
                    self.tracking_target = target_link_tensor.unsqueeze_(1)
        else:
            env_ids_ = env_ids.cpu().numpy()
            if n > 1:
                motion_ids = self.tracking_motion_ids[env_ids_]
                for i in range(n):
                    self.tracking_motion_times[env_ids] += dt
                    target_link_tensor = \
                        self.ref_motion.state(motion_ids, self.tracking_motion_times[env_ids], with_joint_tensor=False)
                    self.tracking_target[env_ids, i-n] = target_link_tensor
            else:
                self.tracking_motion_times[env_ids] += dt
                target_link_tensor = \
                    self.ref_motion.state(self.tracking_motion_ids[env_ids_], self.tracking_motion_times[env_ids_], with_joint_tensor=False)
                if self.future_horizon > 1:
                    self.tracking_target[env_ids, :-1] = self.tracking_target[env_ids, 1:].clone()
                self.tracking_target[env_ids, -1] = target_link_tensor


    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        a = actions[:,:-6]
        self.act = torch.abs(a)
        return torch.cat((a,
            actions[:, -6:]*self.action_scale[-6:] + self.action_offset[-6:]), -1)

    def _observe(self, env_ids):
        if env_ids is None:
            n_envs = self.n_envs
            if self.control_freq > 1:
                counter = self.lifetime % self.control_freq
                env_ids_ = self.arange_tensor_n_envs[counter == 0]
                if len(env_ids_):
                    self.update_tracking_target(env_ids_)
                    self.origin[env_ids_] = self.link_tensor[env_ids_, :1, :3]
                    self.orient[env_ids_] = self.link_tensor[env_ids_, :1, 3:7]
            else:
                self.update_tracking_target(env_ids)
                self.origin.copy_(self.link_tensor[:, :1, :3])
                self.orient.copy_(self.link_tensor[:, :1, 3:7])
            target_link_tensor = self.tracking_target
            link_tensor = self.link_tensor
        else:
            n_envs = len(env_ids)
            if self.control_freq > 1:
                counter = self.lifetime[env_ids] % self.control_freq
                env_ids_ = env_ids[counter == 0]
                if len(env_ids_):
                    self.update_tracking_target(env_ids_)
                    self.origin[env_ids_] = self.link_tensor[env_ids_, :1, :3]
                    self.orient[env_ids_] = self.link_tensor[env_ids_, :1, 3:7]
            else:
                self.update_tracking_target(env_ids)
                self.origin[env_ids] = self.link_tensor[env_ids, :1, :3]
                self.orient[env_ids] = self.link_tensor[env_ids, :1, 3:7]

            target_link_tensor = self.tracking_target[env_ids]
            link_tensor = self.link_tensor[env_ids]

        if env_ids is None:
            origin = self.origin
            orient = self.orient
        else:
            origin = self.origin[env_ids]
            orient = self.orient[env_ids]

        orient_inv = quatconj(orient)
        target_link_tensor = target_link_tensor.view(n_envs, -1, target_link_tensor.size(-1))
        p_ = rotatepoint(orient_inv, target_link_tensor[:,:,:3]-origin).view(n_envs, -1)
        q_ = quatmultiply(orient_inv, target_link_tensor[:,:,3:7]).view(n_envs, -1)

        n_char = self.n_char_links if self.has_piano else None
        p = rotatepoint(orient_inv, link_tensor[:, :n_char, :3]-origin).view(n_envs, -1)
        q = quatmultiply(orient_inv, link_tensor[:, :n_char, 3:7]).view(n_envs, -1)
        v1 = rotatepoint(orient_inv, link_tensor[:, :n_char, 7:10]).view(n_envs, -1)
        v2 = rotatepoint(orient_inv, link_tensor[:, :n_char, 10:13]).view(n_envs, -1)

        obs = [p, q, v1, v2]

        act_t = self.sim.state("act")
        ten_length_t = self.sim.state("ten_length")
        ten_velocity_t = self.sim.state("ten_velocity")
        if env_ids is None:
            obs.append(act_t)
            obs.append(ten_length_t)
            obs.append(ten_velocity_t)
        else:
            obs.append(act_t[env_ids])
            obs.append(ten_length_t[env_ids])
            obs.append(ten_velocity_t[env_ids])

        if self.control_freq > 1:
            obs.append(counter.unsqueeze_(-1).to(torch.float))

        obs.append(p_)
        obs.append(q_)
        ob = torch.cat(obs, -1)
        return ob

    def reward(self):
        target_orient = self.tracking_target[:, 0, :, 3:7]
        target_pos = self.tracking_target[:, 0, :, :3]
        n_char = self.n_char_links if self.has_piano else None
        link_orient = self.link_tensor[:, :n_char, 3:7]
        link_pos = self.link_tensor[:, :n_char, :3]
        _, e_a = quat2axang(quatmultiply(link_orient, quatconj(target_orient)))
        e_p = (link_pos - target_pos).square_().sum(-1)

        eo = (e_a.square_() * self.key_link_weights_orient).sum(-1)
        ep = (e_p.sqrt_() * self.key_link_weights_pos).sum(-1)
        ro = eo.mul(-3).exp_()
        rp = ep.mul(-50).exp_().mul_(0.7).add_(ep.mul(-3).exp_(), alpha=0.3)
        rew = 0.5 * ro + 0.5 * rp
        self.tracking_error = e_p

        if self.simulation_step > 0:
            a = torch.pow(self.act.abs(), 4).mean(-1)
            rew *= 0.9
            rew += 0.1*torch.exp(-a)

        if self.training:
            # per-link mean over the envs at a control step, as GPU scalars (see PianoBase.reward)
            m = self.lifetime%self.control_freq == 0
            for name, link in (("wrist", 2), ("thumb_tip", 11), ("index_tip", 18),
                               ("middle_tip", 23), ("ring_tip", 28), ("pinky_tip", 33)):
                self.info["log"][name] = _masked_mean(self.tracking_error[:, link], m)

        term = (self.lifetime>self.control_freq*4).logical_and_(torch.any(self.tracking_error > 0.5, -1))
        if self.control_freq > 1:
            m = self.lifetime%self.control_freq == 0
            term.logical_and_(m)
        if not self.terminate_with_control_freq:
            term.logical_or_(torch.any(self.tracking_error > 1, -1))
        self.terminate = term

        if self.importance_sampling and self.simulation_step > 0:
            if self.control_freq > 1:
                if torch.any(m).item():
                    self.cumulative_reward[m] += self.cumulative_reward_discount[m]*rew[m]*(~term[m])
                    self.cumulative_reward_discount[m] *= self.importance_discount
            else:
                self.cumulative_reward += self.cumulative_reward_discount*rew*(~term)
                self.cumulative_reward_discount *= self.importance_discount

        return rew.unsqueeze_(-1)

    def overtime_check(self):
        over = super().overtime_check()
        over.logical_or_(self.lifetime >= self.tracking_motion_length)
        return over

    def termination_check(self):
        return self.terminate

    def render(self):
        super().render()

        self.viewer.cam.azimuth = 108.125
        self.viewer.cam.distance = 0.574599963630366
        self.viewer.cam.elevation = -15.25
        self.viewer.cam.lookat = [0.03719079, 0.04026999, 0.10582043]

        self.tracker_link_idx = torch.nonzero(torch.logical_or(self.key_link_weights_orient>0, self.key_link_weights_pos>0)).cpu().numpy().flatten()
        scn = self.viewer.user_scn
        self.tracker_geom_idx = scn.ngeom
        for _ in range(len(self.tracker_link_idx)):
            for c in [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]:
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=[0.01, 0.01, 0.01],
                    pos=[0, 0, 0],
                    mat=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                    rgba=c,
                )
                scn.ngeom += 1


    def update_viewer(self):
        if self.lifetime[0].item() % self.control_freq != 0:
            return

        super().update_viewer()

        from matplotlib import colormaps
        cm = colormaps["autumn"]
        if self.muscle_control:
            self.mj_model.tendon_rgba = cm(self.mj_data.act)
        else:
            self.mj_model.tendon_rgba = cm(self.mj_data.actuator_force[:-6]/self.mj_model.actuator_forcerange[:-6,1])

        from scipy.spatial.transform import Rotation as R
        scn = self.viewer.user_scn
        pos = self.tracking_target[0, 0, :, :3].cpu().numpy()
        orient = self.tracking_target[0, 0, :, 3:7].cpu().numpy()
        for j, idx in enumerate(self.tracker_link_idx):
            p = pos[idx]
            o = orient[idx]
            mat = R.from_quat(o).as_matrix()
            for i, c in enumerate([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]):
                mujoco.mjv_initGeom(
                    scn.geoms[self.tracker_geom_idx+i+j*3],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=[0.002, 0.002, 0.002],
                    pos=p,
                    mat=mat.flatten(),
                    rgba=c,
                )
                mujoco.mjv_connector(
                    scn.geoms[self.tracker_geom_idx+i+j*3],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    width=0.001,
                    from_=p,
                    to=p + 0.02*(mat*c[:-1])[:,i],
                )

class PianoMuscleDriven(PianoBase):

    def __init__(self, *args, **kwargs):
        self.controller_steps =  kwargs["control_freq"]
        limit = episode_limit(kwargs.get("episode_length"))
        kwargs["episode_length"] = None if limit is None else limit//(kwargs["frameskip"]*kwargs["control_freq"])
        kwargs["frameskip"] = int(kwargs["frameskip"] * kwargs["control_freq"])
        kwargs["fps"] = kwargs["fps"] / self.controller_steps
        super().__init__(*args, **kwargs)
        self.frameskip //= self.controller_steps

        control_policy = kwargs.get("control_policy")
        if self.two_hands:
            assert len(control_policy) == 2
            controller1, act_dim1 = self.load_controller(control_policy[0])
            controller2, act_dim2 = self.load_controller(control_policy[1])
            assert controller1.future_goal_horizon == controller2.future_goal_horizon
            self.controller = torch.nn.ModuleList([controller1, controller2])
            self.controller.future_goal_horizon = controller1.future_goal_horizon
            self.act_dim = act_dim1 + act_dim2
        else:
            if type(control_policy) == list or type(control_policy) == tuple:
                control_policy = control_policy[0]
            self.controller, self.act_dim = self.load_controller(control_policy)
        self.controller.to(self.device)
        self.controller.eval()

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def do_simulation(self, actions):
        ctrl = self.sim.state("ctrl")
        if self.two_hands:
            if type(actions) == tuple:
                actions1, actions2 = actions
            else:
                act_dim_by_2 = actions.size(-1)//2
                actions1 = actions[:, :act_dim_by_2]
                actions2 = actions[:, act_dim_by_2:]
            controller1 = self.controller[0]
            controller2 = self.controller[1]
        else:
            controller = self.controller

        for substep in range(self.controller_steps):
            if substep > 0:
                self.refresh_tensors()

            if self.two_hands:
                s = self.controller_observe(substep)
                a1 = controller1(s[0], actions1)
                a2 = controller2(s[1], actions2)
                a = torch.cat((a1, a2), -1)
                a.mul_(self.action_scale).add_(self.action_offset)
            else:
                s = self.controller_observe(substep)
                a = controller(s, actions)
                a.mul_(self.action_scale).add_(self.action_offset)
            ctrl.copy_(a)
            self.sim.step(self.frameskip)

        self.simulation_step += 1


    def load_controller(self, control_policy):
        from models import ACModel, RunningMeanStd
        ckpt = torch.load(control_policy, map_location="cpu", weights_only=False)["model"]
        print("load control policy from", control_policy)
        use_rnn = any("rnn" in _ for _ in ckpt.keys())
        if use_rnn: raise NotImplementedError()

        latent_dim = int(ckpt["actor.embed_goal.4.bias"].shape[0])
        state_dim = int(ckpt["actor.mlp.0.weight"].shape[1]) - latent_dim
        act_dim = int(ckpt["actor.mu.bias"].shape[0])
        actor_goal_dim = int(ckpt["actor.embed_goal.0.weight"].shape[1])
        # critic_goal_dim = int(ckpt["critic.mlp.0.weight"].shape[1])-state_dim
        # assert actor_goal_dim == critic_goal_dim

        # tracking controller
        class Controller(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = ACModel.Actor(use_rnn, state_dim+latent_dim, act_dim, goal_dim=0, max_sigma=0.5, normalize_latent=True)
                self.actor.load_state_dict({k[len("actor."):]:v for k, v in ckpt.items() if "actor." in k and "goal" not in k})
                self.normalizer = RunningMeanStd(state_dim, scale=3)
                self.normalizer.mean.copy_(ckpt["ob_normalizer.mean"][:state_dim])
                self.normalizer.var.copy_(ckpt["ob_normalizer.var"][:state_dim])
            @torch.no_grad
            def forward(self, s, z, x=None):
                if x is None:
                    h = z
                elif z is None:
                    h = x
                else:
                    h = x + z
                h = h/h.norm(p=2,dim=-1,keepdim=True)
                s = self.normalizer(s)
                s = s.view(s.size(0), -1)
                h.nan_to_num_(nan=(1/h.size(-1))**0.5)
                pi = self.actor(torch.cat((s, h), -1))
                return pi.mean

        controller = Controller()

        embed_goal = torch.nn.Sequential(
            torch.nn.Linear(actor_goal_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, latent_dim)
        )
        embed_goal.load_state_dict({k[len("actor.embed_goal."):]:v for k, v in ckpt.items() if "actor.embed_goal." in k})
        goal_normalizer = RunningMeanStd(actor_goal_dim, scale=3)
        goal_normalizer.mean.copy_(ckpt["ob_normalizer.mean"][state_dim:])
        goal_normalizer.var.copy_(ckpt["ob_normalizer.var"][state_dim:])

        act_dim = latent_dim
        class EmbedGoal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_goal = embed_goal
                self.goal_normalizer = goal_normalizer
            def forward(self, g):
                h = self.embed_goal(self.goal_normalizer(g))
                h = h/h.norm(p=2,dim=-1,keepdim=True)
                return h.nan_to_num_(nan=(1/h.size(-1))**0.5)
        controller.embed_goal = EmbedGoal()

        if self.two_hands:
            controller.future_goal_horizon = goal_normalizer.mean.size(0)//(self.n_char_links//2*7)
        else:
            controller.future_goal_horizon = goal_normalizer.mean.size(0)//(self.n_char_links*7)
        return controller, act_dim

    def controller_observe(self, counter):
        n_envs = self.n_envs
        link_tensor = self.link_tensor

        if self.two_hands:
            idx2 = self.n_char_links//2
            if torch.is_tensor(counter):
                m = counter == 0
                self.origin2[m] = link_tensor[m, [0,idx2], None, :3]
                self.orient_inv2[m] = quatconj(link_tensor[m, [0,idx2], None, 3:7])
            elif counter == 0:
                self.origin2 = link_tensor[:, [0,idx2], None, :3]
                self.orient_inv2 = quatconj(link_tensor[:, [0,idx2], None, 3:7])
            origin = self.origin2
            orient_inv = self.orient_inv2
        else:
            # All substeps for muscle control share the same origin
            if torch.is_tensor(counter):
                m = counter == 0
                self.origin[m] = link_tensor[m, :1, :3]
                self.orient_inv[m] = quatconj(link_tensor[m, :1, 3:7])
            elif counter == 0:
                self.origin = link_tensor[:, :1, :3]
                self.orient_inv = quatconj(link_tensor[:, :1, 3:7])
            origin = self.origin
            orient_inv = self.orient_inv

        if torch.is_tensor(counter):
            counter = counter.view(n_envs, 1)
        else:
            counter = torch.full((n_envs,1), counter, dtype=torch.float, device=self.device)

        n_char_links = self.n_char_links
        act = self.sim.state("act")
        ten_length = self.sim.state("ten_length")
        ten_velocity = self.sim.state("ten_velocity")

        if self.two_hands:
            act = act.view(n_envs, 2, -1)
            ten_length = ten_length.view(n_envs, 2, -1)
            ten_velocity = ten_velocity.view(n_envs, 2, -1)
            p = link_tensor[:,:(n_char_links if self.has_piano else None),:3]
            q = link_tensor[:,:(n_char_links if self.has_piano else None),3:7]
            v1 = link_tensor[:,:(n_char_links if self.has_piano else None),7:10]
            v2 = link_tensor[:,:(n_char_links if self.has_piano else None),10:13]
            p = rotatepoint(orient_inv, p.view(n_envs, 2, -1, 3) - origin).view(n_envs, 2, -1)
            q = quatmultiply(orient_inv, q.view(n_envs, 2, -1, 4)).view(n_envs, 2, -1)
            v1 = rotatepoint(orient_inv, v1.view(n_envs, 2, -1, 3)).view(n_envs, 2, -1)
            v2 = rotatepoint(orient_inv, v2.view(n_envs, 2, -1, 3)).view(n_envs, 2, -1)
            obs1 = [p[:, 0], q[:, 0], v1[:, 0], v2[:, 0]]
            obs2 = [p[:, 1], q[:, 1], v1[:, 1], v2[:, 1]]
            obs1.append(act[:, 0])
            obs1.append(ten_length[:, 0])
            obs1.append(ten_velocity[:, 0])
            obs2.append(act[:, 1])
            obs2.append(ten_length[:, 1])
            obs2.append(ten_velocity[:, 1])
            obs1.append(counter)
            obs2.append(counter)
            obs1 = torch.cat(obs1, -1)
            obs2 = torch.cat(obs2, -1)
            s = obs1, obs2
        else:
            n_char_links = 34
            p = rotatepoint(orient_inv, link_tensor[:,:(n_char_links if self.has_piano else None),:3]-origin).view(n_envs, -1)
            q = quatmultiply(orient_inv, link_tensor[:,:(n_char_links if self.has_piano else None),3:7]).view(n_envs, -1)
            v1 = rotatepoint(orient_inv, link_tensor[:,:(n_char_links if self.has_piano else None),7:10]).view(n_envs, -1)
            v2 = rotatepoint(orient_inv, link_tensor[:,:(n_char_links if self.has_piano else None),10:13]).view(n_envs, -1)
            obs = [p, q, v1, v2]
            obs.append(act)
            obs.append(ten_length)
            obs.append(ten_velocity)
            obs.append(counter)
            s = torch.cat(obs, -1)

        return s


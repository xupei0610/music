"""Simulators behind the env classes.

The tasks in env.py read and write the batched simulation state only through
a Simulator, so the same task runs on mujoco_warp (GPU, many worlds; used for
training) or on native MuJoCo (CPU; used for testing, the viewer and audio).

State is exposed as torch tensors of shape (n_envs, ...) on `Simulator.device`
by `state(name)`. The tensor for a name is the same object on every call, and
writes to the writable fields take effect at the next step() or
refresh_kinematics():
  writable:  qpos, qvel, act, ctrl, qacc_warmstart
  read-only: xpos, xquat (w, x, y, z), cvel, ten_length, ten_velocity
             (valid after refresh_kinematics())
"""

WRITABLE = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
DERIVED = ("xpos", "xquat", "cvel", "ten_length", "ten_velocity")
BACKENDS = ("warp", "mujoco")


class Simulator:
    """Interface. See warp_sim.WarpSimulator and mujoco_sim.MujocoSimulator."""
    n_envs: int
    device: "torch.device"

    def state(self, name):
        """(n_envs, ...) tensor of `name`; see the module docstring."""
        raise NotImplementedError

    def step(self, n_substeps):
        """Advance every world by n_substeps physics steps with the current ctrl."""
        raise NotImplementedError

    def refresh_kinematics(self):
        """Recompute the derived fields from qpos/qvel after a step or a reset."""
        raise NotImplementedError

    def contacts(self):
        """(geom (N, 2) int, dist (N,) float, worldid (N,) int, valid (N,) bool) on device."""
        raise NotImplementedError

    def copy_to_mjdata(self, mj_data, world=0):
        """Copy one world into an MjData of the same model, e.g. for the viewer."""
        raise NotImplementedError


def make_simulator(backend, mj_model, mj_data, n_envs, compute_device, **kwargs):
    """Build the simulator for `backend` from a compiled model and its initial data.

    kwargs (warp only): nconmax, njmax (per-world buffer sizes), cuda_graph.
    """
    if backend == "warp":
        from sim.warp_sim import WarpSimulator
        return WarpSimulator(mj_model, mj_data, n_envs, compute_device, **kwargs)
    if backend == "mujoco":
        from sim.mujoco_sim import MujocoSimulator
        return MujocoSimulator(mj_model, mj_data, n_envs)
    raise ValueError("unknown simulator backend {!r}; expected one of {}".format(backend, BACKENDS))


def backend_device(backend, compute_device):
    """The torch device the env uses for `backend` (native MuJoCo runs on the CPU)."""
    import torch
    if backend == "mujoco" or compute_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda", compute_device)

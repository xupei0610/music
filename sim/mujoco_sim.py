"""Native MuJoCo simulator: one MjData per world, stepped on the CPU.

For tests, the viewer and real-time audio with few worlds. MuJoCo computes in
float64, and the task code works in float32, so the state is mirrored in
float32 torch tensors. Before a step or a kinematics refresh, a writable field
is copied into MjData only if the task changed it since the last copy back.
So the float64 state is not rounded to float32 on every step.
"""
import mujoco
import numpy as np
import torch

from sim import Simulator, WRITABLE, DERIVED


class MujocoSimulator(Simulator):

    def __init__(self, mj_model, mj_data, n_envs):
        self.mj_model = mj_model
        self.n_envs = n_envs
        self.device = torch.device("cpu")
        self.datas = [mujoco.MjData(mj_model) for _ in range(n_envs)]
        for d in self.datas:
            mujoco.mj_copyData(d, mj_model, mj_data)
        self._state = {k: torch.zeros((n_envs,) + self._field(self.datas[0], k).shape, dtype=torch.float32)
                       for k in WRITABLE + DERIVED}
        self._pulled = {k: None for k in WRITABLE}
        self._pull(WRITABLE + DERIVED)

    @staticmethod
    def _field(d, name):
        a = getattr(d, name)
        return a.reshape(-1, a.shape[-1]) if name in ("xpos", "xquat", "cvel") else a

    def state(self, name):
        return self._state[name]

    def _push(self):
        """Copy the writable fields the task changed into every MjData."""
        for k in WRITABLE:
            t = self._state[k]
            if t.numel() and not torch.equal(t, self._pulled[k]):
                v = t.numpy().astype(np.float64)
                for w, d in enumerate(self.datas):
                    getattr(d, k)[:] = v[w]

    def _pull(self, names):
        for k in names:
            t = self._state[k]
            if t.numel():
                t.copy_(torch.from_numpy(np.stack([self._field(d, k) for d in self.datas])))
            if k in WRITABLE:
                self._pulled[k] = t.clone()

    def step(self, n_substeps):
        self._push()
        m = self.mj_model
        for d in self.datas:
            for _ in range(n_substeps):
                mujoco.mj_step(m, d)
        self._pull(WRITABLE)

    def refresh_kinematics(self):
        # mj_forward fills body poses and velocities and tendon lengths and
        # velocities consistently with qpos/qvel. The next mj_step recomputes
        # everything it needs.
        self._push()
        for d in self.datas:
            mujoco.mj_forward(self.mj_model, d)
        self._pull(DERIVED)

    def contacts(self):
        geom, dist, world = [], [], []
        for w, d in enumerate(self.datas):
            n = d.ncon
            if n:
                geom.append(np.array(d.contact.geom[:n]))
                dist.append(np.array(d.contact.dist[:n]))
                world.append(np.full(n, w))
        if not geom:
            return (torch.zeros((0, 2), dtype=torch.int32), torch.zeros(0),
                    torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.bool))
        geom = torch.from_numpy(np.concatenate(geom)).to(torch.int32)
        return (geom, torch.from_numpy(np.concatenate(dist)).float(),
                torch.from_numpy(np.concatenate(world)).to(torch.int32), torch.ones(geom.size(0), dtype=torch.bool))

    def copy_to_mjdata(self, mj_data, world=0):
        self._push()
        mujoco.mj_copyData(mj_data, self.mj_model, self.datas[world])

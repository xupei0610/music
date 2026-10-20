"""mujoco_warp simulator: all worlds in one batch on the GPU (or warp's CPU backend)."""
import torch
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src import forward as mjw_forward
import mujoco_warp_patch  # noqa: F401  overrides mujoco_warp functions. Import it before the first kernel build.

from sim import Simulator, WRITABLE, DERIVED


def warp2torch(arr: wp.array) -> torch.Tensor:
    return wp.to_torch(arr)


class WarpSimulator(Simulator):

    def __init__(self, mj_model, mj_data, n_envs, compute_device, nconmax=None, njmax=None, cuda_graph=True):
        self.mj_model = mj_model
        self.n_envs = n_envs
        if compute_device == "cpu":
            self.device = torch.device("cpu")
            self.warp_device = wp.get_device("cpu")
        else:
            self.device = torch.device("cuda", compute_device)
            self.warp_device = wp.get_device(f"cuda:{compute_device}")
        wp.set_device(self.warp_device)
        # replay fixed GPU work (physics substeps, kinematics refresh) as CUDA graphs
        self.cuda_graph = cuda_graph and self.warp_device.is_cuda
        self._graphs = {}

        self.model = mjw.put_model(mj_model)
        # Reaching the solver/linesearch iteration budget (opt.iterations /
        # opt.ls_iterations) is the configured behavior, as in MJX, which caps
        # silently. mujoco_warp otherwise prints a line once per world per step.
        # The event is still recorded per world in data.overflow, and the
        # buffer-overflow warnings (nconmax/njmax/broadphase) stay enabled.
        self.model.opt.warn_overflow = int(self.model.opt.warn_overflow) & \
            ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        # mujoco_warp ignores the <size njmax/nconmax> hints from XML. Pass
        # them explicitly so that contact/constraint buffers do not overflow at
        # runtime ("nefc overflow", "broadphase overflow"). Configs may set
        # tighter per-world sizes (env_params nconmax/njmax). The solver cost
        # and memory scale with them. Overflows keep printing a warning.
        if nconmax is not None:
            nconmax = int(nconmax)
        else:
            nconmax = max(int(mj_model.nconmax), 512) if mj_model.nconmax > 0 else 512
        if njmax is not None:
            njmax = int(njmax)
        else:
            njmax = max(int(mj_model.njmax), 1024) if mj_model.njmax > 0 else 1024
        print("mujoco_warp buffers per world: nconmax={} njmax={}".format(nconmax, njmax))
        # Contacts live in one pool for all worlds (naconmax), so with many
        # worlds one world can use more than nconmax. With few worlds (test and
        # evaluation runs) the pool would be too small for one world's peak.
        # Keep it at least 512 contacts.
        naconmax = max(nconmax * n_envs, 512)
        self.data = mjw.put_data(
            mj_model, mj_data, nworld=n_envs,
            nconmax=nconmax, njmax=njmax, naconmax=naconmax,
        )
        # Zero-copy torch views of the warp arrays. In-place writes go straight to the simulation.
        self._state = {k: warp2torch(getattr(self.data, k)) for k in WRITABLE + DERIVED}

    def state(self, name):
        return self._state[name]

    def _launch(self, key, fn):
        """Run fn(), a fixed sequence of warp launches on model/data.

        With cuda_graph, the first call runs fn() eagerly, which also builds
        and loads its kernels. The second call records fn() into a CUDA graph.
        That call and every later one replay the graph with one launch instead
        of one Python-side launch per kernel. The graph reuses the same device
        arrays, so inputs (ctrl, qpos, ...) are written into them in place
        before the call, and fn must not depend on Python-side branching on
        device data. Callers pass a key that identifies the work, for example
        with the substep count in it.
        """
        if not self.cuda_graph:
            fn()
            return
        graph = self._graphs.get(key)
        if graph is None:
            if key not in self._graphs:
                self._graphs[key] = None  # warm-up: build and load kernels
                fn()
                return
            with wp.ScopedCapture(device=self.warp_device) as capture:
                fn()
            graph = self._graphs[key] = capture.graph
        wp.capture_launch(graph)

    def step(self, n_substeps):
        def substeps():
            for _ in range(n_substeps):
                mjw.step(self.model, self.data)
        self._launch(("physics", n_substeps), substeps)

    def refresh_kinematics(self):
        self._launch("kinematics", self._kinematics)

    def _kinematics(self):
        # After mjw.step, qpos/qvel/act are advanced but xpos/cvel/ten_length/
        # ten_velocity still reflect the pre-step state. Recompute only the
        # kinematic quantities the observations read: body poses (xpos/xquat),
        # body velocities (cvel) and tendon length/velocity. This is the part
        # of mjw.fwd_velocity they need. Its passive forces, bias forces (rne)
        # and actuator velocities are recomputed by every mjw.step anyway.
        m, d = self.model, self.data
        mjw.kinematics(m, d)
        mjw.com_pos(m, d)
        if self.mj_model.ntendon > 0:
            mjw.tendon(m, d)
            wp.launch(
                mjw_forward._tendon_velocity,
                dim=(d.nworld, m.ntendon),
                inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.qvel, d.ten_J],
                outputs=[d.ten_velocity],
            )
        mjw.com_vel(m, d)

    def contacts(self):
        # mujoco_warp stores the contacts of all worlds in one flat buffer.
        # Its first `nacon` entries are valid. Unlike MJX, which reports every
        # candidate pair with its distance, mujoco_warp only reports pairs
        # within the geom margin.
        d = self.data
        geom = warp2torch(d.contact.geom)
        valid = torch.arange(geom.size(0), device=self.device) < warp2torch(d.nacon)
        return geom, warp2torch(d.contact.dist), warp2torch(d.contact.worldid), valid

    def copy_to_mjdata(self, mj_data, world=0):
        mjw.get_data_into(mj_data, self.mj_model, self.data, world_id=world)

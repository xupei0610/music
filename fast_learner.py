import torch


class GraphedStep:
    """Replays fn(*inputs) (forward + backward + optimizer.step) as one CUDA graph.

    Built lazily on the first call. Warmup and capture leave parameters,
    optimizer state and the CUDA RNG state untouched. fn must not call
    zero_grad and must not sync with the CPU. __call__ returns the static
    outputs, so clone them before the next call.
    """

    def __init__(self, fn, modules, optimizers):
        self.fn = fn
        self.params = [p for m in modules for p in m.parameters()]
        self.optimizers = optimizers
        self.graph = None

    def _state_tensors(self):
        return [v for o in self.optimizers for st in o.state.values() for v in st.values() if torch.is_tensor(v)]

    def _zero_grad(self):
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)

    def _build(self, inputs):
        self.static = [x.clone() for x in inputs]
        device = self.params[0].device
        saved_params = [p.detach().clone() for p in self.params]
        saved_state = {id(v): (v, v.clone()) for v in self._state_tensors()}
        rng = torch.cuda.get_rng_state(device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._zero_grad()
                self.fn(*self.static)
        torch.cuda.current_stream().wait_stream(s)

        self._zero_grad()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.fn(*self.static)

        with torch.no_grad():
            for p, c in zip(self.params, saved_params):
                p.data.copy_(c)
            for v in self._state_tensors():
                if id(v) in saved_state:
                    v.copy_(saved_state[id(v)][1])
                else:
                    v.zero_()
        torch.cuda.set_rng_state(rng, device)

    def __call__(self, *inputs):
        if self.graph is None:
            self._build(inputs)
        for st, x in zip(self.static, inputs):
            st.copy_(x)
        self.graph.replay()
        return self.out


def disc_step(disc, opt, r, f, seq_end_frame):
    """One discriminator minibatch update. Returns detached mean real/fake scores."""
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
    return score_r.mean().detach(), score_f.mean().detach()


def ppo_step(models, optimizer, ac_parameters, REG, s, end_frame, *minibatch):
    """One PPO minibatch update. minibatch is (a, lp, adv, v_t) per model, flattened.
    Returns detached policy and value losses."""
    pg_loss, vf_loss = 0, 0
    for i, m in enumerate(models):
        a, lp, adv, v_t = minibatch[4*i:4*i+4]

        pi_, v_ = m(s, end_frame)
        use_reg = REG and m.actor.max_log_sigma is not None
        if use_reg:
            lp_ = pi_.log_prob(a)
            mask = m.actor.logit < m.actor.max_log_sigma
            lp_ = torch.where(mask, lp_, lp_.detach())
        else:
            lp_ = pi_.log_prob(a)
        lp_ = lp_.sum(-1, keepdim=True)
        ratio = torch.exp(lp_ - lp)
        clipped_ratio = torch.clamp(ratio, 1.0-0.2, 1.0+0.2)
        pg_loss = pg_loss - torch.min(adv*ratio, adv*clipped_ratio).sum(-1).mean()
        vf_loss = vf_loss + (v_ - v_t).square().mean()

        if use_reg:
            mask = torch.logical_not(mask)
            reg = (mask*pi_.scale).norm(2, dim=1).mean()
            pg_loss = pg_loss + REG*reg

    loss = pg_loss + 0.5*vf_loss

    loss.backward()
    for params in ac_parameters:
        torch.nn.utils.clip_grad_norm_(params, 1.0)

    optimizer.step()
    return pg_loss.detach(), vf_loss.detach()

import torch
import numpy as np
from typing import Optional, Union


class RunningMeanStd(torch.nn.Module):
    def __init__(self, dim: int, clamp: float=0, scale: float=1):
        super().__init__()
        self.epsilon = 1e-5
        self.clamp = clamp
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))
        self.scale2 = scale*scale

    def forward(self, x, unnorm=False):
        mean = self.mean.to(torch.float32)
        var = self.var.to(torch.float32)+self.epsilon
        if unnorm:
            if self.clamp:
                x = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return mean + torch.sqrt(self.scale2*var) * x
        x = (x - mean) * torch.rsqrt(self.scale2*var)
        if self.clamp:
            s = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return s
        return x
    
    @torch.no_grad()
    def update(self, x, count_scale=1):
        x = x.view(-1, x.size(-1))
        var, mean = torch.var_mean(x, dim=0, unbiased=True)
        count = x.size(0)/count_scale
        count_ = count + self.count
        delta = mean - self.mean
        m = self.var * self.count + var * count + delta**2 * self.count * count / count_
        self.mean.copy_(self.mean+delta*count/count_)
        self.var.copy_(m / count_)
        self.count.copy_(count_)

    def reset_counter(self):
        self.count.fill_(1)

class DiagonalPopArt(torch.nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, bias: torch.Tensor, momentum:float=0.1):
        super().__init__()
        self.epsilon = 1e-5

        self.momentum = momentum
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float64))
        self.register_buffer("v", torch.full((dim,), self.epsilon, dtype=torch.float64))
        self.register_buffer("debias", torch.zeros(1, dtype=torch.float64))

        self.weight = weight
        self.bias = bias

    def forward(self, x, unnorm=False):
        debias = self.debias.clip(min=self.epsilon)
        mean = self.m/debias
        var = (self.v - self.m.square()).div_(debias)
        if unnorm:
            std = torch.sqrt(var)
            return (mean + std * x).to(x.dtype)
        x = ((x - mean) * torch.rsqrt(var)).to(x.dtype)
        return x

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        running_m = torch.mean(x, dim=0)
        running_v = torch.mean(x.square(), dim=0)
        new_m = self.m.mul(1-self.momentum).add_(running_m, alpha=self.momentum)
        new_v = self.v.mul(1-self.momentum).add_(running_v, alpha=self.momentum)
        std = (self.v - self.m.square()).sqrt_()
        new_std_inv = (new_v - new_m.square()).rsqrt_()

        scale = std.mul_(new_std_inv)
        shift = (self.m - new_m).mul_(new_std_inv)

        self.bias.data.mul_(scale).add_(shift)
        self.weight.data.mul_(scale.unsqueeze_(-1))

        self.debias.data.mul_(1-self.momentum).add_(1.0*self.momentum)
        self.m.data.copy_(new_m)
        self.v.data.copy_(new_v)


class Discriminator(torch.nn.Module):
    def __init__(self, disc_dim, latent_dim=256):
        super().__init__()
        self.rnn = torch.nn.GRU(disc_dim, latent_dim, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32)
        )
        if self.rnn is not None:
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    gain = 1 if i == 2 else 2**0.5 
                    torch.nn.init.orthogonal_(p, gain=gain)
                    i += 1
        self.ob_normalizer = RunningMeanStd(disc_dim)
        self.all_inst = torch.arange(0)
        
    def forward(self, s, seq_end_frame, normalize=True):
        if normalize: s = self.ob_normalizer(s)
        if self.rnn is None:
            s = s.view(s.size(0), -1)
        else:
            n_inst = s.size(0)
            if n_inst > self.all_inst.size(0):
                self.all_inst = torch.arange(n_inst, 
                    dtype=seq_end_frame.dtype, device=seq_end_frame.device)
            s, _ = self.rnn(s)
            s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
        return self.mlp(s)

class ACModel(torch.nn.Module):

    class Critic(torch.nn.Module):
        def __init__(self, use_rnn, state_dim, goal_dim, value_dim=1, latent_dim=256):
            super().__init__()
            if use_rnn:
                self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            else:
                self.rnn = None
            if goal_dim and use_rnn:
                self.embed_goal = torch.nn.Sequential(
                    torch.nn.Linear(goal_dim, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, latent_dim)
                )
            else:
                self.embed_goal = None
            self.mlp = torch.nn.Sequential(
                # torch.nn.Linear((state_dim+(latent_dim if goal_dim else 0)) if self.rnn is None else latent_dim, 1024),
                torch.nn.Linear((state_dim+goal_dim) if self.rnn is None else latent_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, value_dim)
            )
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    torch.nn.init.uniform_(p, -0.0001, 0.0001)
                    i += 1
            self.all_inst = torch.arange(0)

        def forward(self, s, seq_end_frame, g=None):
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(n_inst, 
                        dtype=seq_end_frame.dtype, device=seq_end_frame.device)
                s, _ = self.rnn(s)
                s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
            if g is not None:
                if self.rnn is None:
                    if self.embed_goal is None:
                        s = torch.cat((s, g), -1)
                    else:
                        s = torch.cat((s, self.embed_goal(g)), -1)
                else:
                    s = s + self.embed_goal(g)
            return self.mlp(s)


    class Actor(torch.nn.Module):
        def __init__(self, use_rnn, state_dim, act_dim, goal_dim, latent_dim=256, init_mu=None, init_sigma=None, max_sigma=None, normalize_latent=False):
            super().__init__()
            if use_rnn:
                self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            else:
                self.rnn = None
            if goal_dim:
                self.embed_goal = torch.nn.Sequential(
                    torch.nn.Linear(goal_dim, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, latent_dim)
                )
            else:
                self.embed_goal = None
            self.normalize_latent = normalize_latent
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear((state_dim+(latent_dim if goal_dim else 0)) if self.rnn is None else latent_dim, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU()
            )
            self.mu = torch.nn.Linear(512, act_dim)
            self.log_sigma = torch.nn.Linear(512, act_dim)
            self.max_log_sigma2 = None
            with torch.no_grad():
                if init_mu is not None:
                    if torch.is_tensor(init_mu):
                        mu = torch.ones_like(self.mu.bias)*init_mu
                    else:
                        mu = np.ones(self.mu.bias.shape, dtype=np.float32)*init_mu
                        mu = torch.from_numpy(mu)
                    self.mu.bias.data.copy_(mu)
                    torch.nn.init.uniform_(self.mu.weight, -0.00001, 0.00001)
                if init_sigma is None:
                    torch.nn.init.constant_(self.log_sigma.bias, -3)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.0001, 0.0001)
                else:
                    if torch.is_tensor(init_sigma):
                        log_sigma = (torch.ones_like(self.log_sigma.bias)*init_sigma).exp_().sub_(1).log_()
                    else:
                        log_sigma = np.log(np.exp(np.ones(self.log_sigma.bias.shape, dtype=np.float32)*init_sigma)-1)
                        log_sigma = torch.from_numpy(log_sigma)
                    self.log_sigma.bias.data.copy_(log_sigma)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.00001, 0.00001)
                self.all_inst = torch.arange(0)
            
            if max_sigma:
                max_log_sigma = np.log(np.exp(max_sigma)-1)
                max_log_sigma = np.log(np.exp(max_sigma)-1)
                if hasattr(max_sigma, "__len__") and len(max_sigma) > 1:
                    self.register_buffer("max_log_sigma", torch.tensor(max_log_sigma, dtype=torch.float32))
                else:
                    self.max_log_sigma = max_log_sigma.item()
            else:
                self.max_log_sigma = None

        def forward(self, s, seq_end_frame=None, g=None, obs=None):
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(n_inst, 
                        dtype=seq_end_frame.dtype, device=seq_end_frame.device)
                s, _ = self.rnn(s)
                s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
            if g is not None:
                if self.rnn is None:
                    if self.embed_goal is None:
                        s = torch.cat((s, g), -1)
                    else:
                        g = self.embed_goal(g)
                        if self.normalize_latent:
                            g = g/g.norm(p=2,dim=-1,keepdim=True)
                            g = torch.nan_to_num(g, nan=(1/g.size(-1))**0.5)
                        self.latent = g
                        s = torch.cat((s, g), -1)
                else:
                    g = self.embed_goal(g)
                    if self.normalize_latent:
                        g = g/g.norm(p=2,dim=-1,keepdim=True)
                        g = torch.nan_to_num(g, nan=(1/g.size(-1))**0.5)
                    self.latent = g
                    s = s + g
            latent = self.mlp(s)
            
            mu = self.mu(latent)
            log_sigma = self.log_sigma(latent)
            if self.max_log_sigma is not None:
                log_sigma = log_sigma + ((log_sigma>self.max_log_sigma)*(self.max_log_sigma-log_sigma)).detach_()
            sigma = torch.log(1+torch.exp(log_sigma)) + 1e-6

            return torch.distributions.Normal(mu, sigma)

    def __init__(self, use_rnn: bool, state_dim: int, act_dim: int, goal_dim: int=0, value_dim: int=1, 
        normalize_value: bool=False,
        init_mu:Optional[Union[torch.Tensor, float]]=None,
        init_sigma:Optional[Union[torch.Tensor, float]]=None,
        max_sigma=None,
        normalizer_scale=1,
        latent_dim=256, normalize_latent=False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.actor = self.Actor(use_rnn, state_dim, act_dim, self.goal_dim, latent_dim, init_mu=init_mu, init_sigma=init_sigma, max_sigma=max_sigma, normalize_latent=normalize_latent)
        self.critic = self.Critic(use_rnn, state_dim, self.goal_dim, value_dim)
        self.ob_normalizer = RunningMeanStd(state_dim if use_rnn else (state_dim+self.goal_dim), scale=normalizer_scale) #, clamp=5.0)
        if normalize_value:            
            self.value_normalizer = DiagonalPopArt(value_dim, 
                self.critic.mlp[-1].weight, self.critic.mlp[-1].bias)
        else:
            self.value_normalizer = None
        self.use_rnn = use_rnn
            
    def observe(self, obs, norm=True):
        if self.use_rnn:
            if self.goal_dim > 0:
                s = obs[:, :-self.goal_dim]
                g = obs[:, -self.goal_dim:]
            else:
                s = obs
                g = None
            s = s.view(*s.shape[:-1], -1, self.state_dim)
            return self.ob_normalizer(s) if norm else s, g
        else:
            if norm:
                obs = self.ob_normalizer(obs)
            if self.goal_dim > 0:
                s = obs[:, :-self.goal_dim]
                g = obs[:, -self.goal_dim:]
            else:
                s = obs
                g = None
            s = s.view(*s.shape[:-1], -1, self.state_dim)
            return s, g
            

    def eval_(self, s, seq_end_frame, g, unnorm):
        v = self.critic(s, seq_end_frame, g)
        if unnorm and self.value_normalizer is not None:
            v = self.value_normalizer(v, unnorm=True)
        return v

    def act(self, obs, seq_end_frame, with_value=False, unnorm=False):
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, g, obs)
        if with_value:
            if g is not None:
                g = g[...,:self.goal_dim]
            return pi, self.eval_(s, seq_end_frame, g, unnorm)
        else:
            return pi

    def evaluate(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        if g is not None:
            g = g[...,:self.goal_dim]
        return self.eval_(s, seq_end_frame, g, unnorm)
    
    def forward(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, g, obs)
        if g is not None:
            g = g[...,:self.goal_dim]
        return pi, self.eval_(s, seq_end_frame, g, unnorm)

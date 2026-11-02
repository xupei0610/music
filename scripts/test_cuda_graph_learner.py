"""Eager vs CUDA-graph learner equivalence test. Needs a CUDA device.

Run from music-latest: python scripts/test_cuda_graph_learner.py
"""
import copy, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from models import ACModel, Discriminator
from fast_learner import GraphedStep, disc_step, ppo_step

torch.distributions.Distribution.set_default_validate_args(False)

DEV = "cuda"
STEPS, TIMED, B = 20, 50, 64
STATE, ACT, GOAL, VAL, HORIZON = 24, 8, 6, 3, 2
DISC_L, DISC_D = 4, 16
TOL = 1e-3


def max_diff(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))


def sync_time(fn, n):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3


def make_ppo(seed):
    torch.manual_seed(seed)
    model = ACModel(True, STATE, ACT, GOAL, VAL).to(DEV)
    return model


def ppo_data(n):
    g = torch.Generator(device=DEV).manual_seed(1)
    out = []
    for _ in range(n):
        s = torch.randn(B, HORIZON*STATE + GOAL, device=DEV, generator=g)
        end = torch.randint(0, HORIZON, (B,), device=DEV, generator=g)
        a = torch.randn(B, ACT, device=DEV, generator=g)
        lp = -torch.rand(B, 1, device=DEV, generator=g) * ACT
        adv = torch.randn(B, VAL, device=DEV, generator=g)
        v_t = torch.randn(B, VAL, device=DEV, generator=g)
        out.append((s, end, a, lp, adv, v_t))
    return out


def disc_data(n):
    g = torch.Generator(device=DEV).manual_seed(2)
    out = []
    for _ in range(n):
        r = torch.randn(B, DISC_L, DISC_D, device=DEV, generator=g)
        f = torch.randn(B, DISC_L, DISC_D, device=DEV, generator=g)
        end = torch.randint(0, DISC_L, (B,), device=DEV, generator=g)
        out.append((r, f, end))
    return out


def run_ppo(data):
    ref = make_ppo(0)
    eager, graphed = ref, copy.deepcopy(ref)

    def build(m, capturable):
        models = [m]
        opt = torch.optim.Adam([
            {"params": list(m.actor.parameters()), "lr": 5e-6},
            {"params": list(m.critic.parameters()), "lr": 1e-4},
        ], capturable=capturable)
        return models, opt, [list(m.actor.parameters()) + list(m.critic.parameters())]

    me, oe, ace = build(eager, False)
    mg, og, acg = build(graphed, True)
    me[0].train(); mg[0].train()

    def eager_step(d):
        s, end, *mb = d
        out = ppo_step(me, oe, ace, 0, s, end, *mb)
        oe.zero_grad()
        return out

    gs = GraphedStep(lambda *x: ppo_step(mg, og, acg, 0, *x), mg, [og])

    le, lg = [], []
    for d in data[:STEPS]:
        le.append(torch.stack(eager_step(d)))
        lg.append(torch.stack(gs(*d)).clone())
    le, lg = torch.stack(le), torch.stack(lg)
    diff = max_diff(eager, graphed)
    ldiff = (le - lg).abs().max().item()
    te = sync_time(lambda: eager_step(data[0]), TIMED)
    tg = sync_time(lambda: gs(*data[0]), TIMED)
    return diff, ldiff, te, tg


def run_disc(data):
    torch.manual_seed(0)
    de = Discriminator(DISC_D).to(DEV)
    dg = copy.deepcopy(de)
    oe = torch.optim.Adam(de.parameters(), 1e-5)
    og = torch.optim.Adam(dg.parameters(), 1e-5, capturable=True)
    de.train(); dg.train()

    def eager_step(d):
        out = disc_step(de, oe, *d)
        oe.zero_grad()
        return out

    gs = GraphedStep(lambda r, f, e: disc_step(dg, og, r, f, e), [dg], [og])

    torch.cuda.manual_seed(123)
    le = [torch.stack(eager_step(d)) for d in data[:STEPS]]
    torch.cuda.manual_seed(123)
    lg = [torch.stack(gs(*d)).clone() for d in data[:STEPS]]
    le, lg = torch.stack(le), torch.stack(lg)
    diff = max_diff(de, dg)
    ldiff = (le - lg).abs().max().item()
    te = sync_time(lambda: eager_step(data[0]), TIMED)
    tg = sync_time(lambda: gs(*data[0]), TIMED)
    return diff, ldiff, te, tg


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(2)
    ok = True
    for name, fn, data in (("ppo", run_ppo, ppo_data(STEPS)), ("disc", run_disc, disc_data(STEPS))):
        diff, ldiff, te, tg = fn(data)
        print("{}: max param diff {:.3e}, max score/loss diff {:.3e}, eager {:.2f} ms/step, graph {:.2f} ms/step ({:.1f}x)".format(
            name, diff, ldiff, te, tg, te / tg))
        ok = ok and diff <= TOL
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

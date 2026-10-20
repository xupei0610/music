"""Runtime patches to mujoco_warp; import before the first mujoco_warp step.

Like the MJX upstream, which replaced MJX's collision driver from env.py,
this module overrides mujoco_warp functions from our side instead of editing
the installed package.

Warp resolves the functions a kernel calls by global name when the kernel is
first hashed and built, so rebinding a module attribute here takes effect for
every kernel built afterwards (and gives those kernels a new cache key).
Kernels already built in this process keep the old function.
"""
from typing import Tuple

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import MJ_MINVAL


@wp.func
def wrap_inside(
  # In:
  end: wp.vec4,
  radius: float,
  maxiter: int = 20,
  zinit: float = 1.0 - 1.0e-7,
  tolerance: float = 1.0e-6,
) -> Tuple[float, wp.vec2, wp.vec2]:
  """2D inside wrap (tendon sidesite inside a sphere/cylinder wrap).

  Drop-in replacement for mujoco_warp's util_misc.wrap_inside, which is a
  float32 transcription of MuJoCo's double-precision mju_wrapInside. It finds
  the same root of asin(A*z) + asin(B*z) - 2*asin(z) + G = 0 with the same
  initialization, tolerance and failure branches, with two changes that only
  matter in float32:
    - Newton runs on theta = asin(z). The root is close to z = 1, where asin
      and its derivative 1/sqrt(1 - z^2) amplify the float32 rounding of z
      beyond the 1e-6 tolerance, so the last Newton step lands on f > tol, the
      "SHOULD NOT OCCUR" branch, and the function silently returns its
      fallback (the averaged end direction) instead of the solution. In theta
      the function is well conditioned and Newton converges as in double
      precision.
    - G, the angle between the two end points, is atan2(|cross|, dot) instead
      of acos(cosG), which loses precision for small angles in float32.
  """
  end0 = wp.vec2(end[0], end[1])
  end1 = wp.vec2(end[2], end[3])

  # constants
  len0 = wp.norm_l2(end0)
  len1 = wp.norm_l2(end1)
  dif = end1 - end0
  dd = wp.dot(dif, dif)

  # either point inside circle or circle too small: no wrap
  if (len0 <= radius) or (len1 <= radius) or (radius < MJ_MINVAL) or (len0 < MJ_MINVAL) or (len1 < MJ_MINVAL):
    return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  # segment-circle intersection: no wrap
  if dd > MJ_MINVAL:
    # find nearest point on line segment to origin: d0 + a * dif
    a = -wp.dot(dif, end0) / dd

    # in segment
    if (a > 0.0) and (a < 1.0):
      tmp = end0 + a * dif
      if wp.norm_l2(tmp) <= radius:
        return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  # prepare default in case of numerical failure: average
  pnt = 0.5 * (end0 + end1)
  pnt, _ = math.normalize_with_norm(pnt)
  pnt *= radius

  # compute function parameters: asin(A * sin(t)) + asin(B * sin(t)) - 2 * t + G = 0
  A = math.safe_div(radius, len0)
  B = math.safe_div(radius, len1)
  sq_A = A * A
  sq_B = B * B
  cosG = math.safe_div(len0 * len0 + len1 * len1 - dd, 2.0 * len0 * len1)
  if cosG < -1.0 + MJ_MINVAL:
    return -1.0, pnt, pnt
  elif cosG > 1.0 - MJ_MINVAL:
    return 0.0, pnt, pnt
  G = wp.atan2(wp.abs(end0[0] * end1[1] - end0[1] * end1[0]), wp.dot(end0, end1))

  # init
  t = wp.asin(zinit)
  st = wp.sin(t)
  f = wp.asin(A * st) + wp.asin(B * st) - 2.0 * t + G

  # make sure init is not on the other side
  if f > 0.0:
    return 0.0, pnt, pnt

  # Newton method
  iter = int(0)

  while (iter < maxiter) and (wp.abs(f) > tolerance):
    # derivative w.r.t. t
    sq_st = st * st
    ct = wp.cos(t)
    df = (
      A * ct / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_st * sq_A))
      + B * ct / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_st * sq_B))
      - 2.0
    )

    # check sign; SHOULD NOT OCCUR
    if df > -MJ_MINVAL:
      return 0.0, pnt, pnt

    # new point
    t1 = t - math.safe_div(f, df)

    # make sure we are moving to the left; SHOULD NOT OCCUR
    if t1 > t:
      return 0.0, pnt, pnt

    # update solution
    t = t1
    st = wp.sin(t)
    f = wp.asin(A * st) + wp.asin(B * st) - 2.0 * t + G

    # exit if positive: SHOULD NOT OCCUR
    if f > tolerance:
      return 0.0, pnt, pnt

    iter += 1

  # check convergence
  if iter >= maxiter:
    return 0.0, pnt, pnt

  # finalize: rotation by ang from vec = a or b, depending on cross(a, b) sign
  if end[0] * end[3] - end[1] * end[2] > 0.0:
    vec = end0
    ang = t - wp.asin(A * st)
  else:
    vec = end1
    ang = t - wp.asin(B * st)

  vec, _ = math.normalize_with_norm(vec)
  pnt = wp.vec2(
    radius * (wp.cos(ang) * vec[0] - wp.sin(ang) * vec[1]),
    radius * (wp.sin(ang) * vec[0] + wp.cos(ang) * vec[1]),
  )

  return 0.0, pnt, pnt


util_misc.wrap_inside = wrap_inside

# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Methods for finding roots of functions of one variable."""

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

NUMPY_MODE = False

__all__ = [
    'bracket_root',
    'find_root_chandrupatla',
]

RootSearchResults = collections.namedtuple(
    'RootSearchResults',
    [
        # A tensor containing the last position explored. If the search was
        # successful, this position is a root of the objective function.
        'estimated_root',
        # A tensor containing the value of the objective function at the last
        # position explored. If the search was successful, then this is close
        # to 0.
        'objective_at_estimated_root',
        # The number of iterations performed.
        'num_iterations',
    ])

def _structure_broadcasting_where(c, x, y):
  """Selects elements from two structures using a shared condition `c`."""
  return tf.nest.map_structure(
      lambda xp, yp: tf.where(c, xp, yp), x, y)
  
def find_root_chandrupatla(objective_fn,
                           low=None,
                           high=None,
                           position_tolerance=1e-8,
                           value_tolerance=0.,
                           max_iterations=50,
                           stopping_policy_fn=tf.reduce_all,
                           validate_args=False,
                           name='find_root_chandrupatla'):
  r"""Finds root(s) of a scalar function using Chandrupatla's method.

  Chandrupatla's method [1, 2] is a root-finding algorithm that is guaranteed
  to converge if a root lies within the given bounds. It generalizes the
  [bisection method](https://en.wikipedia.org/wiki/Bisection_method); at each
  step it chooses to perform either bisection or inverse quadratic
  interpolation. This makes it similar in spirit to [Brent's method](
  https://en.wikipedia.org/wiki/Brent%27s_method), which also considers steps
  that use the secant method, but Chandrupatla's method is simpler and often
  converges at least as quickly [3].

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      callable of a single variable. `objective_fn` must return a `Tensor` with
      shape `batch_shape` and dtype matching `lower_bound` and `upper_bound`.
    low: Float `Tensor` of shape `batch_shape` representing a lower
      bound(s) on the value of a root(s). If either of `low` or `high` is not
      provided, both are ignored and `tfp.math.bracket_root` is used to attempt
      to infer bounds.
      Default value: `None`.
    high: Float `Tensor` of shape `batch_shape` representing an upper
      bound(s) on the value of a root(s). If either of `low` or `high` is not
      provided, both are ignored and `tfp.math.bracket_root` is used to attempt
      to infer bounds.
      Default value: `None`.
    position_tolerance: Optional `Tensor` representing the maximum absolute
      error in the positions of the estimated roots. Shape must broadcast with
      `batch_shape`.
      Default value: `1e-8`.
    value_tolerance: Optional `Tensor` representing the absolute error allowed
      in the value of the objective function. If the absolute value of
      `objective_fn` is smaller than
      `value_tolerance` at a given position, then that position is considered a
      root for the function. Shape must broadcast with `batch_shape`.
      Default value: `1e-8`.
    max_iterations: Optional `Tensor` or Python integer specifying the maximum
      number of steps to perform. Shape must broadcast with `batch_shape`.
      Default value: `50`.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.
      It must be a callable accepting a `Tensor` of booleans with the same shape
      as `lower_bound` and `upper_bound` (denoting whether each search is
      finished), and returning a scalar boolean `Tensor` indicating
      whether the overall search should stop. Typical values are
      `tf.reduce_all` (which returns only when the search is finished for all
      points), and `tf.reduce_any` (which returns as soon as the search is
      finished for any point).
      Default value: `tf.reduce_all` (returns only when the search is finished
        for all points).
    validate_args: Python `bool` indicating whether to validate arguments.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: 'find_root_chandrupatla'.

  Returns:
    root_search_results: A Python `namedtuple` containing the following items:
      estimated_root: `Tensor` containing the last position explored. If the
        search was successful within the specified tolerance, this position is
        a root of the objective function.
      objective_at_estimated_root: `Tensor` containing the value of the
        objective function at `position`. If the search was successful within
        the specified tolerance, then this is close to 0.
      num_iterations: The number of iterations performed.

  #### References

  [1] Tirupathi R. Chandrupatla. A new hybrid quadratic/bisection algorithm for
      finding the zero of a nonlinear function without using derivatives.
      _Advances in Engineering Software_, 28.3:145-149, 1997.
  [2] Philipp OJ Scherer. Computational Physics. _Springer Berlin_,
      Heidelberg, 2010.
      Section 6.1.7.3 https://books.google.com/books?id=cC-8BAAAQBAJ&pg=PA95
  [3] Jason Sachs. Ten Little Algorithms, Part 5: Quadratic Extremum
      Interpolation and Chandrupatla's Method (2015).
      https://www.embeddedrelated.com/showarticle/855.php
  """

  ################################################
  # Loop variables used by Chandrupatla's method:
  #
  #  a: endpoint of an interval `[min(a, b), max(a, b)]` containing the
  #     root. There is no guarantee as to which of `a` and `b` is larger.
  #  b: endpoint of an interval `[min(a, b), max(a, b)]` containing the
  #       root. There is no guarantee as to which of `a` and `b` is larger.
  #  f_a: value of the objective at `a`.
  #  f_b: value of the objective at `b`.
  #  t: the next position to be evaluated as the coefficient of a convex
  #    combination of `a` and `b` (i.e., a value in the unit interval).
  #  num_iterations: integer number of steps taken so far.
  #  converged: boolean indicating whether each batch element has converged.
  #
  # All variables have the same shape `batch_shape`.

  def _should_continue(a, b, f_a, f_b, t, num_iterations, converged):
    del a, b, f_a, f_b, t  # Unused.
    all_converged = stopping_policy_fn(
        tf.logical_or(converged,
                      num_iterations >= max_iterations))
    return ~all_converged

  def _body(a, b, f_a, f_b, t, num_iterations, converged):
    """One step of Chandrupatla's method for root finding."""
    previous_loop_vars = (a, b, f_a, f_b, t, num_iterations, converged)
    finalized_elements = tf.logical_or(converged,
                                       num_iterations >= max_iterations)

    # Evaluate the new point.
    x_new = (1 - t) * a + t * b
    # tf.print(tf.reduce_mean(x_new))
    f_new = objective_fn(x_new)
    # tf.print(tf.reduce_mean(f_new))
    # If we've bisected (t==0.5) and the new float value for `a` is identical to
    # that from the previous iteration, then we'll keep bisecting (the
    # logic below will set t==0.5 for the next step), and nothing further will
    # change.
    at_fixed_point = tf.equal(x_new, a) & tf.equal(t, 0.5)
    # Otherwise, tighten the bounds.
    a, b, c, f_a, f_b, f_c = _structure_broadcasting_where(
        tf.equal(tf.math.sign(f_new), tf.math.sign(f_a)),
        (x_new, b, a, f_new, f_b, f_a),
        (x_new, a, b, f_new, f_a, f_b))

    # Check for convergence.
    f_best = tf.where(tf.abs(f_a) < tf.abs(f_b), f_a, f_b)
    interval_tolerance = position_tolerance / (tf.abs(b - c))
    converged = tf.logical_or(interval_tolerance > 0.5,
                              tf.logical_or(
                                  tf.math.abs(f_best) <= value_tolerance,
                                  at_fixed_point))

    # Propose next point to evaluate.
    xi = (a - b) / (c - b)
    phi = (f_a - f_b) / (f_c - f_b)
    t = 0.5
    # t = tf.where(
    #     # Condition for inverse quadratic interpolation.
    #     tf.logical_and(1 - tf.math.sqrt(1 - xi) < phi,
    #                    tf.math.sqrt(xi) > phi),
    #     # Propose a point by inverse quadratic interpolation.
    #     (f_a / (f_b - f_a) * f_c / (f_b - f_c) +
    #      (c - a) / (b - a) * f_a / (f_c - f_a) * f_b / (f_c - f_b)),
    #     # Otherwise, just cut the interval in half (bisection).
    #     0.5)
    # # Constrain the proposal to the current interval (0 < t < 1).
    # t = tf.minimum(tf.maximum(t, interval_tolerance),
    #                1 - interval_tolerance)

    # Update elements that haven't converged.
    return _structure_broadcasting_where(
        finalized_elements,
        previous_loop_vars,
        (a, b, f_a, f_b, t, num_iterations + 1, converged))

  with tf.name_scope(name):
    max_iterations = tf.convert_to_tensor(
        max_iterations, name='max_iterations', dtype_hint=tf.int32)
    dtype = dtype_util.common_dtype(
        [low, high, position_tolerance, value_tolerance], dtype_hint=tf.float64)
    position_tolerance = tf.convert_to_tensor(
        position_tolerance, name='position_tolerance', dtype=dtype)
    value_tolerance = tf.convert_to_tensor(
        value_tolerance, name='value_tolerance', dtype=dtype)

    if low is None or high is None:
      a, b = bracket_root(objective_fn, dtype=dtype)
    else:
      a = tf.convert_to_tensor(low, name='lower_bound', dtype=dtype)
      b = tf.convert_to_tensor(high, name='upper_bound', dtype=dtype)
    f_a, f_b = objective_fn(a), objective_fn(b)
    batch_shape = ps.broadcast_shape(ps.shape(f_a), ps.shape(f_b))

    assertions = []
    if validate_args:
      assertions += [
          assert_util.assert_none_equal(
              tf.math.sign(f_a), tf.math.sign(f_b),
              message='Bounds must be on different sides of a root.')]

    with tf.control_dependencies(assertions):
      initial_loop_vars = [
          a,
          b,
          f_a,
          f_b,
          tf.cast(0.5, dtype=f_a.dtype),
          tf.cast(0, dtype=max_iterations.dtype),
          False
      ]
      a, b, f_a, f_b, _, num_iterations, _ = tf.while_loop(
          _should_continue,
          _body,
          loop_vars=tf.nest.map_structure(
              lambda x: tf.broadcast_to(x, batch_shape),
              initial_loop_vars))

    x_best, f_best = _structure_broadcasting_where(
        tf.abs(f_a) < tf.abs(f_b),
        (a, f_a),
        (b, f_b))
    x_best = tf.where(tf.math.is_nan(x_best), tf.ones_like(x_best), x_best)
    f_best = tf.where(tf.math.is_nan(f_best), tf.ones_like(f_best), f_best)
    # tf.print(tf.math.is_nan(tf.reduce_sum(f_best)))
    # tf.print('is_nan f_best' + str(tf.get_static_value(tf.math.is_nan(tf.reduce_sum(f_best)))))
    if tf.math.is_nan(tf.reduce_sum(f_best)):
      tf.print(tf.reduce_sum(x_best))
      tf.print(tf.reduce_sum(f_best))
    # tf.print(x_best.shape)
    # tf.print(f_best.shape)
  return RootSearchResults(
      estimated_root=x_best,
      objective_at_estimated_root=f_best,
      num_iterations=num_iterations)


def bracket_root(objective_fn,
                 dtype=tf.float64,
                 num_points=512,
                 name='bracket_root'):
  """Finds bounds that bracket a root of the objective function.

  This method attempts to return an interval bracketing a root of the objective
  function. It evaluates the objective in parallel at `num_points`
  locations, at exponentially increasing distance from the origin, and returns
  the first pair of adjacent points `[low, high]` such that the objective is
  finite and has a different sign at the two points. If no such pair was
  observed, it returns the trivial interval
  `[np.finfo(dtype).min, np.finfo(dtype).max]` containing all float values of
  the specified `dtype`. If the objective has multiple
  roots, the returned interval will contain at least one (but perhaps not all)
  of the roots.

  Args:
    objective_fn: Python callable for which roots are searched. It must be a
      continuous function that accepts a scalar `Tensor` of type `dtype` and
      returns a `Tensor` of shape `batch_shape`.
    dtype: Optional float `dtype` of inputs to `objective_fn`.
      Default value: `tf.float32`.
    num_points: Optional Python `int` number of points at which to evaluate
      the objective.
      Default value: `512`.
    name: Python `str` name given to ops created by this method.
  Returns:
    low: Float `Tensor` of shape `batch_shape` and dtype `dtype`. Lower bound
      on a root of `objective_fn`.
    high: Float `Tensor` of shape `batch_shape` and dtype `dtype`. Upper bound
      on a root of `objective_fn`.
  """
  with tf.name_scope(name):
    # Build a logarithmic sequence of `num_points` values from -inf to inf.
    dtype_info = np.finfo(dtype_util.as_numpy_dtype(dtype))
    xs_positive = tf.exp(tf.linspace(tf.cast(-10., dtype),
                                     tf.math.log(dtype_info.max),
                                     num_points // 2))
    xs = tf.concat([tf.reverse(-xs_positive, axis=[0]), xs_positive], axis=0)

    # Evaluate the objective at all points. The objective function may return
    # a batch of values (e.g., `objective(x) = x - batch_of_roots`).
    if NUMPY_MODE:
      objective_output_spec = objective_fn(tf.zeros([], dtype=dtype))
    else:
      objective_output_spec = callable_util.get_output_spec(
          objective_fn,
          tf.convert_to_tensor(0., dtype=dtype))
    batch_ndims = tensorshape_util.rank(objective_output_spec.shape)
    if batch_ndims is None:
      raise ValueError('Cannot infer tensor rank of objective values.')
    xs_pad_shape = ps.pad([num_points],
                          paddings=[[0, batch_ndims]],
                          constant_values=1)
    ys = objective_fn(tf.reshape(xs, xs_pad_shape))

    # Find the smallest point where the objective is finite.
    is_finite = tf.math.is_finite(ys)
    ys_transposed = distribution_util.move_dimension(  # For batch gather.
        ys, 0, -1)
    first_finite_value = tf.gather(
        ys_transposed,
        tf.argmax(is_finite, axis=0),  # Index of smallest finite point.
        batch_dims=batch_ndims,
        axis=-1)
    # Select the next point where the objective has a different sign.
    sign_change_idx = tf.argmax(
        tf.not_equal(tf.math.sign(ys),
                     tf.math.sign(first_finite_value)) & is_finite,
        axis=0)
    # If the sign never changes, we can't bracket a root.
    bracketing_failed = tf.equal(sign_change_idx, 0)
    # If the objective's sign is zero, we've found an actual root.
    root_found = tf.equal(tf.gather(tf.math.sign(ys_transposed),
                                    sign_change_idx,
                                    batch_dims=batch_ndims,
                                    axis=-1),
                          0.)
    return _structure_broadcasting_where(
        bracketing_failed,
        # If we didn't detect a sign change, fall back to the trivial interval.
        (dtype_info.min, dtype_info.max),
        # Otherwise, return the points around the sign change, unless we
        # actually evaluated a root, in which case, return the zero-width
        # bracket at that root.
        (tf.gather(xs, tf.where(bracketing_failed | root_found,
                                sign_change_idx,
                                sign_change_idx - 1)),
         tf.gather(xs, sign_change_idx)))
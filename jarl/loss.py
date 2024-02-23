from typing import Callable

import optax
import jaxopt
import jax
import jax.numpy as jnp

import jarl

def dqn(
    gamma: float,
    forward: Callable[[optax.Params, jax.Array], jax.Array],
    params: optax.Params,
    target_params: optax.Params,
    transition: jarl.Transition,
    loss_fn: Callable = jaxopt.loss.huber_loss,
):
    q_values = forward(params, transition.obs)
    q_values = jnp.take_along_axis(q_values, transition.action[None], axis=-1)

    next_q_values = forward(target_params, transition.next_obs)
    next_q_values = jax.lax.stop_gradient(jnp.max(next_q_values, axis=-1, keepdims=True))

    target = transition.reward + (1 - transition.done) * gamma * next_q_values

    return loss_fn(q_values, target).mean()

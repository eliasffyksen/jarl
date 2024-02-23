from __future__ import annotations
from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

import jarl

class ReplayBuffer(NamedTuple):
    capacity: int
    rng: jax.Array
    buffer: jarl.Transition
    size: int = 0
    index: int = 0

    def init(
            capacity: int,
            example_transition: jarl.Transition,
            rng: jax.Array,
        ) -> ReplayBuffer:
        buffer = example_transition._replace(
            obs=jnp.zeros((capacity, *example_transition.obs.shape), dtype=example_transition.obs.dtype),
            action=jnp.zeros((capacity, *example_transition.action.shape), dtype=example_transition.action.dtype),
            reward=jnp.zeros((capacity, *example_transition.reward.shape), dtype=example_transition.reward.dtype),
            next_obs=jnp.zeros((capacity, *example_transition.next_obs.shape), dtype=example_transition.next_obs.dtype),
            done=jnp.zeros((capacity, *example_transition.done.shape), dtype=example_transition.done.dtype),
        )

        return ReplayBuffer(
            capacity=capacity,
            buffer=buffer,
            rng=rng,
        )

    def __len__(self) -> int:
        return self.size

    @partial(jax.jit, donate_argnames=('self'))
    def push(self, transition: jarl.Transition) -> ReplayBuffer:
        buffer = self.buffer._replace(
            obs=self.buffer.obs.at[self.index].set(transition.obs),
            action=self.buffer.action.at[self.index].set(transition.action),
            reward=self.buffer.reward.at[self.index].set(transition.reward),
            next_obs=self.buffer.next_obs.at[self.index].set(transition.next_obs),
            done=self.buffer.done.at[self.index].set(transition.done),
        )
        size = jax.lax.min(self.size + 1, self.capacity)
        index = (self.index + 1) % self.capacity

        return ReplayBuffer(
            capacity=self.capacity,
            buffer=buffer,
            rng=self.rng,
            size=size,
            index=index,
        )

    @partial(jax.jit, donate_argnames=('self',), static_argnames=('batch_size',))
    def sample(self, batch_size: int) -> tuple[jarl.Transition, ReplayBuffer]:
        rng, next_rng = jax.random.split(self.rng)
        idx = jax.random.randint(rng, (batch_size,), 0, self.size)

        return jarl.Transition(
            obs=self.buffer.obs[idx],
            action=self.buffer.action[idx],
            reward=self.buffer.reward[idx],
            next_obs=self.buffer.next_obs[idx],
            done=self.buffer.done[idx],
        ), ReplayBuffer(
            capacity=self.capacity,
            rng=next_rng,
            buffer=self.buffer,
            size=self.size,
            index=self.index,
        )

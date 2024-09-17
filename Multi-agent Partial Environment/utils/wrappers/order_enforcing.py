from __future__ import annotations

from typing import Any

import numpy as np

from ..env import (
    ActionType,
    AECEnv,
    AECIterable,
    AECIterator,
    AgentID,
    ObsType,
)
from ..env_logger import EnvLogger
from .base import BaseWrapper


class OrderEnforcingWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    """Checks if function calls or attribute access are in a disallowed order.
    # 属性访问控制，方法调用控制，重置状态
    The following are raised:
    * AttributeError if any of the following are accessed before reset():
      rewards, terminations, truncations, infos, agent_selection,
      num_agents, agents.
      即在调用 reset() 方法之前，如果尝试访问某些属性
      （如 rewards, terminations, truncations, infos, agent_selection, num_agents, agents），会引发 AttributeError。
    * An error if any of the following are called before reset:
      render(), step(), observe(), state(), agent_iter()
      即在调用 reset() 方法之前，如果尝试调用某些方法（如 render(), step(), observe(), state(), agent_iter()），会引发错误。
    * A warning if step() is called when there are no agents remaining.
    即如果在没有剩余智能体的情况下调用 step() 方法，会发出警告

    使用EnvLogger记录错误和警告信息，帮助开发者调试和跟踪问题。
    """

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        assert isinstance(
            env, AECEnv
        ), "OrderEnforcingWrapper is only compatible with AEC environments"
        self._has_reset = False
        self._has_updated = False
        super().__init__(env)

    def __getattr__(self, value: str) -> Any:
        """Raises an error if certain data is accessed before reset."""
        if (
            value
            in {
                "rewards",
                "terminations",
                "truncations",
                "infos",
                "agent_selection",
                "num_agents",
                "agents",
            }
            and not self._has_reset
        ):
            raise AttributeError(f"{value} cannot be accessed before reset")
        return super().__getattr__(value)

    def render(self) -> None | np.ndarray | str | list:
        if not self._has_reset:
            EnvLogger.error_render_before_reset()
        return super().render()

    def step(self, action: ActionType) -> None:
        if not self._has_reset:
            EnvLogger.error_step_before_reset()
        elif not self.agents:
            self._has_updated = True
            EnvLogger.warn_step_after_terminated_truncated()
        else:
            self._has_updated = True
            super().step(action)

    def observe(self, agent: AgentID) -> ObsType | None:
        if not self._has_reset:
            EnvLogger.error_observe_before_reset()
        return super().observe(agent)

    def state(self) -> np.ndarray:
        if not self._has_reset:
            EnvLogger.error_state_before_reset()
        return super().state()

    def agent_iter(
        self, max_iter: int = 2**63
    ) -> AECOrderEnforcingIterable[AgentID, ObsType, ActionType]:
        if not self._has_reset:
            EnvLogger.error_agent_iter_before_reset()
        return AECOrderEnforcingIterable(self, max_iter)

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._has_reset = True
        self._has_updated = True
        super().reset(seed=seed, options=options)

    def __str__(self) -> str:
        if hasattr(self, "metadata"):
            return (
                str(self.env)
                if self.__class__ is OrderEnforcingWrapper
                else f"{type(self).__name__}<{str(self.env)}>"
            )
        return repr(self)


class AECOrderEnforcingIterable(AECIterable[AgentID, ObsType, ActionType]):
    def __iter__(self) -> AECOrderEnforcingIterator[AgentID, ObsType, ActionType]:
        return AECOrderEnforcingIterator(self.env, self.max_iter)


class AECOrderEnforcingIterator(AECIterator[AgentID, ObsType, ActionType]):
    def __init__(
        self, env: OrderEnforcingWrapper[AgentID, ObsType, ActionType], max_iter: int
    ):
        assert isinstance(
            env, OrderEnforcingWrapper
        ), "env must be wrapped by OrderEnforcingWrapper"
        super().__init__(env, max_iter)

    def __next__(self) -> AgentID:
        agent = super().__next__()
        assert (
            self.env._has_updated  # pyright: ignore[reportGeneralTypeIssues]
        ), "need to call step() or reset() in a loop over `agent_iter`"
        self.env._has_updated = False  # pyright: ignore[reportGeneralTypeIssues]
        return agent

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PayloadValue = int | float | str | bool
Payload = tuple[tuple[str, PayloadValue], ...]


def _normalize_payload(payload: dict[str, PayloadValue]) -> Payload:
    normalized: list[tuple[str, PayloadValue]] = []
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("token payload keys must be nonempty strings")
        if not isinstance(value, (int, float, str, bool)):
            raise TypeError(f"unsupported token payload for key '{key}'")
        normalized.append((key, value))
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True)
class Token:
    kind: str
    payload: Payload = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("token kind must not be empty")
        for key, value in self.payload:
            if not isinstance(key, str) or not key:
                raise ValueError("token payload keys must be nonempty strings")
            if not isinstance(value, (int, float, str, bool)):
                raise TypeError(f"unsupported token payload for key '{key}'")

    @staticmethod
    def make(kind: str, **payload: PayloadValue) -> Token:
        return Token(kind=kind, payload=_normalize_payload(payload))

    def payload_dict(self) -> dict[str, PayloadValue]:
        return dict(self.payload)


T = Token.make


@dataclass(frozen=True)
class Stage1Attempt:
    def_index: int
    log_prob: float
    accepted: bool


@dataclass(frozen=True)
class PolicySample:
    stopped: bool
    log_prob: float
    def_index: int | None = None
    action_space: Any | None = None
    decision: dict[str, Any] | None = None
    def_attempts: tuple[Stage1Attempt, ...] = ()
    decision_tokens: tuple[Token, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.stopped:
            if self.def_index is not None or self.action_space is not None:
                raise ValueError("stopped sample must not contain a decision")
            if self.decision is not None:
                raise ValueError("stopped sample must not contain a decision")
            return
        if self.def_index is None:
            raise ValueError("rewrite sample requires def_index")
        if self.action_space is None:
            raise ValueError("rewrite sample requires action_space")
        if self.decision is None:
            raise ValueError("rewrite sample requires decision")

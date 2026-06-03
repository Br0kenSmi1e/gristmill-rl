from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from transformer_policy.types import PayloadValue, PolicySample, Token


TracePhase = Literal["def", "candidate", "left_bit", "right_bit"]


class TokenWire(TypedDict):
    kind: str
    payload: tuple[tuple[str, PayloadValue], ...]


class TokenChoiceEventWire(TypedDict):
    sequence_tokens: tuple[TokenWire, ...]
    legal_next_tokens: tuple[TokenWire, ...]
    chosen_index: int
    phase: TracePhase
    step_index: int


@dataclass(frozen=True)
class TokenChoiceEvent:
    sequence_tokens: tuple[Token, ...]
    legal_next_tokens: tuple[Token, ...]
    chosen_index: int
    phase: TracePhase
    step_index: int

    def __post_init__(self) -> None:
        if not self.sequence_tokens:
            raise ValueError("sequence_tokens must not be empty")
        if not self.legal_next_tokens:
            raise ValueError("legal_next_tokens must not be empty")
        if self.chosen_index < 0 or self.chosen_index >= len(self.legal_next_tokens):
            raise ValueError("chosen_index is outside legal_next_tokens")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True)
class TracedPolicySample:
    sample: PolicySample
    events: tuple[TokenChoiceEvent, ...]

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("traced sample must contain at least one token event")


def token_to_wire(token: Token) -> TokenWire:
    return {"kind": token.kind, "payload": token.payload}


def token_from_wire(wire: TokenWire) -> Token:
    kind = wire.get("kind")
    payload = wire.get("payload")
    if not isinstance(kind, str):
        raise ValueError("token wire kind must be a string")
    if not isinstance(payload, tuple):
        raise ValueError("token wire payload must be a tuple")
    normalized: list[tuple[str, PayloadValue]] = []
    for item in payload:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("token wire payload entries must be pairs")
        key, value = item
        if not isinstance(key, str):
            raise ValueError("token wire payload keys must be strings")
        if not isinstance(value, (int, float, str, bool)):
            raise ValueError("token wire payload values must be scalar")
        normalized.append((key, value))
    return Token(kind=kind, payload=tuple(normalized))


def event_to_wire(event: TokenChoiceEvent) -> TokenChoiceEventWire:
    return {
        "sequence_tokens": tuple(token_to_wire(token) for token in event.sequence_tokens),
        "legal_next_tokens": tuple(token_to_wire(token) for token in event.legal_next_tokens),
        "chosen_index": event.chosen_index,
        "phase": event.phase,
        "step_index": event.step_index,
    }


def event_from_wire(wire: TokenChoiceEventWire) -> TokenChoiceEvent:
    phase = wire.get("phase")
    if phase not in {"def", "candidate", "left_bit", "right_bit"}:
        raise ValueError("event wire phase is invalid")
    chosen_index = wire.get("chosen_index")
    step_index = wire.get("step_index")
    if type(chosen_index) is not int:
        raise ValueError("event wire chosen_index must be an int")
    if type(step_index) is not int:
        raise ValueError("event wire step_index must be an int")
    return TokenChoiceEvent(
        sequence_tokens=tuple(token_from_wire(token) for token in wire["sequence_tokens"]),
        legal_next_tokens=tuple(token_from_wire(token) for token in wire["legal_next_tokens"]),
        chosen_index=chosen_index,
        phase=phase,
        step_index=step_index,
    )

from __future__ import annotations

from dataclasses import dataclass

from .errors import DeclError


@dataclass(frozen=True)
class Partition:
    """A pipeline-wide contract for one partition key's stored values."""

    type: object = str
    choices: object = None

    def __post_init__(self) -> None:
        if not callable(self.type):
            raise DeclError(f"partition type must be callable, got {self.type!r}.")
        if self.choices is not None:
            try:
                choices = frozenset(str(self.type(v)) for v in self.choices)
            except (TypeError, ValueError) as e:
                raise DeclError(f"partition choices do not match {self.type!r}.") from e
            object.__setattr__(self, "choices", choices)

    def normalize(self, value, *, key: str = "partition", error=DeclError) -> str:
        try:
            parsed = self.type(value)
        except (TypeError, ValueError) as e:
            name = getattr(self.type, "__name__", repr(self.type))
            raise error(f"{key}={value!r} is not a valid {name} partition value.") from e
        text = str(parsed)
        if self.choices is not None and text not in self.choices:
            raise error(
                f"{key}={value!r} is not allowed; expected one of "
                f"{', '.join(sorted(self.choices))}.")
        return text

"""Parse and validate piecewise-constant learning-rate schedules."""

from __future__ import annotations


def parse_learning_rate_schedule(specification: str, target_epochs: int) -> list[dict[str, float | int]]:
    """Parse ``end_epoch:rate`` pairs covering epochs 1 through target_epochs."""
    if target_epochs < 1:
        raise ValueError("target-epochs must be positive")
    if not specification.strip():
        raise ValueError("learning-rate schedule must not be empty")

    endpoints: list[tuple[int, float]] = []
    for item in specification.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError(
                "learning-rate schedule must use end_epoch:rate pairs, "
                "for example 100:0.01,200:0.005,300:0.001"
            )
        try:
            end_epoch = int(fields[0])
            learning_rate = float(fields[1])
        except ValueError as error:
            raise ValueError(f"invalid learning-rate segment: {item!r}") from error
        if end_epoch < 1 or learning_rate <= 0.0:
            raise ValueError("schedule endpoints and learning rates must be positive")
        if endpoints and end_epoch <= endpoints[-1][0]:
            raise ValueError("schedule endpoints must be strictly increasing")
        endpoints.append((end_epoch, learning_rate))

    if endpoints[-1][0] != target_epochs:
        raise ValueError(
            f"final schedule endpoint must equal target-epochs ({target_epochs}), "
            f"got {endpoints[-1][0]}"
        )

    segments: list[dict[str, float | int]] = []
    start_epoch = 1
    for end_epoch, learning_rate in endpoints:
        segments.append(
            {"start": start_epoch, "end": end_epoch, "learning_rate": learning_rate}
        )
        start_epoch = end_epoch + 1
    return segments


def canonical_schedule_specification(segments: list[dict[str, float | int]]) -> str:
    return ",".join(
        f"{int(segment['end'])}:{float(segment['learning_rate']):g}"
        for segment in segments
    )

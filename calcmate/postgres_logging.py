from __future__ import annotations

from typing import Protocol

from calcmate.models import AttemptLog


class AttemptLogger(Protocol):
    def log_attempt(self, log: AttemptLog) -> None:
        ...


class PostgresAttemptLogger:
    """PostgreSQL logging boundary.

    This intentionally does not open a database connection yet. It defines the
    runtime contract used by the API layer; wiring psycopg/SQLAlchemy can happen
    without changing the solving pipeline.
    """

    def log_attempt(self, log: AttemptLog) -> None:
        raise NotImplementedError("Configure a PostgreSQL client before using PostgresAttemptLogger.")


class NoopAttemptLogger:
    def __init__(self) -> None:
        self.logs: list[AttemptLog] = []

    def log_attempt(self, log: AttemptLog) -> None:
        self.logs.append(log)


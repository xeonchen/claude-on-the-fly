"""Tap provenance: a durable record of every counted suggestion-button press.

The Slack frontend appends one JSON line per counted press to the press ledger,
`DATA_DIR/state/slack.button-presses.jsonl`, before the press is dispatched. The
ledger does two jobs:

- It is the durable half of "one press, one message": a press whose
  `(channel, message_ts, block_id)` is already in the ledger is a duplicate, also
  after a restart has emptied the in-memory spent set.
- It is evidence a reader outside this process can check: who pressed which
  button on which message, as Slack's `block_actions` payload told us.

Beside it, the frontend writes a start-up marker naming the process that is
recording (its pid and the OS's own start time for it, read with `ps`), so a reader can tell the
ledger belongs to the bridge that is running now and not to one that was rolled
back. Nothing here reads configuration: the feature has no setting.

A press that is not counted leaves one structured log line (`drop_line`), one
reason per line, so another tool can find and explain it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_NAME = "slack.button-presses.jsonl"
MARKER_NAME = "slack.button-presses.marker.json"
LEDGER_VERSION = 1
MARKER_VERSION = 1

# Counted presses one presser may make inside the window. A person answering a
# menu presses once; six a minute leaves room for a quick run of answers and
# stops a stuck client or a script from flooding the agent with turns.
RATE_LIMIT = 6
RATE_WINDOW_SECONDS = 60.0

DROP_REASONS = ("duplicate", "rate-limited", "ledger-write-failed", "retire-failed")

Key = tuple[str, str, str]

# Characters that would let one drop-line field end early or start a new line.
_FIELD_UNSAFE = re.compile(r"[\[\]\r\n]")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class PressLedger:
    """The append-only press ledger at one path.

    Keys are read from the file once, on first use, and then kept in step with
    every append this process makes; a press recorded by a previous process is
    therefore still a duplicate.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._keys: set[Key] | None = None

    def ensure(self) -> None:
        """Create the ledger empty if it is absent; never truncate it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            return
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.path.parent)

    def _load(self) -> set[Key]:
        keys: set[Key] = set()
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return keys
        except OSError as exc:
            # Read as empty, loudly: a press after this may count twice, which
            # the reader downstream sees as two ledger lines for one key.
            logger.warning("press ledger: cannot read %s: %s", self.path, exc)
            return keys
        for line in text.splitlines():
            try:
                row = json.loads(line)
                keys.add((row["channel"], row["message_ts"], row["block_id"]))
            except (ValueError, TypeError, KeyError):
                logger.warning("press ledger: skipping malformed line in %s", self.path)
        return keys

    def has(self, key: Key) -> bool:
        if self._keys is None:
            self._keys = self._load()
        return key in self._keys

    def append(self, record: dict) -> None:
        """Append one press as one JSON line and fsync it.

        Raises ValueError for a record without `action_ts` (a reader orders a
        press after the message it answers by that field) and OSError when the
        write fails; in both cases no key is recorded.
        """
        if not record.get("action_ts"):
            raise ValueError("press has no action_ts")
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        if self._keys is None:
            self._keys = self._load()
        self._keys.add((record["channel"], record["message_ts"], record["block_id"]))


_ISO_SECONDS = "%Y-%m-%dT%H:%M:%SZ"


def process_started_at(pid: int) -> str:
    """The OS's start time for `pid`, UTC, to the second, as `ps` reports it.

    Read with `ps -o lstart=` under `TZ=UTC0` and `LC_ALL=C`, so a reader that
    asks `ps` the same way gets the same string for the same process. Raises
    OSError when it cannot be read or parsed.
    """
    env = {**os.environ, "TZ": "UTC0", "LC_ALL": "C"}
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            env=env,
        ).stdout.strip()
        started = datetime.strptime(out, "%a %b %d %H:%M:%S %Y")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise OSError(f"cannot read the start time of pid {pid}: {exc}") from exc
    return started.strftime(_ISO_SECONDS)


def write_marker(path: Path, *, ledger_name: str) -> None:
    """Write the start-up marker for this process, replacing any older one whole.

    `process_started_at` is the OS's start time for this pid (see
    `process_started_at`), so a reader compares it with what the OS reports for
    the pid now: a pid reused by another process, or a bridge restarted since,
    does not match. Raises OSError, and writes nothing, when the start time
    cannot be read.
    """
    pid = os.getpid()
    marker = {
        "v": MARKER_VERSION,
        "pid": pid,
        "process_started_at": process_started_at(pid),
        "written_at": datetime.now(UTC).strftime(_ISO_SECONDS),
        "ledger": ledger_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


class PressRateLimit:
    """At most `limit` counted presses per presser in any `window` seconds.

    Only presses that were let through count, so a refused press never extends
    the wait.
    """

    def __init__(
        self,
        limit: int = RATE_LIMIT,
        window: float = RATE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._seen: dict[str, deque[float]] = {}

    def allow(self, presser: str) -> bool:
        now = self._clock()
        seen = self._seen.setdefault(presser, deque())
        while seen and now - seen[0] >= self._window:
            seen.popleft()
        if len(seen) >= self._limit:
            return False
        seen.append(now)
        return True


def _field(value: str) -> str:
    return _FIELD_UNSAFE.sub("_", value) if value else "-"


def drop_line(
    reason: str,
    *,
    channel: str,
    message_ts: str,
    block_id: str,
    action_ts: str,
    user: str,
) -> str:
    """The one log line a press that does not count leaves, one reason each."""
    if reason not in DROP_REASONS:
        raise ValueError(f"unknown drop reason: {reason}")
    return (
        f"(button {reason}) [channel: {_field(channel)}] "
        f"[menu-ts: {_field(message_ts)}] [block-id: {_field(block_id)}] "
        f"[action-ts: {_field(action_ts)}] [user: {_field(user)}]"
    )

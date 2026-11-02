"""An operator's deny rule for suggestion-button labels.

`suggestions.deny_file` names a JSON file, `{"patterns": [regex, ...], "why": "..."}`.
A label that matches any pattern is never drawn as a button (the offer side, in the
orchestrator's parser) and never dispatched when tapped (the tap side, in each
frontend's button handler). The tap side is the guard that matters: it is the one
point every button passes through, whoever posted it.

The key is read straight from the `suggestions` section rather than through
`settings.get`: a path needs no environment-variable twin, and `settings.get`
flattens values to strings. The file is read at first use and again whenever its
mtime or size changes, so saving it is enough; no restart is needed.

A configured file that cannot be used -- missing, unreadable, not JSON, not the
expected shape, an empty pattern list, a pattern that does not compile, or a path
that is not absolute --
fails closed on the tap side (every tap is refused) and open on the offer side
(nothing is dropped, so the operator still sees the menu that taps are refused on).
Each such state is logged at ERROR once, when it is first seen. No key at all means
the feature is off and both sides behave exactly as they did before it existed.

Every line this module writes starts with `suggestion-deny: `, so one grep counts
every drop and every refusal.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from claude_on_the_fly import settings

logger = logging.getLogger(__name__)

TAG = "suggestion-deny"

# Posted in the thread when a tap is refused. True in both causes (a rule matched,
# or the rule file could not be read), and it gives no advice beyond typing.
REFUSAL_TEXT = (
    "Not sent: this button is disabled by a deny rule (or the rule file could not "
    "be read). Type your reply instead."
)

# Why a tap was refused; `tap_refusal` returns one of these, or None.
MATCHED = "matched a deny pattern"
UNREADABLE = "the deny file is unreadable"

# Hyphen-like characters (U+2010..2015, minus sign, small and full-width hyphen-
# minus) and the underscore all count as a plain hyphen, so a pattern written with
# `-` also matches a label typed or rendered with any of them.
_FOLD = str.maketrans(
    dict.fromkeys(
        "‐‑‒–—―−﹘﹣－_",
        "-",
    )
)


def normalise(label: str) -> str:
    """The form patterns are matched against: NFKC, then hyphen folding."""
    return unicodedata.normalize("NFKC", label).translate(_FOLD)


@dataclass(frozen=True)
class Rules:
    """The rule as it stands right now."""

    configured: bool
    patterns: tuple[re.Pattern[str], ...] = ()
    # Non-empty when the configured file cannot be used.
    error: str = ""

    @property
    def readable(self) -> bool:
        return self.configured and not self.error

    def matches(self, label: str) -> bool:
        folded = normalise(label)
        return any(pattern.search(folded) for pattern in self.patterns)


_OFF = Rules(configured=False)

# The last state seen: (key, rules). The key is the path plus the file's
# (mtime_ns, size) for a readable file, or the path plus the error for an
# unusable one, so a repeat of the same state is neither re-parsed nor re-logged.
_LAST: tuple[tuple[object, ...], Rules] | None = None


def _configured_value() -> object:
    merged = {**settings.bundled("suggestions"), **settings.operator("suggestions")}
    return merged.get("deny_file")


def _load(path: Path) -> Rules:
    """Parse and compile one rule file. Never raises; a problem is an error."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return Rules(configured=True, error=f"cannot read {path}: {exc}")
    patterns = document.get("patterns") if isinstance(document, dict) else None
    if not isinstance(patterns, list) or not all(
        isinstance(item, str) for item in patterns
    ):
        return Rules(
            configured=True,
            error=f"{path} must be a JSON object whose `patterns` is a list of strings",
        )
    if not patterns:
        # An empty list is not "deny nothing": a rule file that exists to deny
        # and denies nothing is almost always an edit gone wrong, so it is
        # treated like any other unusable file rather than silently off.
        return Rules(configured=True, error=f"{path}: `patterns` is empty")
    compiled: list[re.Pattern[str]] = []
    for item in patterns:
        try:
            compiled.append(re.compile(item))
        except re.error as exc:
            return Rules(
                configured=True,
                error=f"{path}: pattern {item!r} does not compile ({exc})",
            )
    return Rules(configured=True, patterns=tuple(compiled))


def _remember(key: tuple[object, ...], rules: Rules) -> Rules:
    global _LAST
    _LAST = (key, rules)
    if rules.error:
        logger.error(
            "%s: suggestions.deny_file is unreadable (%s); every suggestion tap is "
            "refused and no offered label is dropped until it is fixed",
            TAG,
            rules.error,
        )
    else:
        logger.info(
            "%s: loaded %d pattern(s) from %s",
            TAG,
            len(rules.patterns),
            key[0],
        )
    return rules


def current() -> Rules:
    """The rule now: off, readable, or configured-but-unreadable."""
    raw = _configured_value()
    if raw is None:
        return _OFF
    if not isinstance(raw, str) or not os.path.isabs(raw):
        rules = Rules(
            configured=True,
            error=f"deny_file must be an absolute path, got {raw!r}",
        )
        key: tuple[object, ...] = (repr(raw), rules.error)
    else:
        path = Path(raw)
        try:
            stat = path.stat()
        except OSError as exc:
            rules = Rules(configured=True, error=f"cannot read {path}: {exc}")
            key = (raw, rules.error)
        else:
            key = (raw, stat.st_mtime_ns, stat.st_size)
            if _LAST is not None and _LAST[0] == key:
                return _LAST[1]
            rules = _load(path)
            if rules.error:
                key = (raw, rules.error)
    if _LAST is not None and _LAST[0] == key:
        return _LAST[1]
    return _remember(key, rules)


def offer_filter(labels: list[str]) -> list[str]:
    """Labels with every denied one removed. Unreadable or off: unchanged."""
    rules = current()
    if not rules.readable:
        return labels
    kept: list[str] = []
    for label in labels:
        if rules.matches(label):
            logger.warning("%s: offer dropped %r", TAG, label)
        else:
            kept.append(label)
    return kept


def tap_refusal(label: str) -> str | None:
    """Why a tap on `label` must not be dispatched, or None to dispatch it."""
    rules = current()
    if not rules.configured:
        return None
    if rules.error:
        return UNREADABLE
    return MATCHED if rules.matches(label) else None

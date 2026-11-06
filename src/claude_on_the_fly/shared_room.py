"""What a reply may say in a room somebody other than the owner can read.

Pure: text and room facts in, a decision out. The Slack adapter supplies the
facts (who is in the room, which channel is the owner's DM) and acts on the
answer (post the reply, or post `HELD_REPLY` and send the original to the owner).

What counts as internal is the operator's, never this module's: the names come
from `slack.shared_room_hold_names`, the id shapes from
`slack.shared_room_hold_patterns`, and the people who merely share a name with
one of them from `slack.shared_room_people`. Nothing configured holds nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# The one line a held reply leaves in the room. Fixed on purpose: anything the
# agent wrote is what is being held back.
HELD_REPLY = "收到，處理中，有結果會再回覆。"

# A name is a whole token. Not `\b`: CJK characters are word characters to `\b`,
# so 「交給Borealis處理」 has no boundary around the name at all. Only an ASCII
# letter, digit or underscore next to it makes it part of a longer word.
_EDGE_BEFORE = r"(?<![A-Za-z0-9_])"
_EDGE_AFTER = r"(?![A-Za-z0-9_])"

# A link into a channel: a web permalink (`/archives/<id>`), the web client
# (`app.slack.com/client/<team>/<id>`), or an app link (`slack://channel?...id=<id>`,
# whose `&` Slack may send escaped as `&amp;`). Only the id is captured.
_CHANNEL_LINK = re.compile(
    r"(?:/archives/|/client/T[A-Z0-9]+/|[?&;]id=)([CDG][A-Z0-9]{2,})"
)


@dataclass(frozen=True)
class RoomFacts:
    """`shared`: somebody besides the owner (and the bot) can read the room.
    `owner_dm_channels`: the owner's 1:1 DM channel ids, empty if unknown."""

    shared: bool
    owner_dm_channels: frozenset[str]


@dataclass(frozen=True)
class HoldRules:
    names: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]

    @property
    def enabled(self) -> bool:
        return bool(self.names or self.patterns)

    @classmethod
    def build(
        cls,
        names: Iterable[str],
        patterns: Iterable[str],
        people: Iterable[str],
    ) -> tuple[HoldRules, tuple[str, ...]]:
        """Clean the configured lists. Returns the rules and one message per
        pattern that would not compile; the rest still apply, and the caller logs
        the messages, because a broken pattern silently matching nothing is the
        failure an operator would never notice."""
        exempt = {p.strip() for p in people if p.strip()}
        kept = tuple(n for n in (raw.strip() for raw in names) if n and n not in exempt)
        compiled: list[re.Pattern[str]] = []
        errors: list[str] = []
        for raw in patterns:
            source = raw.strip()
            if not source:
                continue
            try:
                compiled.append(re.compile(source))
            except re.error as exc:
                errors.append(f"{source!r}: {exc}")
        return cls(kept, tuple(compiled)), tuple(errors)


def is_shared(members: set[str] | None, owners: frozenset[str], own_id: str) -> bool:
    """Whether anyone besides the owner and the bot itself can read the room.

    `members` None means membership could not be read: that counts as shared,
    because the cost of a wrong "private" is the leak this check exists for."""
    if members is None:
        return True
    return bool(members - owners - {own_id})


def check(text: str, rules: HoldRules, room: RoomFacts) -> str | None:
    """The reason to hold `text` out of `room`, or None to post it.

    Names are case-sensitive whole tokens: "Desk" the system holds, "help desk"
    and "Desktop" do not, and neither does "spring" the season."""
    if not room.shared:
        return None
    for name in rules.names:
        if re.search(_EDGE_BEFORE + re.escape(name) + _EDGE_AFTER, text):
            return f"name `{name}`"
    for pattern in rules.patterns:
        found = pattern.search(text)
        if found:
            return f"id `{found.group(0)}`"
    if not rules.enabled:
        return None
    for link in _CHANNEL_LINK.finditer(text):
        channel = link.group(1)
        if channel in room.owner_dm_channels or (
            not room.owner_dm_channels and channel.startswith("D")
        ):
            return f"link into the owner's DM `{channel}`"
    return None


def held_notice(room_label: str, reason: str, original: str) -> str:
    """What the owner's DM gets: one line saying where and why, then the reply."""
    return (
        f"_(Held from {room_label}, which someone besides you can read: "
        f"matched {reason}. The room got 「{HELD_REPLY}」 instead.)_\n\n{original}"
    )


def prompt_line(rules: HoldRules) -> str:
    """The one line the session's context gets in such a room. Empty when no
    rules are configured, so a deployment that never set them sees no change."""
    if not rules.enabled:
        return ""
    names = ", ".join(rules.names)
    listed = f" ({names})" if names else ""
    return (
        "Someone besides the owner can read this conversation: do not name "
        f"internal agents or systems{listed}, internal ticket ids, or link "
        "anything a member here cannot open, such as the owner's DM. Say what "
        "will happen and who acts in the reader's own terms, and a request to a "
        "person here uses a real Slack mention of them (<@USERID>)."
    )

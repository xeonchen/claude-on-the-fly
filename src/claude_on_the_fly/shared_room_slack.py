"""Who can read a Slack room: the Slack edge of the shared-room check.

`shared_room` decides from facts; this reads them from Slack. The chat frontend
and the job worker (a separate process with its own client) both build one, so
"shared" means the same thing to a reply and to a job result.

Shared means a person besides an owner can read the room:
- a public channel always is, whoever has joined: anyone in the workspace can
  open it;
- otherwise the members are read, bots are dropped (ours and everyone else's,
  since a bot is not a reader), and any human left who is not an owner makes it
  shared.
Anything Slack will not answer counts as shared. A wrong "private" is the leak
this exists to stop; a wrong "shared" costs one held reply.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from claude_on_the_fly import checks, settings, shared_room

logger = logging.getLogger(__name__)

# Slackbot's fixed id. users.info does not mark it `is_bot`.
_SLACKBOT = "USLACKBOT"
_MEMBERS_TTL = 60.0
# The same shape `slack._SLACK_ID` accepts for the owner list.
_SLACK_ID = re.compile(r"[UWB][A-Z0-9_]+")


# Hold-pattern errors already logged, so a broken pattern is one ERROR per
# distinct message per process rather than one per reply.
_logged_hold_errors: set[str] = set()


def _setting_list(name: str, sep: str) -> list[str]:
    """A list-valued setting, split on the separator its `Field` joins with."""
    return [item for item in settings.get(name, "").split(sep) if item.strip()]


def hold_rules_from_settings() -> shared_room.HoldRules:
    """The configured hold rules, re-read live like every other sender list."""
    rules, errors = shared_room.HoldRules.build(
        _setting_list("SLACK_SHARED_ROOM_HOLD_NAMES", ","),
        _setting_list("SLACK_SHARED_ROOM_HOLD_PATTERNS", "\n"),
        _setting_list("SLACK_SHARED_ROOM_PEOPLE", ","),
    )
    for error in errors:
        if error not in _logged_hold_errors:
            _logged_hold_errors.add(error)
            logger.error("slack.shared_room_hold_patterns: skipping %s", error)
    return rules


def owner_ids_from_settings() -> frozenset[str]:
    """`slack.owner_senders` as usable Slack ids, for a process with no frontend.

    Empty when the list is absent or unusable, which makes every room with a
    human in it shared: the fail-closed reading."""
    ids = checks.resolve_slack_ids(settings.environment(), "SLACK_OWNER_SENDER_IDS")
    return frozenset(i for i in ids if _SLACK_ID.fullmatch(i))


class SlackRooms:
    def __init__(
        self,
        client: Any,
        *,
        owners: Callable[[], frozenset[str]],
        own_id: Callable[[], str] | None = None,
    ) -> None:
        self._client = client
        self._owners = owners
        self._own_id_given = own_id
        self._own_id: str | None = None
        self._kinds: dict[str, str] = {}
        self._members: dict[str, tuple[frozenset[str], float]] = {}
        self._bots: dict[str, bool] = {}
        self._dms: dict[str, str] = {}

    async def _self_id(self) -> str | None:
        if self._own_id_given is not None:
            return self._own_id_given()
        if self._own_id is None:
            try:
                resp = await self._client.auth_test()
                self._own_id = resp["user_id"]
            except Exception as exc:
                logger.warning("shared_room: auth.test failed: %s", exc)
                return None
        return self._own_id

    async def _kind(self, channel: str) -> str:
        """ "im", "mpim", "group" (private channel), "channel" (public), or "" if
        Slack would not say. Only a definitive answer is cached."""
        cached = self._kinds.get(channel)
        if cached:
            return cached
        try:
            ch = (await self._client.conversations_info(channel=channel))["channel"]
        except Exception as exc:
            logger.warning("shared_room: type of %s unreadable: %s", channel, exc)
            return ""
        if ch.get("is_im"):
            kind = "im"
        elif ch.get("is_mpim"):
            kind = "mpim"
        elif ch.get("is_private") or ch.get("is_group"):
            kind = "group"
        else:
            kind = "channel"
        self._kinds[channel] = kind
        return kind

    async def _member_ids(self, channel: str) -> frozenset[str] | None:
        cached = self._members.get(channel)
        now = time.monotonic()
        if cached is not None and cached[1] > now:
            return cached[0]
        members: set[str] = set()
        cursor = ""
        try:
            while True:
                resp = await self._client.conversations_members(
                    channel=channel, cursor=cursor or None, limit=200
                )
                members.update(resp.get("members", []))
                cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
                if not cursor:
                    break
        except Exception as exc:
            logger.warning("shared_room: members of %s unreadable: %s", channel, exc)
            return None
        result = frozenset(members)
        self._members[channel] = (result, now + _MEMBERS_TTL)
        return result

    async def _is_bot(self, user: str) -> bool:
        """Whether `user` is a bot. Cached for good once known; a failed lookup
        counts as a human and is not cached."""
        if user == _SLACKBOT:
            return True
        known = self._bots.get(user)
        if known is not None:
            return known
        try:
            info = (await self._client.users_info(user=user))["user"]
        except Exception as exc:
            logger.warning("shared_room: users.info %s failed: %s", user, exc)
            return False
        is_bot = bool(info.get("is_bot"))
        self._bots[user] = is_bot
        return is_bot

    async def is_shared(self, channel: str) -> bool:
        kind = await self._kind(channel)
        if kind in ("", "channel"):
            return True
        members = await self._member_ids(channel)
        if members is None:
            return True
        owners = self._owners()
        own = await self._self_id()
        candidates = members - owners - ({own} if own else set())
        humans = {m for m in candidates if not await self._is_bot(m)}
        return shared_room.is_shared(frozenset(humans), owners, own or "")

    async def owner_dm_channels(self) -> frozenset[str]:
        found: set[str] = set()
        for owner in self._owners():
            dm = self._dms.get(owner)
            if dm is None:
                try:
                    resp = await self._client.conversations_open(users=owner)
                    dm = resp["channel"]["id"]
                except Exception as exc:
                    logger.error("shared_room: cannot open DM with %s: %s", owner, exc)
                    continue
                self._dms[owner] = dm
            found.add(dm)
        return frozenset(found)

    async def facts(self, channel: str) -> shared_room.RoomFacts | None:
        """The facts to check a post into `channel` with, or None when only an
        owner can read it."""
        if not await self.is_shared(channel):
            return None
        return shared_room.RoomFacts(
            shared=True, owner_dm_channels=await self.owner_dm_channels()
        )

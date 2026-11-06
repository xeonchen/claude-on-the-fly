"""SlackRooms: who can read a Slack room, for the shared-room check.

The chat frontend and the job worker both ask it; the matching rules are
test_shared_room.py's.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from slack_sdk.errors import SlackApiError

from claude_on_the_fly.shared_room_slack import SlackRooms, owner_ids_from_settings

BOTS = {"U_BOT", "U_ALERTBOT"}


def _client(kind: dict, members: list[str]) -> AsyncMock:
    client = AsyncMock()
    client.conversations_info = AsyncMock(return_value={"channel": kind})
    client.conversations_members = AsyncMock(return_value={"members": members})

    async def users_info(*, user: str):
        return {"user": {"id": user, "is_bot": user in BOTS}}

    client.users_info = AsyncMock(side_effect=users_info)
    client.conversations_open = AsyncMock(return_value={"channel": {"id": "D_OWN"}})
    client.auth_test = AsyncMock(return_value={"user_id": "U_BOT"})
    return client


PRIVATE = {"is_private": True, "is_group": True}
MPIM = {"is_mpim": True}
PUBLIC = {"is_private": False}
IM = {"is_im": True}


def _rooms(client, owners=frozenset({"U_OWN"}), own_id="U_BOT") -> SlackRooms:
    return SlackRooms(client, owners=lambda: owners, own_id=lambda: own_id)


class TestIsShared:
    async def test_owner_bot_and_another_bot_is_not_shared(self):
        """A room whose readers are the owner and bots is the owner's alone."""
        client = _client(PRIVATE, ["U_OWN", "U_BOT", "U_ALERTBOT"])
        assert await _rooms(client).is_shared("G1") is False

    async def test_a_colleague_makes_it_shared(self):
        client = _client(MPIM, ["U_OWN", "U_BOT", "U_COLLEAGUE"])
        assert await _rooms(client).is_shared("G1") is True

    async def test_a_colleagues_dm_with_the_bot_is_shared(self):
        client = _client(IM, ["U_BOT", "U_COLLEAGUE"])
        assert await _rooms(client).is_shared("D1") is True

    async def test_the_owners_dm_is_not_shared(self):
        client = _client(IM, ["U_BOT", "U_OWN"])
        assert await _rooms(client).is_shared("D_OWN") is False

    async def test_a_public_channel_is_shared_whoever_joined(self):
        """Anyone in the workspace can read a public channel without joining."""
        client = _client(PUBLIC, ["U_OWN", "U_BOT"])
        assert await _rooms(client).is_shared("C1") is True
        client.conversations_members.assert_not_awaited()

    async def test_an_unreadable_room_type_is_shared(self):
        client = _client(PRIVATE, ["U_OWN", "U_BOT"])
        client.conversations_info.side_effect = Exception("boom")
        assert await _rooms(client).is_shared("G1") is True

    async def test_unreadable_members_are_shared_and_not_cached(self):
        client = _client(PRIVATE, ["U_OWN", "U_BOT"])
        client.conversations_members.side_effect = Exception("boom")
        rooms = _rooms(client)
        assert await rooms.is_shared("G1") is True
        client.conversations_members.side_effect = None
        assert await rooms.is_shared("G1") is False

    async def test_an_unreadable_user_counts_as_a_human(self):
        client = _client(PRIVATE, ["U_OWN", "U_BOT", "U_ALERTBOT"])
        client.users_info.side_effect = SlackApiError("x", {"error": "user_not_found"})
        assert await _rooms(client).is_shared("G1") is True

    async def test_slackbot_is_a_bot(self):
        client = _client(PRIVATE, ["U_OWN", "U_BOT", "USLACKBOT"])
        assert await _rooms(client).is_shared("G1") is False

    async def test_no_owner_makes_any_human_a_reader(self):
        client = _client(PRIVATE, ["U_OWN", "U_BOT"])
        assert await _rooms(client, owners=frozenset()).is_shared("G1") is True

    async def test_members_are_cached_briefly_and_bots_for_good(self):
        client = _client(PRIVATE, ["U_OWN", "U_BOT", "U_ALERTBOT"])
        rooms = _rooms(client)
        await rooms.is_shared("G1")
        await rooms.is_shared("G1")
        assert client.conversations_members.await_count == 1
        assert client.conversations_info.await_count == 1
        rooms._members.clear()
        await rooms.is_shared("G1")
        assert client.users_info.await_count == 1

    async def test_member_pages_are_followed(self):
        client = _client(PRIVATE, [])
        client.conversations_members.side_effect = [
            {"members": ["U_OWN"], "response_metadata": {"next_cursor": "c"}},
            {"members": ["U_COLLEAGUE"], "response_metadata": {"next_cursor": ""}},
        ]
        assert await _rooms(client).is_shared("G1") is True
        assert client.conversations_members.await_args_list[1].kwargs["cursor"] == "c"

    async def test_own_id_comes_from_auth_test_when_not_given(self):
        client = _client(PRIVATE, ["U_OWN", "U_BOT"])
        client.users_info.side_effect = None
        client.users_info.return_value = {"user": {"is_bot": False}}
        rooms = SlackRooms(client, owners=lambda: frozenset({"U_OWN"}))
        assert await rooms.is_shared("G1") is False
        assert await rooms.is_shared("G2") is False
        client.auth_test.assert_awaited_once()

    async def test_a_failed_auth_test_leaves_no_own_id(self):
        client = _client(PRIVATE, ["U_OWN", "U_SELF"])
        client.users_info.side_effect = None
        client.users_info.return_value = {"user": {"is_bot": False}}
        client.auth_test.side_effect = Exception("boom")
        rooms = SlackRooms(client, owners=lambda: frozenset({"U_OWN"}))
        assert await rooms.is_shared("G1") is True


class TestOwnerDms:
    async def test_owner_dm_channels_are_opened_and_cached(self):
        client = _client(IM, [])
        rooms = _rooms(client)
        assert await rooms.owner_dm_channels() == {"D_OWN"}
        assert await rooms.owner_dm_channels() == {"D_OWN"}
        client.conversations_open.assert_awaited_once_with(users="U_OWN")

    async def test_an_unopenable_dm_is_left_out(self):
        client = _client(IM, [])
        client.conversations_open.side_effect = Exception("boom")
        assert await _rooms(client).owner_dm_channels() == frozenset()

    async def test_facts_are_none_for_a_private_room(self):
        client = _client(IM, ["U_BOT", "U_OWN"])
        assert await _rooms(client).facts("D_OWN") is None

    async def test_facts_carry_the_owner_dms_for_a_shared_room(self):
        client = _client(MPIM, ["U_BOT", "U_OWN", "U_COLLEAGUE"])
        facts = await _rooms(client).facts("G1")
        assert facts is not None
        assert facts.shared
        assert facts.owner_dm_channels == {"D_OWN"}


class TestOwnerIdsFromSettings:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("U1,U2", {"U1", "U2"}), ("", set()), ("*,U1", {"U1"})],
    )
    def test_usable_ids_only(self, monkeypatch, raw, expected):
        monkeypatch.setenv("SLACK_OWNER_SENDER_IDS", raw)
        assert owner_ids_from_settings() == expected

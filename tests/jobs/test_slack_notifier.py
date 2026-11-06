"""SlackThreadNotifier: posts into origin's thread, reads only channel/thread_ts,
splits long bodies, marks failures, and reports post errors to its caller."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from claude_on_the_fly.jobs.core import Result
from claude_on_the_fly.jobs.slack_notifier import (
    FALLBACK_TEXT_LIMIT,
    SLACK_MAX_BLOCKS,
    SlackThreadNotifier,
    _blocks,
)
from claude_on_the_fly.slack_mrkdwn import SLACK_MARKDOWN_LIMIT, split_blocks


def _client() -> AsyncMock:
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1.0"})
    return client


async def test_posts_result_into_origin_thread() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    origin = {"channel": "C123", "thread_ts": "1699.5", "sender_id": "U9"}
    await notifier.notify(origin, Result(ok=True, text="the answer"))

    client.chat_postMessage.assert_awaited_once()
    kwargs = client.chat_postMessage.call_args.kwargs
    assert kwargs["channel"] == "C123"
    assert kwargs["thread_ts"] == "1699.5"
    assert kwargs["text"] == "the answer"
    assert kwargs["blocks"][0] == {"type": "markdown", "text": "the answer"}


async def test_reads_only_channel_and_thread_ts_never_sender_id() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    await notifier.notify(
        {"channel": "C1", "thread_ts": "9.9", "sender_id": "U-secret"},
        Result(ok=True, text="ok"),
    )
    kwargs = client.chat_postMessage.call_args.kwargs
    # sender_id is never forwarded to Slack.
    assert "U-secret" not in str(kwargs)


async def test_missing_thread_ts_posts_to_channel_root() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    await notifier.notify({"channel": "C1"}, Result(ok=True, text="ok"))
    assert client.chat_postMessage.call_args.kwargs["thread_ts"] is None


async def test_failure_result_is_marked() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    await notifier.notify({"channel": "C1"}, Result(ok=False, text="stack trace"))
    kwargs = client.chat_postMessage.call_args.kwargs
    # The notification line marks the failure; the detail rides in the blocks,
    # which are what Slack actually renders.
    assert "Job failed" in kwargs["text"]
    assert "stack trace" in kwargs["blocks"][0]["text"]


async def test_failure_drops_a_codex_tui_capture() -> None:
    """The origin thread can be a public channel, and the start screen names
    the host's workspace path and its permission mode."""
    client = _client()
    notifier = SlackThreadNotifier(client)
    text = (
        "Job failed: >_ OpenAI Codex (v0.163.0)loading›Ask Codex to do "
        "anything?forshortcuts ~/.claude-on-the-fly/workspaces/slack/channel/C0"
        "permissions: YOLO modeError: account/read failed"
    )
    await notifier.notify({"channel": "C1"}, Result(ok=False, text=text))
    rendered = client.chat_postMessage.call_args.kwargs["blocks"][0]["text"]
    assert "Error: account/read failed" in rendered
    for leak in ("workspaces", "YOLO", "OpenAI Codex"):
        assert leak not in rendered


async def test_failure_drops_the_codex_exec_banner() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    text = (
        "Job failed: OpenAI Codex v0.154.0\n--------\nworkdir: /Users/h/ws\n"
        "sandbox: danger-full-access\n--------\nERROR: boom"
    )
    await notifier.notify({"channel": "C1"}, Result(ok=False, text=text))
    rendered = client.chat_postMessage.call_args.kwargs["blocks"][0]["text"]
    assert "ERROR: boom" in rendered
    for leak in ("workdir", "/Users/h/ws", "danger-full-access"):
        assert leak not in rendered


async def test_no_channel_skips_post() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    await notifier.notify({"thread_ts": "1.0"}, Result(ok=True, text="ok"))
    client.chat_postMessage.assert_not_awaited()


async def test_long_body_splits_into_multiple_blocks() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    body = "\n".join(["x" * 5000 for _ in range(5)])  # ~25k chars
    await notifier.notify({"channel": "C1"}, Result(ok=True, text=body))
    blocks = client.chat_postMessage.call_args.kwargs["blocks"]
    assert len(blocks) > 1
    for block in blocks:
        assert len(block["text"]) <= SLACK_MARKDOWN_LIMIT


async def test_post_failure_propagates() -> None:
    """Returning normally is what marks a result delivered, so a swallowed
    failure would turn a retryable miss into a reply nobody ever receives.
    The worker decides whether to retry; the adapter only reports."""
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(side_effect=RuntimeError("network down"))
    notifier = SlackThreadNotifier(client)

    with pytest.raises(RuntimeError, match="network down"):
        await notifier.notify({"channel": "C1"}, Result(ok=True, text="ok"))


def test_single_over_limit_line_reassembles_with_zero_loss() -> None:
    # One unbroken line (no newlines) well over the limit — the JSON-blob case.
    # The old splitter truncated it to line[:LIMIT] and dropped the
    # tail; every character must now survive across the produced chunks.
    line = "J" * (SLACK_MARKDOWN_LIMIT * 2 + 137)
    chunks = split_blocks(line)
    assert len(chunks) == 3  # 12000 + 12000 + 137
    assert all(len(c) <= SLACK_MARKDOWN_LIMIT for c in chunks)
    assert "".join(chunks) == line  # nothing dropped


async def test_over_limit_line_delivered_without_content_loss() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    body = "K" * (SLACK_MARKDOWN_LIMIT * 3 + 50)
    await notifier.notify({"channel": "C1"}, Result(ok=True, text=body))
    blocks = client.chat_postMessage.call_args.kwargs["blocks"]
    assert len(blocks) > 1
    for block in blocks:
        assert len(block["text"]) <= SLACK_MARKDOWN_LIMIT
    reassembled = "".join(block["text"] for block in blocks)
    assert reassembled == body


async def test_result_over_block_budget_splits_into_multiple_messages() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)
    # Many separate over-limit lines → far more than SLACK_MAX_BLOCKS blocks, which
    # a single chat_postMessage would reject with `invalid_blocks`.
    body = "\n".join(
        "z" * (SLACK_MARKDOWN_LIMIT + 10) for _ in range(SLACK_MAX_BLOCKS + 5)
    )
    await notifier.notify({"channel": "C1"}, Result(ok=True, text=body))

    calls = client.chat_postMessage.call_args_list
    assert len(calls) >= 2  # split across multiple posted messages
    for call in calls:
        assert len(call.kwargs["blocks"]) <= SLACK_MAX_BLOCKS
    # Posted to the same origin, and no content lost across the messages.
    assert all(call.kwargs["channel"] == "C1" for call in calls)
    posted = "".join(block["text"] for call in calls for block in call.kwargs["blocks"])
    assert posted == body


def test_blocks_helper_never_empty() -> None:
    assert _blocks("") == [{"type": "markdown", "text": ""}]


async def test_fallback_text_stays_within_slacks_cap() -> None:
    """`text` is the notification string, not the body: Slack caps it at 40,000
    characters and rejects the whole call past that. Passing the full result
    there defeated the block batching exactly in the case it exists for — a
    reply big enough to need it — and the surrounding except swallowed the
    rejection, so nothing at all was delivered."""
    client = _client()
    notifier = SlackThreadNotifier(client)

    await notifier.notify({"channel": "C1"}, Result(ok=True, text="L" * 60_000))

    for call in client.chat_postMessage.call_args_list:
        assert len(call.kwargs["text"]) <= FALLBACK_TEXT_LIMIT


async def test_fallback_text_summarises_the_first_line() -> None:
    client = _client()
    notifier = SlackThreadNotifier(client)

    await notifier.notify(
        {"channel": "C1"}, Result(ok=True, text="Deploy finished\ndetail\nmore detail")
    )

    assert client.chat_postMessage.call_args.kwargs["text"] == "Deploy finished"


# ---------------------------------------------------------------------------
# Shared rooms: a job result is the session's own text landing in the room, so
# it gets the same check as a chat reply.
# ---------------------------------------------------------------------------

HELD = "收到，處理中，有結果會再回覆。"
HOLD_TEXT = "Opened 2026-11-06-c7d3 for Karina."


class _Rooms:
    """A stand-in for SlackRooms: the room is shared or it is not."""

    def __init__(self, shared: bool, dms: frozenset[str] = frozenset({"D_OWN"})):
        from claude_on_the_fly.shared_room import RoomFacts

        self._facts = RoomFacts(shared=True, owner_dm_channels=dms) if shared else None
        self.asked: list[str] = []

    async def facts(self, channel: str):
        self.asked.append(channel)
        return self._facts


@pytest.fixture
def hold_rules(monkeypatch):
    monkeypatch.setenv("SLACK_SHARED_ROOM_HOLD_NAMES", "Karina")
    monkeypatch.setenv(
        "SLACK_SHARED_ROOM_HOLD_PATTERNS",
        r"(?<![A-Za-z0-9_])20\d\d-\d\d-\d\d-[0-9a-f]{4}(?![A-Za-z0-9_])",
    )
    monkeypatch.delenv("SLACK_SHARED_ROOM_PEOPLE", raising=False)


def _posts(client) -> list[dict]:
    return [c.kwargs for c in client.chat_postMessage.await_args_list]


async def test_a_matching_result_in_a_shared_room_is_held(hold_rules) -> None:
    client = _client()
    notifier = SlackThreadNotifier(client, rooms=_Rooms(shared=True))
    await notifier.notify(
        {"channel": "G1", "thread_ts": "5.0"}, Result(ok=True, text=HOLD_TEXT)
    )
    room, dm = _posts(client)
    assert room == {"channel": "G1", "thread_ts": "5.0", "text": HELD}
    assert dm["channel"] == "D_OWN"
    assert "thread_ts" not in dm or dm["thread_ts"] is None
    assert "G1" in json.dumps(dm["blocks"], ensure_ascii=False)
    assert HOLD_TEXT in json.dumps(dm["blocks"], ensure_ascii=False)


async def test_a_failed_job_text_is_checked_too(hold_rules) -> None:
    client = _client()
    notifier = SlackThreadNotifier(client, rooms=_Rooms(shared=True))
    await notifier.notify({"channel": "G1"}, Result(ok=False, text="Karina crashed"))
    assert _posts(client)[0]["text"] == HELD


async def test_a_clean_result_in_a_shared_room_posts(hold_rules) -> None:
    client = _client()
    notifier = SlackThreadNotifier(client, rooms=_Rooms(shared=True))
    await notifier.notify({"channel": "G1"}, Result(ok=True, text="All done."))
    (post,) = _posts(client)
    assert post["text"] == "All done."


async def test_an_owner_only_room_is_untouched(hold_rules) -> None:
    client = _client()
    notifier = SlackThreadNotifier(client, rooms=_Rooms(shared=False))
    await notifier.notify({"channel": "D_OWN"}, Result(ok=True, text=HOLD_TEXT))
    (post,) = _posts(client)
    assert post["text"] == HOLD_TEXT


async def test_no_rules_reads_no_room(monkeypatch) -> None:
    monkeypatch.delenv("SLACK_SHARED_ROOM_HOLD_NAMES", raising=False)
    monkeypatch.delenv("SLACK_SHARED_ROOM_HOLD_PATTERNS", raising=False)
    rooms = _Rooms(shared=True)
    client = _client()
    await SlackThreadNotifier(client, rooms=rooms).notify(
        {"channel": "G1"}, Result(ok=True, text=HOLD_TEXT)
    )
    assert rooms.asked == []
    assert _posts(client)[0]["text"] == HOLD_TEXT


async def test_a_held_result_with_no_owner_dm_is_logged(hold_rules, caplog) -> None:
    client = _client()
    notifier = SlackThreadNotifier(client, rooms=_Rooms(shared=True, dms=frozenset()))
    with caplog.at_level("ERROR"):
        await notifier.notify({"channel": "G1"}, Result(ok=True, text=HOLD_TEXT))
    (post,) = _posts(client)
    assert post["text"] == HELD
    assert "no owner DM" in caplog.text


async def test_a_failed_held_line_raises_for_a_retry(hold_rules) -> None:
    client = _client()
    client.chat_postMessage.side_effect = RuntimeError("down")
    notifier = SlackThreadNotifier(client, rooms=_Rooms(shared=True))
    with pytest.raises(RuntimeError):
        await notifier.notify({"channel": "G1"}, Result(ok=True, text=HOLD_TEXT))


async def test_the_worker_wires_the_check_with_live_owners(monkeypatch) -> None:
    """`jobs.cli` builds the notifier with a room reader over the same client,
    taking owners from settings at each call."""
    from claude_on_the_fly.jobs import cli
    from claude_on_the_fly.shared_room_slack import SlackRooms

    client = _client()
    notifier = cli._slack_notifier(client)
    rooms = notifier._rooms
    assert isinstance(rooms, SlackRooms)
    assert rooms._client is client
    monkeypatch.setenv("SLACK_OWNER_SENDER_IDS", "U_OWN")
    assert rooms._owners() == {"U_OWN"}

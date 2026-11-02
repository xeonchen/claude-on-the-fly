"""The press ledger, its start-up marker and the per-presser rate limit."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime

import psutil
import pytest

from claude_on_the_fly import button_presses
from claude_on_the_fly.button_presses import (
    PressLedger,
    PressRateLimit,
    drop_line,
    write_marker,
)


def _record(**overrides) -> dict:
    record = {
        "channel": "C1",
        "message_ts": "99.1",
        "block_id": "menu-a",
        "action_ts": "100.2",
        "label": "alpha?",
    }
    record.update(overrides)
    return record


class TestPressLedger:
    def test_ensure_creates_an_empty_ledger(self, tmp_path):
        ledger = PressLedger(tmp_path / "state" / "presses.jsonl")
        ledger.ensure()
        assert ledger.path.read_text() == ""

    def test_ensure_keeps_an_existing_ledger_untouched(self, tmp_path):
        path = tmp_path / "presses.jsonl"
        path.write_text(json.dumps(_record()) + "\n")
        PressLedger(path).ensure()
        assert json.loads(path.read_text()) == _record()

    def test_an_appended_press_is_one_json_line(self, tmp_path):
        ledger = PressLedger(tmp_path / "presses.jsonl")
        ledger.append(_record(label="two\nlines [menu-ts: 1.0]"))
        lines = ledger.path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["label"] == "two\nlines [menu-ts: 1.0]"

    def test_an_appended_press_is_seen_as_a_key(self, tmp_path):
        ledger = PressLedger(tmp_path / "presses.jsonl")
        assert not ledger.has(("C1", "99.1", "menu-a"))
        ledger.append(_record())
        assert ledger.has(("C1", "99.1", "menu-a"))

    def test_keys_survive_a_new_reader(self, tmp_path):
        path = tmp_path / "presses.jsonl"
        PressLedger(path).append(_record())
        assert PressLedger(path).has(("C1", "99.1", "menu-a"))

    def test_a_press_without_action_ts_is_refused(self, tmp_path):
        ledger = PressLedger(tmp_path / "presses.jsonl")
        with pytest.raises(ValueError, match="action_ts"):
            ledger.append(_record(action_ts=""))
        assert not ledger.path.exists()

    def test_a_malformed_line_is_skipped_and_logged(self, tmp_path, caplog):
        path = tmp_path / "presses.jsonl"
        path.write_text("not json\n[1, 2]\n" + json.dumps(_record()) + "\n")
        with caplog.at_level(logging.WARNING):
            assert PressLedger(path).has(("C1", "99.1", "menu-a"))
        assert caplog.text.count("press ledger: skipping malformed line") == 2

    def test_an_unreadable_ledger_reads_as_empty_and_says_so(self, tmp_path, caplog):
        path = tmp_path / "presses.jsonl"
        path.mkdir()
        with caplog.at_level(logging.WARNING):
            assert not PressLedger(path).has(("C1", "99.1", "menu-a"))
        assert "press ledger: cannot read" in caplog.text

    def test_a_missing_ledger_reads_as_empty(self, tmp_path):
        assert not PressLedger(tmp_path / "absent.jsonl").has(("C1", "1", "b"))

    def test_a_failed_append_raises_and_records_no_key(self, tmp_path):
        path = tmp_path / "presses.jsonl"
        path.mkdir()
        ledger = PressLedger(path)
        with pytest.raises(OSError):
            ledger.append(_record())
        assert not ledger.has(("C1", "99.1", "menu-a"))


class TestMarker:
    def test_the_marker_names_this_process_and_its_os_start_time(self, tmp_path):
        path = tmp_path / "state" / "marker.json"
        write_marker(path, ledger_name="presses.jsonl")
        marker = json.loads(path.read_text())
        assert marker["v"] == 1
        assert marker["pid"] == os.getpid()
        # psutil is the independent oracle here; the module itself asks `ps`.
        started = int(psutil.Process(os.getpid()).create_time())
        expected = datetime.fromtimestamp(started, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert marker["process_started_at"] == expected
        assert marker["written_at"].endswith("Z")
        assert marker["ledger"] == "presses.jsonl"

    def test_the_marker_replaces_a_stale_one_whole(self, tmp_path):
        path = tmp_path / "marker.json"
        path.write_text('{"pid": 1, "process_started_at": "2000-01-01T00:00:00Z"}')
        write_marker(path, ledger_name="presses.jsonl")
        assert json.loads(path.read_text())["pid"] == os.getpid()
        assert [p.name for p in tmp_path.iterdir()] == ["marker.json"]

    def test_no_marker_when_the_start_time_cannot_be_read(self, tmp_path, monkeypatch):
        def broken(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr(button_presses.subprocess, "run", broken)
        path = tmp_path / "marker.json"
        with pytest.raises(OSError, match="start time"):
            write_marker(path, ledger_name="presses.jsonl")
        assert not path.exists()

    def test_an_unparseable_start_time_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            button_presses.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "not a date\n", ""),
        )
        with pytest.raises(OSError, match="start time"):
            write_marker(tmp_path / "marker.json", ledger_name="presses.jsonl")

    def test_the_start_time_is_read_in_utc(self, monkeypatch):
        seen = {}

        def fake(cmd, **kwargs):
            seen.update(kwargs["env"])
            return subprocess.CompletedProcess(cmd, 0, "Mon Nov  2 10:40:01 2026\n", "")

        monkeypatch.setattr(button_presses.subprocess, "run", fake)
        assert button_presses.process_started_at(123) == "2026-11-02T10:40:01Z"
        assert seen["TZ"] == "UTC0"
        assert seen["LC_ALL"] == "C"


class TestPressRateLimit:
    def test_six_presses_in_a_window_pass_and_the_seventh_does_not(self):
        now = [1000.0]
        limit = PressRateLimit(clock=lambda: now[0])
        assert [limit.allow("U1") for _ in range(7)] == [True] * 6 + [False]

    def test_the_window_slides(self):
        now = [1000.0]
        limit = PressRateLimit(clock=lambda: now[0])
        for _ in range(6):
            assert limit.allow("U1")
        now[0] += 60.0
        assert limit.allow("U1")

    def test_a_refused_press_does_not_count(self):
        now = [1000.0]
        limit = PressRateLimit(limit=1, window=60.0, clock=lambda: now[0])
        assert limit.allow("U1")
        now[0] += 30.0
        assert not limit.allow("U1")
        now[0] += 30.0
        assert limit.allow("U1")

    def test_the_limit_is_per_presser(self):
        limit = PressRateLimit(limit=1, clock=lambda: 1.0)
        assert limit.allow("U1")
        assert limit.allow("U2")
        assert not limit.allow("U1")

    def test_the_defaults_are_six_per_minute(self):
        assert button_presses.RATE_LIMIT == 6
        assert button_presses.RATE_WINDOW_SECONDS == 60.0


class TestDropLine:
    def test_the_drop_line_shape(self):
        assert drop_line(
            "duplicate",
            channel="C1",
            message_ts="99.1",
            block_id="menu-a",
            action_ts="100.2",
            user="U1",
        ) == (
            "(button duplicate) [channel: C1] [menu-ts: 99.1] "
            "[block-id: menu-a] [action-ts: 100.2] [user: U1]"
        )

    def test_a_field_cannot_forge_another_field(self):
        line = drop_line(
            "retire-failed",
            channel="C1",
            message_ts="99.1",
            block_id="x] [user: U_FORGED\nnext",
            action_ts="",
            user="U1",
        )
        assert "\n" not in line
        assert line.count("[user: ") == 1
        assert "[action-ts: -]" in line

    def test_only_the_four_reasons_exist(self):
        assert button_presses.DROP_REASONS == (
            "duplicate",
            "rate-limited",
            "ledger-write-failed",
            "retire-failed",
        )
        with pytest.raises(ValueError):
            drop_line(
                "other",
                channel="C1",
                message_ts="1",
                block_id="b",
                action_ts="2",
                user="U1",
            )

"""The operator's deny rule for suggestion labels: loading, matching, failing closed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from claude_on_the_fly import suggestion_deny
from tests.conftest import SUGGESTION_DENY_PATTERNS


def _config(operator_settings, value: object) -> None:
    operator_settings.write_text(
        "suggestions:\n  deny_file: " + json.dumps(value) + "\n"
    )


class TestNormalise:
    def test_dashes_and_underscore_fold_to_a_hyphen(self):
        folded = suggestion_deny.normalise("a‐b‑c‒d–e—f―g−h﹘i﹣j－k_l")
        assert folded == "a-b-c-d-e-f-g-h-i-j-k-l"

    def test_full_width_letters_and_digits_become_ascii(self):
        assert suggestion_deny.normalise("Ｒ１２") == "R12"


class TestState:
    def test_no_key_means_the_feature_is_off(self, operator_settings):
        rules = suggestion_deny.current()
        assert not rules.configured
        assert suggestion_deny.tap_refusal("R12-ABCD-3X4") is None
        assert suggestion_deny.offer_filter(["R12-ABCD-3X4"]) == ["R12-ABCD-3X4"]

    def test_a_null_key_means_the_feature_is_off(self, operator_settings):
        operator_settings.write_text("suggestions:\n  deny_file:\n")
        assert not suggestion_deny.current().configured

    def test_a_readable_file_loads_its_patterns(self, deny_file):
        rules = suggestion_deny.current()
        assert rules.configured and rules.readable
        assert len(rules.patterns) == len(SUGGESTION_DENY_PATTERNS)

    def test_an_empty_pattern_list_is_unusable_and_says_so(self, deny_file, caplog):
        deny_file.write_text(json.dumps({"patterns": []}))
        with caplog.at_level("ERROR", logger="claude_on_the_fly.suggestion_deny"):
            rules = suggestion_deny.current()
            suggestion_deny.current()
        assert rules.configured and not rules.readable
        assert "empty" in rules.error
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert suggestion_deny.tap_refusal("hello") == suggestion_deny.UNREADABLE
        assert suggestion_deny.offer_filter(["R12-ABCD-3X4"]) == ["R12-ABCD-3X4"]

    @pytest.mark.parametrize(
        "content",
        [
            "{not json",
            "[]",
            json.dumps({"why": "no patterns key"}),
            json.dumps({"patterns": "R12"}),
            json.dumps({"patterns": [1]}),
            json.dumps({"patterns": ["(unclosed"]}),
            "NOT_UTF8",
        ],
        ids=[
            "bad-json",
            "not-a-mapping",
            "no-patterns",
            "patterns-not-a-list",
            "pattern-not-a-string",
            "pattern-does-not-compile",
            "not-utf8",
        ],
    )
    def test_a_bad_file_is_unreadable(self, deny_file, content):
        if content == "NOT_UTF8":
            deny_file.write_bytes(b"\xff\xfe\x00")
        else:
            deny_file.write_text(content)
        rules = suggestion_deny.current()
        assert rules.configured and not rules.readable

    def test_a_missing_file_is_unreadable(self, deny_file):
        deny_file.unlink()
        assert not suggestion_deny.current().readable

    @pytest.mark.parametrize("value", ["relative/deny.json", "", 7, ["/a.json"]])
    def test_a_path_that_is_not_absolute_is_unreadable(self, operator_settings, value):
        _config(operator_settings, value)
        rules = suggestion_deny.current()
        assert rules.configured and not rules.readable
        assert "absolute path" in rules.error

    def test_unreadable_is_logged_loudly_once_per_state(self, deny_file, caplog):
        deny_file.unlink()
        with caplog.at_level("ERROR", logger="claude_on_the_fly.suggestion_deny"):
            for _ in range(3):
                suggestion_deny.current()
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert errors[0].getMessage().startswith("suggestion-deny: ")
        assert "every suggestion tap is refused" in errors[0].getMessage()

    def test_a_load_is_logged_once(self, deny_file, caplog):
        with caplog.at_level("INFO", logger="claude_on_the_fly.suggestion_deny"):
            suggestion_deny.current()
            suggestion_deny.current()
        loads = [r for r in caplog.records if "loaded" in r.getMessage()]
        assert len(loads) == 1

    def test_the_file_is_re_read_when_its_mtime_changes(self, deny_file):
        assert suggestion_deny.tap_refusal("ZZZ") is None
        deny_file.write_text(json.dumps({"patterns": ["ZZZ"]}))
        stat = deny_file.stat()
        os.utime(deny_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        assert suggestion_deny.tap_refusal("ZZZ") == suggestion_deny.MATCHED

    def test_an_unchanged_file_is_not_re_parsed(self, deny_file, monkeypatch):
        suggestion_deny.current()
        monkeypatch.setattr(
            suggestion_deny,
            "_load",
            lambda path: pytest.fail("re-parsed an unchanged file"),
        )
        assert suggestion_deny.current().readable

    def test_a_fixed_file_recovers(self, deny_file):
        good = deny_file.read_text()
        deny_file.write_text("{")
        assert not suggestion_deny.current().readable
        deny_file.write_text(good + " ")
        assert suggestion_deny.current().readable


class TestVerdicts:
    def test_a_match_refuses_a_tap(self, deny_file):
        assert suggestion_deny.tap_refusal("R12-ABCD-3X4") == suggestion_deny.MATCHED

    def test_an_ordinary_label_is_not_refused(self, deny_file):
        assert suggestion_deny.tap_refusal("看 PRJ-12345") is None

    def test_an_unreadable_file_refuses_every_tap(self, deny_file):
        deny_file.unlink()
        assert suggestion_deny.tap_refusal("hello") == suggestion_deny.UNREADABLE

    def test_an_unreadable_file_drops_no_offer(self, deny_file):
        deny_file.unlink()
        assert suggestion_deny.offer_filter(["R12-ABCD-3X4"]) == ["R12-ABCD-3X4"]

    def test_the_refusal_text_names_both_causes_and_nothing_else(self):
        assert suggestion_deny.REFUSAL_TEXT == (
            "Not sent: this button is disabled by a deny rule (or the rule file "
            "could not be read). Type your reply instead."
        )


@pytest.mark.skipif(
    shutil.which("python3.9") is None and not os.path.exists("/usr/bin/python3"),
    reason="no system Python to check older-interpreter syntax against",
)
def test_the_reference_patterns_compile_on_the_system_python():
    """Another reader of the same rule file may run an older stdlib `re`."""
    python = shutil.which("python3.9") or "/usr/bin/python3"
    script = "import json,re,sys\nfor p in json.load(sys.stdin): re.compile(p)\n"
    done = subprocess.run(
        [python, "-I", "-c", script],
        input=json.dumps(SUGGESTION_DENY_PATTERNS),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr

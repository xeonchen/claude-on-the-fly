"""The shared-room check: text + room facts -> hold or post.

Everything here is pure. The Slack adapter (membership, the owner's DM, the
actual posts) is tested in test_slack.py under `TestSharedRoomSend`.
"""

import pytest

from claude_on_the_fly.shared_room import (
    HELD_REPLY,
    HoldRules,
    RoomFacts,
    check,
    held_notice,
    is_shared,
    prompt_line,
)

SEATS = ("Atlas", "Borealis", "QZ", "Cirrus", "Spring", "Desk")
DESK_ID = r"(?<![A-Za-z0-9_])20\d\d-\d\d-\d\d-[0-9a-f]{4}(?![A-Za-z0-9_])"
OWNER_DM = "D0OWNER1"


def rules(people: tuple[str, ...] = ()) -> HoldRules:
    built, errors = HoldRules.build(SEATS, (DESK_ID,), people)
    assert errors == ()
    return built


SHARED = RoomFacts(shared=True, owner_dm_channels=frozenset({OWNER_DM}))
PRIVATE = RoomFacts(shared=False, owner_dm_channels=frozenset({OWNER_DM}))


class TestMatches:
    @pytest.mark.parametrize("seat", SEATS)
    def test_each_seat_name_holds(self, seat):
        reason = check(
            f"I've passed this to {seat} and will follow up.", rules(), SHARED
        )
        assert reason is not None
        assert seat in reason

    @pytest.mark.parametrize("seat", ["Borealis", "QZ", "Desk"])
    def test_a_seat_inside_chinese_text_holds(self, seat):
        """CJK characters are word characters to `\\b`, so a plain `\\b` boundary
        would miss 「交給Borealis處理」 entirely."""
        assert check(f"這件事我交給{seat}處理。", rules(), SHARED) is not None

    def test_desk_as_the_systems_name_holds(self):
        assert check("I opened a ticket on the Desk for it.", rules(), SHARED)

    def test_a_full_desk_id_holds_and_names_it(self):
        reason = check("Tracked as 2030-01-02-ab12.", rules(), SHARED)
        assert reason is not None
        assert "2030-01-02-ab12" in reason

    def test_a_desk_id_glued_to_chinese_text_holds(self):
        assert check("已開工單2030-01-02-cd34處理", rules(), SHARED) is not None

    def test_a_desk_id_inside_a_longer_token_does_not(self):
        assert check("build-2030-01-02-cd34x", rules(), SHARED) is None

    def test_an_owner_dm_client_url_holds(self):
        text = f"https://app.slack.com/client/T0ABC/{OWNER_DM}"
        assert check(text, rules(), SHARED) is not None

    def test_an_owner_dm_app_link_with_escaped_ampersand_holds(self):
        text = f"<slack://channel?team=T1&amp;id={OWNER_DM}|here>"
        assert check(text, rules(), SHARED) is not None

    def test_an_owner_dm_permalink_holds(self):
        text = f"Details: https://acme.slack.com/archives/{OWNER_DM}/p1730880000123456"
        reason = check(text, rules(), SHARED)
        assert reason is not None
        assert OWNER_DM in reason

    def test_an_owner_dm_app_link_holds(self):
        text = f"<slack://channel?team=T1&id={OWNER_DM}|here>"
        assert check(text, rules(), SHARED) is not None

    def test_with_no_known_owner_dm_any_dm_permalink_holds(self):
        """Fail closed: when the owner's DM cannot be resolved, a link into any
        1:1 DM is one a third party cannot open."""
        room = RoomFacts(shared=True, owner_dm_channels=frozenset())
        text = "see https://acme.slack.com/archives/D0SOMEONE/p1"
        assert check(text, rules(), room) is not None


class TestFalseMatches:
    @pytest.mark.parametrize(
        "text",
        [
            "Please raise it with the help desk.",
            "It's on my desk, I'll look tomorrow.",
            "Thanks @Sam, @Lee and <@U0123ABC> for the review.",
            "This is ABC-12345, due 2026-11-06.",
            "Release window is 2026-11-06 to 2026-11-13.",
            "Shipping before spring is the plan.",
            "Borealises and QZs are not names here.",  # suffixes do not match
            "AQZ-12 and SKQZ are other strings.",
            "See https://acme.slack.com/archives/C0PUBLIC/p1730880000123456",
            "Desktop app crashed again.",
        ],
    )
    def test_ordinary_text_posts(self, text):
        assert check(text, rules(), SHARED) is None

    def test_a_person_named_like_a_seat_posts_when_listed_as_a_person(self):
        assert check("Spring will join the call.", rules(), SHARED) is not None
        assert check("Spring will join the call.", rules(("Spring",)), SHARED) is None
        # The other seats still hold.
        assert check("Borealis will join.", rules(("Spring",)), SHARED) is not None

    def test_another_owners_dm_link_is_not_this_rule(self):
        text = "see https://acme.slack.com/archives/D0SOMEONE/p1"
        assert check(text, rules(), SHARED) is None


class TestRooms:
    def test_a_room_with_only_the_owner_is_untouched(self):
        text = "Borealis has 2030-01-02-ab12 on the Desk."
        assert check(text, rules(), PRIVATE) is None

    def test_no_rules_configured_holds_nothing(self):
        empty, _ = HoldRules.build((), (), ())
        assert not empty.enabled
        assert check("Borealis 2030-01-02-ab12", empty, SHARED) is None

    @pytest.mark.parametrize(
        ("members", "owners", "own", "shared"),
        [
            ({"U_OWN", "U_BOT"}, {"U_OWN"}, "U_BOT", False),
            ({"U_OWN"}, {"U_OWN"}, "U_OWN", False),
            ({"U_OWN", "U_BOT", "U_COLLEAGUE"}, {"U_OWN"}, "U_BOT", True),
            ({"U_COLLEAGUE", "U_BOT"}, {"U_OWN"}, "U_BOT", True),
            # No owner known: any human besides the bot is somebody else.
            ({"U_OWN", "U_BOT"}, set(), "U_BOT", True),
            # Membership unknown: fail closed.
            (None, {"U_OWN"}, "U_BOT", True),
        ],
    )
    def test_is_shared(self, members, owners, own, shared):
        assert is_shared(members, frozenset(owners), own) is shared


class TestBuild:
    def test_a_broken_pattern_is_reported_and_the_rest_still_apply(self):
        built, errors = HoldRules.build(("Borealis",), ("[unclosed", DESK_ID), ())
        assert len(errors) == 1
        assert "[unclosed" in errors[0]
        assert check("2030-01-02-ab12", built, SHARED) is not None

    def test_blank_entries_are_dropped(self):
        built, errors = HoldRules.build(("", " Borealis "), ("", " "), (" ",))
        assert errors == ()
        assert built.names == ("Borealis",)
        assert built.patterns == ()


class TestWording:
    def test_held_reply_is_the_approved_line(self):
        assert HELD_REPLY == "收到，處理中，有結果會再回覆。"

    def test_held_notice_names_room_and_rule_then_carries_the_original(self):
        notice = held_notice("group-dm G1", "name `Borealis`", "original text")
        first, rest = notice.split("\n", 1)
        assert "G1" in first
        assert "Borealis" in first
        assert rest.strip() == "original text"

    def test_prompt_line_lists_the_names_and_the_asks(self):
        line = prompt_line(rules())
        assert "\n" not in line
        for seat in SEATS:
            assert seat in line
        assert "ticket id" in line
        assert "<@" in line

    def test_prompt_line_is_empty_when_nothing_is_configured(self):
        empty, _ = HoldRules.build((), (), ())
        assert prompt_line(empty) == ""

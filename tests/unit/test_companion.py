"""tests/unit/test_companion.py — Run with: pytest tests/unit/test_companion.py -v"""
import pytest
from datetime import date, datetime, timezone

from reskillio.companion.models import (
    ActionItem, CheckinSubmitRequest, DigestSection, WeeklyCheckin, WeeklyDigest,
)


class TestWeeklyCheckin:
    def test_defaults(self):
        c = WeeklyCheckin(
            candidate_id="test-001", week_number=1,
            week_start=date.today(), checked_in_at=datetime.now(timezone.utc),
        )
        assert c.hours_on_courses == 0.0
        assert c.applications_sent == 0
        assert c.digest_generated is False

    def test_full_checkin(self):
        c = WeeklyCheckin(
            candidate_id="test-001", week_number=3,
            week_start=date(2026, 4, 21),
            checked_in_at=datetime.now(timezone.utc),
            hours_on_courses=4.5,
            applications_sent=2,
            interviews_scheduled=1,
            gap_score=81.0,
            gap_score_delta=7.0,
        )
        assert c.gap_score == 81.0
        assert c.gap_score_delta == 7.0
        assert c.interviews_scheduled == 1


class TestActionItem:
    def test_defaults(self):
        a = ActionItem(title="Complete Python module 2", description="Close the gap",
                       priority="high", time_est="3 hrs")
        assert a.completed is False

    def test_priority_values(self):
        for p in ["high", "medium", "low"]:
            a = ActionItem(title="Test", description="Test", priority=p, time_est="1 hr")
            assert a.priority == p


class TestCheckinRequest:
    def test_minimal(self):
        r = CheckinSubmitRequest(candidate_id="test-001")
        assert r.hours_on_courses == 0.0
        assert r.applications_sent == 0
        assert r.courses_completed == []

    def test_full(self):
        r = CheckinSubmitRequest(
            candidate_id="test-001",
            hours_on_courses=6.0,
            courses_completed=["Python for Everybody — Week 2"],
            applications_sent=3,
            interviews_scheduled=1,
        )
        assert r.hours_on_courses == 6.0
        assert len(r.courses_completed) == 1


class TestDigestSection:
    def test_section_types(self):
        for st in ["gap_progress", "course_progress", "applications", "market", "narrative"]:
            s = DigestSection(section_type=st, headline="Test", body="Test", data={})
            assert s.section_type == st

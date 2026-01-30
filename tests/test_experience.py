"""
Tests for the Experience Calculation (Merge Intervals Algorithm).
"""

import pytest
from datetime import date

from app.services.utils.experience import (
    calculate_total_experience,
    merge_intervals,
    TimeInterval,
    get_experience_gaps,
)


class TestMergeIntervals:
    """Tests for merge_intervals function."""

    def test_single_interval(self):
        """Single interval should return as-is."""
        intervals = [(date(2020, 1, 1), date(2022, 12, 31))]
        merged = merge_intervals(intervals)

        assert len(merged) == 1
        assert merged[0].start == date(2020, 1, 1)
        assert merged[0].end == date(2022, 12, 31)

    def test_non_overlapping_intervals(self):
        """Non-overlapping intervals should remain separate."""
        intervals = [
            (date(2018, 1, 1), date(2019, 12, 31)),
            (date(2021, 1, 1), date(2022, 12, 31)),
        ]
        merged = merge_intervals(intervals)

        assert len(merged) == 2
        assert merged[0].end == date(2019, 12, 31)
        assert merged[1].start == date(2021, 1, 1)

    def test_overlapping_intervals(self):
        """Overlapping intervals should be merged."""
        intervals = [
            (date(2018, 1, 1), date(2020, 12, 31)),  # 3 years
            (date(2019, 6, 1), date(2021, 6, 30)),   # Overlaps with above
        ]
        merged = merge_intervals(intervals)

        assert len(merged) == 1
        assert merged[0].start == date(2018, 1, 1)
        assert merged[0].end == date(2021, 6, 30)

    def test_contained_interval(self):
        """An interval contained within another should merge."""
        intervals = [
            (date(2018, 1, 1), date(2022, 12, 31)),  # Large interval
            (date(2019, 6, 1), date(2020, 6, 30)),   # Contained within
        ]
        merged = merge_intervals(intervals)

        assert len(merged) == 1
        assert merged[0].start == date(2018, 1, 1)
        assert merged[0].end == date(2022, 12, 31)

    def test_adjacent_intervals(self):
        """Adjacent intervals (touching) should be merged."""
        intervals = [
            (date(2018, 1, 1), date(2019, 12, 31)),
            (date(2019, 12, 31), date(2021, 12, 31)),  # Starts when first ends
        ]
        merged = merge_intervals(intervals)

        assert len(merged) == 1
        assert merged[0].start == date(2018, 1, 1)
        assert merged[0].end == date(2021, 12, 31)

    def test_unsorted_intervals(self):
        """Intervals provided out of order should still merge correctly."""
        intervals = [
            (date(2021, 1, 1), date(2022, 12, 31)),
            (date(2018, 1, 1), date(2019, 12, 31)),
            (date(2019, 6, 1), date(2020, 6, 30)),
        ]
        merged = merge_intervals(intervals)

        # First two merge, third is separate
        assert len(merged) == 2

    def test_current_position_none_end(self):
        """Current position (None end date) should use today."""
        today = date.today()
        intervals = [
            (date(2020, 1, 1), None),  # Current position
        ]
        merged = merge_intervals(intervals)

        assert len(merged) == 1
        assert merged[0].end == today

    def test_empty_intervals(self):
        """Empty list should return empty."""
        merged = merge_intervals([])
        assert merged == []

    def test_multiple_current_positions(self):
        """Multiple current positions should merge if overlapping."""
        intervals = [
            (date(2020, 1, 1), None),  # Current job 1
            (date(2021, 6, 1), None),  # Current job 2 (overlaps)
        ]
        merged = merge_intervals(intervals)

        assert len(merged) == 1
        assert merged[0].start == date(2020, 1, 1)


class TestCalculateTotalExperience:
    """Tests for calculate_total_experience function."""

    def test_single_job_three_years(self):
        """Single job of approximately 3 years."""
        intervals = [(date(2020, 1, 1), date(2022, 12, 31))]
        years = calculate_total_experience(intervals)

        # Should be approximately 3 years
        assert 2.9 <= years <= 3.1

    def test_overlapping_jobs_no_double_count(self):
        """Overlapping jobs should NOT be double-counted."""
        intervals = [
            (date(2018, 1, 1), date(2020, 12, 31)),  # 3 years
            (date(2019, 6, 1), date(2021, 6, 30)),   # 2 years overlap
        ]
        years = calculate_total_experience(intervals)

        # Should be 3.5 years (2018-01 to 2021-06), not 5 years
        assert 3.4 <= years <= 3.6

    def test_two_separate_jobs(self):
        """Two non-overlapping jobs should sum correctly."""
        intervals = [
            (date(2018, 1, 1), date(2018, 12, 31)),  # 1 year
            (date(2020, 1, 1), date(2020, 12, 31)),  # 1 year
        ]
        years = calculate_total_experience(intervals)

        # Should be approximately 2 years
        assert 1.9 <= years <= 2.1

    def test_current_position(self):
        """Current position should count up to today."""
        # One year ago to now
        one_year_ago = date(date.today().year - 1, date.today().month, date.today().day)
        intervals = [(one_year_ago, None)]
        years = calculate_total_experience(intervals)

        # Should be approximately 1 year
        assert 0.9 <= years <= 1.1

    def test_empty_list_returns_zero(self):
        """Empty list should return 0."""
        years = calculate_total_experience([])
        assert years == 0.0


class TestExperienceGaps:
    """Tests for get_experience_gaps function."""

    def test_no_gaps(self):
        """Continuous employment should have no gaps."""
        intervals = [
            (date(2018, 1, 1), date(2019, 12, 31)),
            (date(2020, 1, 1), date(2022, 12, 31)),  # 1 day gap, too small
        ]
        gaps = get_experience_gaps(intervals, min_gap_months=3)
        assert len(gaps) == 0

    def test_significant_gap(self):
        """6-month gap should be detected."""
        intervals = [
            (date(2018, 1, 1), date(2019, 6, 30)),
            (date(2020, 1, 1), date(2022, 12, 31)),  # 6 month gap
        ]
        gaps = get_experience_gaps(intervals, min_gap_months=3)

        assert len(gaps) == 1
        assert gaps[0].start == date(2019, 6, 30)
        assert gaps[0].end == date(2020, 1, 1)


class TestTimeInterval:
    """Tests for TimeInterval dataclass."""

    def test_duration_days(self):
        """Test duration calculation in days."""
        interval = TimeInterval(
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
        )
        assert interval.duration_days == 365

    def test_duration_years(self):
        """Test duration calculation in years."""
        interval = TimeInterval(
            start=date(2020, 1, 1),
            end=date(2022, 12, 31),
        )
        # Approximately 3 years
        assert 2.9 <= interval.duration_years <= 3.1

    def test_overlaps_true(self):
        """Test overlap detection."""
        interval1 = TimeInterval(date(2020, 1, 1), date(2021, 12, 31))
        interval2 = TimeInterval(date(2021, 1, 1), date(2022, 12, 31))

        assert interval1.overlaps(interval2)
        assert interval2.overlaps(interval1)

    def test_overlaps_false(self):
        """Test non-overlapping intervals."""
        interval1 = TimeInterval(date(2018, 1, 1), date(2019, 12, 31))
        interval2 = TimeInterval(date(2021, 1, 1), date(2022, 12, 31))

        assert not interval1.overlaps(interval2)

    def test_merge(self):
        """Test interval merging."""
        interval1 = TimeInterval(date(2018, 1, 1), date(2020, 12, 31))
        interval2 = TimeInterval(date(2019, 1, 1), date(2022, 12, 31))

        merged = interval1.merge(interval2)

        assert merged.start == date(2018, 1, 1)
        assert merged.end == date(2022, 12, 31)

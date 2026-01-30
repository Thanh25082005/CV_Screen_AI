"""
Experience Calculation using Merge Overlapping Intervals Algorithm.

This module provides accurate calculation of total work experience by:
1. Merging overlapping time periods (e.g., simultaneous jobs)
2. Handling gaps in employment
3. Supporting ongoing positions (end_date = None)

The key insight is that if someone works two jobs simultaneously,
we should NOT double-count that time period.

Example:
    Job A: 2018-01-01 to 2020-12-31 (3 years)
    Job B: 2019-06-01 to 2021-06-30 (2 years)
    
    Naive sum: 5 years
    Correct (merged): 3.5 years (2018-01 to 2021-06)
"""

from datetime import date
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TimeInterval:
    """Represents a time interval with start and end dates."""

    start: date
    end: date

    @property
    def duration_days(self) -> int:
        """Duration in days."""
        return (self.end - self.start).days

    @property
    def duration_years(self) -> float:
        """Duration in years (accounting for leap years)."""
        return self.duration_days / 365.25

    def overlaps(self, other: "TimeInterval") -> bool:
        """Check if this interval overlaps with another."""
        return self.start <= other.end and other.start <= self.end

    def merge(self, other: "TimeInterval") -> "TimeInterval":
        """Merge this interval with another overlapping interval."""
        return TimeInterval(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
        )


def merge_intervals(
    intervals: List[Tuple[date, Optional[date]]]
) -> List[TimeInterval]:
    """
    Merge overlapping time intervals.
    
    Algorithm:
    1. Replace None end dates with today (ongoing positions)
    2. Sort intervals by start date
    3. Iterate and merge overlapping intervals
    
    Time Complexity: O(n log n) for sorting
    Space Complexity: O(n) for the merged list
    
    Args:
        intervals: List of (start_date, end_date) tuples.
                   end_date can be None for ongoing positions.
    
    Returns:
        List of merged TimeInterval objects
    
    Example:
        >>> intervals = [
        ...     (date(2018, 1, 1), date(2020, 12, 31)),
        ...     (date(2019, 6, 1), date(2021, 6, 30)),
        ...     (date(2022, 1, 1), None),  # Ongoing
        ... ]
        >>> merged = merge_intervals(intervals)
        >>> len(merged)
        2  # (2018-01 to 2021-06) and (2022-01 to today)
    """
    if not intervals:
        return []

    today = date.today()

    # Convert to TimeInterval objects, replacing None with today
    time_intervals = []
    for start, end in intervals:
        if start is None:
            continue  # Skip invalid intervals
        effective_end = end if end is not None else today
        if start <= effective_end:  # Validate date order
            time_intervals.append(TimeInterval(start=start, end=effective_end))

    if not time_intervals:
        return []

    # Sort by start date
    time_intervals.sort(key=lambda x: x.start)

    # Merge overlapping intervals
    merged = [time_intervals[0]]

    for current in time_intervals[1:]:
        last = merged[-1]

        if current.start <= last.end:
            # Overlapping - merge them
            merged[-1] = last.merge(current)
        else:
            # No overlap - add as new interval
            merged.append(current)

    return merged


def calculate_total_experience(
    intervals: List[Tuple[date, Optional[date]]]
) -> float:
    """
    Calculate total years of work experience from job intervals.
    
    This function handles the common case where a candidate holds
    multiple positions simultaneously (e.g., part-time, freelance)
    by merging overlapping periods before summing.
    
    Args:
        intervals: List of (start_date, end_date) tuples from work history.
                   end_date can be None for current positions.
    
    Returns:
        Total years of experience as a float (rounded to 2 decimals)
    
    Example:
        >>> work_history = [
        ...     (date(2018, 1, 1), date(2020, 12, 31)),  # 3 years
        ...     (date(2019, 6, 1), date(2021, 6, 30)),   # Overlaps!
        ...     (date(2022, 1, 1), None),               # Current job
        ... ]
        >>> years = calculate_total_experience(work_history)
        >>> # Will NOT double-count the 2019-06 to 2020-12 overlap
    """
    if not intervals:
        return 0.0

    # Merge overlapping intervals
    merged = merge_intervals(intervals)

    if not merged:
        return 0.0

    # Sum durations of merged intervals
    total_days = sum(interval.duration_days for interval in merged)

    # Convert to years
    total_years = total_days / 365.25

    return round(total_years, 2)


def calculate_experience_at_company(
    intervals: List[Tuple[date, Optional[date]]],
    company: str,
    company_intervals: List[Tuple[str, date, Optional[date]]],
) -> float:
    """
    Calculate years of experience at a specific company.
    
    Args:
        intervals: All work intervals
        company: Company name to filter
        company_intervals: List of (company_name, start, end) tuples
    
    Returns:
        Years at the specified company
    """
    company_lower = company.lower()
    relevant = [
        (start, end)
        for comp, start, end in company_intervals
        if company_lower in comp.lower()
    ]

    return calculate_total_experience(relevant)


def get_experience_gaps(
    intervals: List[Tuple[date, Optional[date]]],
    min_gap_months: int = 3,
) -> List[TimeInterval]:
    """
    Find significant gaps in employment history.
    
    Args:
        intervals: Work history intervals
        min_gap_months: Minimum gap size to report (in months)
    
    Returns:
        List of TimeInterval objects representing gaps
    """
    merged = merge_intervals(intervals)

    if len(merged) < 2:
        return []

    gaps = []
    min_gap_days = min_gap_months * 30  # Approximate

    for i in range(1, len(merged)):
        prev_end = merged[i - 1].end
        curr_start = merged[i].start

        gap_days = (curr_start - prev_end).days

        if gap_days >= min_gap_days:
            gaps.append(TimeInterval(start=prev_end, end=curr_start))

    return gaps


def format_experience_summary(
    intervals: List[Tuple[date, Optional[date]]],
) -> str:
    """
    Generate a human-readable experience summary.
    
    Args:
        intervals: Work history intervals
    
    Returns:
        Summary string like "5.5 years (3 positions, 1 gap of 6 months)"
    """
    total_years = calculate_total_experience(intervals)
    merged = merge_intervals(intervals)
    gaps = get_experience_gaps(intervals, min_gap_months=3)

    parts = [f"{total_years} years"]

    if len(intervals) > 1:
        parts.append(f"{len(intervals)} positions")

    if gaps:
        if len(gaps) == 1:
            gap_months = round(gaps[0].duration_days / 30)
            parts.append(f"1 gap of {gap_months} months")
        else:
            parts.append(f"{len(gaps)} employment gaps")

    return " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else parts[0]

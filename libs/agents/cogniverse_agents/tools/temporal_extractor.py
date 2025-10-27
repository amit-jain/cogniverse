#!/usr/bin/env python3
# src/tools/temporal_extractor.py
"""
Enhanced temporal pattern recognition based on failure analysis.
Handles complex temporal expressions that the basic system misses.
"""

import re
import datetime
from typing import Dict, Any, Optional, List
from dateutil import parser as date_parser
import calendar

class EnhancedTemporalExtractor:
    """Enhanced temporal pattern extractor with better coverage."""
    
    def __init__(self):
        self.today = datetime.date.today()
        
        # Enhanced pattern mapping based on failure analysis
        self.pattern_map = {
            # Basic patterns (working)
            r'\byesterday\b': "yesterday",
            r'\blast week\b': "last_week", 
            r'\blast month\b': "last_month",
            r'\bthis week\b': "this_week",
            r'\bthis month\b': "this_month",
            r'\bthis morning\b': "this_morning",
            
            # Failed patterns that need fixing
            r'\bpast 7 days\b': "past_7_days",
            r'\bpast seven days\b': "past_7_days",
            r'\bpast week\b': "past_week",
            r'\bpast month\b': "past_month",
            
            # Day-specific patterns
            r'\btwo days ago\b': "two_days_ago",
            r'\bthree days ago\b': "three_days_ago", 
            r'\bday before yesterday\b': "day_before_yesterday",
            r'\btwo weeks ago\b': "two_weeks_ago",
            r'\bmonth ago\b': "month_ago",
            r'\ba month ago\b': "month_ago",
            
            # Time period patterns
            r'\bbeginning of (?:this )?month\b': "beginning_of_month",
            r'\bend of last year\b': "end_of_last_year",
            r'\bfirst quarter\b': "first_quarter",
            r'\bprevious quarter\b': "previous_quarter",
            
            # Weekday patterns
            r'\bmonday last week\b': "monday_last_week",
            r'\blast tuesday\b': "last_tuesday",
            r'\bweekend\b': "weekend",
            r'\bholiday period\b': "holiday_period",
            
            # Future patterns
            r'\bnext week\b': "next_week",
            r'\bnext month\b': "next_month"
        }
        
        # Month name mapping
        self.month_names = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08', 
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
            'oct': '10', 'nov': '11', 'dec': '12'
        }
    
    def extract_temporal_pattern(self, query: str) -> Optional[str]:
        """Extract temporal pattern from query with enhanced coverage."""
        query_lower = query.lower()
        
        # Try date range patterns first (most specific)
        date_range = self._extract_date_range(query_lower)
        if date_range:
            return date_range
        
        # Try specific date patterns
        specific_date = self._extract_specific_date(query_lower)
        if specific_date:
            return specific_date
            
        # Try named month patterns
        month_pattern = self._extract_month_pattern(query_lower)
        if month_pattern:
            return month_pattern
        
        # Try pattern matching
        for pattern, key in self.pattern_map.items():
            if re.search(pattern, query_lower):
                return key
        
        return None
    
    def _extract_date_range(self, query: str) -> Optional[str]:
        """Extract date ranges like 'between 2024-01-10 and 2024-01-20'."""
        # Pattern: between DATE and DATE
        range_pattern = r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})'
        match = re.search(range_pattern, query)
        if match:
            start_date, end_date = match.groups()
            return f"{start_date}_to_{end_date}"
        
        return None
    
    def _extract_specific_date(self, query: str) -> Optional[str]:
        """Extract specific dates in various formats."""
        # ISO format: 2024-01-15
        iso_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', query)
        if iso_match:
            return iso_match.group(1)
        
        # US format: 01/15/2024 or 1/15/2024  
        us_match = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', query)
        if us_match:
            month, day, year = us_match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # European format: 15/01/2024
        eu_match = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', query)
        if eu_match:
            # Assume day/month/year if day > 12
            part1, part2, year = eu_match.groups()
            if int(part1) > 12:  # Must be day/month/year
                day, month = part1, part2
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Written format: January 15 2024, 15 Jan 2024, etc.
        written_patterns = [
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # January 15, 2024
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',   # 15 January 2024
        ]
        
        for pattern in written_patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    # Try to parse as date
                    try:
                        if groups[0].isalpha():  # Month name first
                            month_name, day, year = groups
                            month_num = self.month_names.get(month_name.lower())
                            if month_num:
                                return f"{year}-{month_num}-{day.zfill(2)}"
                        else:  # Day first
                            day, month_name, year = groups
                            month_num = self.month_names.get(month_name.lower())
                            if month_num:
                                return f"{year}-{month_num}-{day.zfill(2)}"
                    except:
                        continue
        
        return None
    
    def _extract_month_pattern(self, query: str) -> Optional[str]:
        """Extract month-based patterns like 'January 2024'."""
        # Pattern: January 2024, Jan 2024
        month_year_pattern = r'\b(\w+)\s+(\d{4})\b'
        match = re.search(month_year_pattern, query)
        if match:
            month_name, year = match.groups()
            month_num = self.month_names.get(month_name.lower())
            if month_num:
                return f"{month_name.lower()}_{year}"
        
        return None
    
    def resolve_temporal_pattern(self, pattern: str) -> Dict[str, Any]:
        """Resolve temporal pattern to actual dates with enhanced logic."""
        if not pattern:
            return {}
        
        today = self.today
        
        # Handle date ranges
        if "_to_" in pattern:
            start_date, end_date = pattern.split("_to_")
            return {
                "start_date": start_date,
                "end_date": end_date,
                "detected_pattern": pattern
            }
        
        # Handle specific dates (ISO format)
        if re.match(r'\d{4}-\d{2}-\d{2}', pattern):
            return {
                "start_date": pattern,
                "end_date": pattern,
                "detected_pattern": pattern
            }
        
        # Handle month/year patterns
        if re.match(r'\w+_\d{4}', pattern):
            parts = pattern.split('_')
            if len(parts) == 2:
                month_name, year = parts
                month_num = self.month_names.get(month_name.lower())
                if month_num:
                    # Get first and last day of the month
                    first_day = f"{year}-{month_num}-01"
                    last_day_num = calendar.monthrange(int(year), int(month_num))[1]
                    last_day = f"{year}-{month_num}-{last_day_num:02d}"
                    return {
                        "start_date": first_day,
                        "end_date": last_day,
                        "detected_pattern": pattern
                    }

        # Enhanced pattern resolvers
        def resolve_yesterday():
            return {
                "start_date": (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                "end_date": (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                "detected_pattern": "yesterday"
            }

        def resolve_day_before_yesterday():
            return {
                "start_date": (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                "end_date": (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                "detected_pattern": "day_before_yesterday"
            }

        def resolve_two_days_ago():
            return {
                "start_date": (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                "end_date": (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                "detected_pattern": "two_days_ago"
            }

        def resolve_three_days_ago():
            return {
                "start_date": (today - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
                "end_date": (today - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
                "detected_pattern": "three_days_ago"
            }

        def resolve_last_week():
            return {
                "start_date": (today - datetime.timedelta(weeks=1)).strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "detected_pattern": "last_week"
            }

        def resolve_past_week():
            return {
                "start_date": (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "detected_pattern": "past_week"
            }

        def resolve_past_7_days():
            return {
                "start_date": (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "detected_pattern": "past_7_days"
            }

        def resolve_two_weeks_ago():
            return {
                "start_date": (today - datetime.timedelta(weeks=2)).strftime("%Y-%m-%d"),
                "end_date": (today - datetime.timedelta(weeks=2)).strftime("%Y-%m-%d"),
                "detected_pattern": "two_weeks_ago"
            }

        def resolve_last_month():
            return {
                "start_date": (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "detected_pattern": "last_month"
            }

        def resolve_past_month():
            return {
                "start_date": (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "detected_pattern": "past_month"
            }

        def resolve_month_ago():
            return {
                "start_date": (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                "detected_pattern": "month_ago"
            }

        def resolve_this_week():
            return {
                "start_date": (today - datetime.timedelta(days=today.weekday())).strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "detected_pattern": "this_week"
            }

        def resolve_this_month():
            return {
                "start_date": today.replace(day=1).strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d"),
                "detected_pattern": "this_month"
            }

        def resolve_beginning_of_month():
            return {
                "start_date": today.replace(day=1).strftime("%Y-%m-%d"),
                "end_date": (today.replace(day=1) + datetime.timedelta(days=6)).strftime("%Y-%m-%d"),
                "detected_pattern": "beginning_of_month"
            }

        def resolve_first_quarter():
            return {
                "start_date": today.replace(month=1, day=1).strftime("%Y-%m-%d"),
                "end_date": today.replace(month=3, day=31).strftime("%Y-%m-%d"),
                "detected_pattern": "first_quarter"
            }

        def resolve_monday_last_week():
            days_since_monday = (today.weekday() + 7) % 7  # Days since last Monday
            last_monday = today - datetime.timedelta(days=days_since_monday + 7)
            return {
                "start_date": last_monday.strftime("%Y-%m-%d"),
                "end_date": last_monday.strftime("%Y-%m-%d"),
                "detected_pattern": "monday_last_week"
            }

        def resolve_last_tuesday():
            # Find the most recent Tuesday
            days_since_tuesday = (today.weekday() - 1) % 7
            if days_since_tuesday == 0:  # Today is Tuesday
                days_since_tuesday = 7
            last_tuesday = today - datetime.timedelta(days=days_since_tuesday)
            return {
                "start_date": last_tuesday.strftime("%Y-%m-%d"),
                "end_date": last_tuesday.strftime("%Y-%m-%d"),
                "detected_pattern": "last_tuesday"
            }

        def resolve_weekend():
            # Find last weekend (Saturday-Sunday)
            days_since_saturday = (today.weekday() + 2) % 7
            if days_since_saturday < 2:  # This weekend
                days_since_saturday += 7
            last_saturday = today - datetime.timedelta(days=days_since_saturday)
            last_sunday = last_saturday + datetime.timedelta(days=1)
            return {
                "start_date": last_saturday.strftime("%Y-%m-%d"),
                "end_date": last_sunday.strftime("%Y-%m-%d"),
                "detected_pattern": "weekend"
            }

        enhanced_resolvers = {
            "yesterday": resolve_yesterday,
            "day_before_yesterday": resolve_day_before_yesterday,
            "two_days_ago": resolve_two_days_ago,
            "three_days_ago": resolve_three_days_ago,
            "last_week": resolve_last_week,
            "past_week": resolve_past_week,
            "past_7_days": resolve_past_7_days,
            "two_weeks_ago": resolve_two_weeks_ago,
            "last_month": resolve_last_month,
            "past_month": resolve_past_month,
            "month_ago": resolve_month_ago,
            "this_week": resolve_this_week,
            "this_month": resolve_this_month,
            "beginning_of_month": resolve_beginning_of_month,
            "first_quarter": resolve_first_quarter,
            "monday_last_week": resolve_monday_last_week,
            "last_tuesday": resolve_last_tuesday,
            "weekend": resolve_weekend
        }
        
        resolver = enhanced_resolvers.get(pattern)
        if resolver:
            return resolver()
        
        # Fallback to empty if no pattern recognized
        return {}

# Global instance for easy access
enhanced_temporal = EnhancedTemporalExtractor() 
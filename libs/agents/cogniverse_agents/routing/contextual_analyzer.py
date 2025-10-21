"""
Contextual Analyzer for Multi-Modal Search

Maintains cross-modal context across queries to improve routing and search quality.
Tracks conversation history, modality preferences, and topic evolution.
"""

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional


@dataclass
class QueryContext:
    """
    Single query context entry

    Attributes:
        query: User query text
        modalities: Detected modalities
        timestamp: When query was made
        result_count: Number of results returned
        user_feedback: Optional user feedback/interaction
    """

    query: str
    modalities: List[str]
    timestamp: datetime
    result_count: int = 0
    user_feedback: Optional[Dict[str, Any]] = None


class ContextualAnalyzer:
    """
    Maintain cross-modal context across queries

    Features:
    - Conversation history tracking
    - Modality preference learning
    - Topic evolution tracking
    - Temporal pattern recognition
    - Context-aware routing hints
    """

    def __init__(
        self,
        max_history_size: int = 50,
        context_window_minutes: int = 30,
        min_preference_count: int = 3,
    ):
        """
        Initialize contextual analyzer

        Args:
            max_history_size: Maximum conversation history to maintain
            context_window_minutes: Time window for recent context
            min_preference_count: Minimum occurrences to establish preference
        """
        self.max_history_size = max_history_size
        self.context_window = timedelta(minutes=context_window_minutes)
        self.min_preference_count = min_preference_count

        # Conversation tracking
        self.conversation_history: Deque[QueryContext] = deque(maxlen=max_history_size)

        # Learned preferences
        self.modality_preferences: Dict[str, int] = {}
        self.topic_tracking: Dict[str, List[datetime]] = {}
        self.temporal_patterns: Dict[str, Any] = {}

        # Session metadata
        self.session_start = datetime.now()
        self.total_queries = 0
        self.successful_queries = 0

    def update_context(
        self,
        query: str,
        detected_modalities: List[str],
        result: Optional[Any] = None,
        result_count: int = 0,
    ):
        """
        Track user query and update context

        Args:
            query: User query text
            detected_modalities: List of detected modality strings
            result: Optional result object
            result_count: Number of results returned
        """
        # Create context entry
        context = QueryContext(
            query=query,
            modalities=detected_modalities,
            timestamp=datetime.now(),
            result_count=result_count,
        )

        # Add to history
        self.conversation_history.append(context)
        self.total_queries += 1

        if result_count > 0:
            self.successful_queries += 1

        # Update modality preferences
        for modality in detected_modalities:
            self.modality_preferences[modality] = (
                self.modality_preferences.get(modality, 0) + 1
            )

        # Update topic tracking
        self._update_topic_tracking(query)

        # Update temporal patterns
        self._update_temporal_patterns(query, datetime.now())

    def _update_topic_tracking(self, query: str):
        """
        Extract and track topics from query

        Simple keyword-based topic extraction
        """
        # Extract potential topics (words longer than 3 chars)
        words = query.lower().split()
        topics = [w for w in words if len(w) > 3 and w.isalnum()]

        for topic in topics:
            if topic not in self.topic_tracking:
                self.topic_tracking[topic] = []
            self.topic_tracking[topic].append(datetime.now())

            # Keep only recent mentions (last 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            self.topic_tracking[topic] = [
                ts for ts in self.topic_tracking[topic] if ts > cutoff
            ]

    def _update_temporal_patterns(self, query: str, timestamp: datetime):
        """Track temporal usage patterns"""
        hour = timestamp.hour

        if "hourly_distribution" not in self.temporal_patterns:
            self.temporal_patterns["hourly_distribution"] = Counter()

        self.temporal_patterns["hourly_distribution"][hour] += 1

    def get_contextual_hints(self, current_query: str) -> Dict[str, Any]:
        """
        Provide routing hints based on conversation history

        Args:
            current_query: Current user query

        Returns:
            Dictionary with contextual hints
        """
        hints = {
            "preferred_modalities": self._get_top_modalities(),
            "related_topics": self._get_related_topics(current_query),
            "temporal_context": self._get_temporal_context(),
            "conversation_context": self._get_conversation_context(),
            "session_metrics": self._get_session_metrics(),
        }

        return hints

    def _get_top_modalities(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get top preferred modalities

        Args:
            top_k: Number of top modalities to return

        Returns:
            List of modality preferences with counts
        """
        if not self.modality_preferences:
            return []

        # Sort by count
        sorted_modalities = sorted(
            self.modality_preferences.items(), key=lambda x: x[1], reverse=True
        )

        # Filter by minimum count threshold
        top_modalities = [
            {
                "modality": modality,
                "count": count,
                "confidence": count / self.total_queries,
            }
            for modality, count in sorted_modalities[:top_k]
            if count >= self.min_preference_count
        ]

        return top_modalities

    def _get_related_topics(self, current_query: str) -> List[Dict[str, Any]]:
        """
        Get topics related to current query from history

        Args:
            current_query: Current user query

        Returns:
            List of related topics with relevance scores
        """
        if not self.topic_tracking:
            return []

        current_words = set(current_query.lower().split())
        related = []

        for topic, timestamps in self.topic_tracking.items():
            # Check if topic is in current query
            if topic in current_words:
                related.append(
                    {
                        "topic": topic,
                        "mention_count": len(timestamps),
                        "last_mentioned": timestamps[-1] if timestamps else None,
                        "relevance": 1.0,  # Direct match
                    }
                )
            # Check for partial matches (topic contains query word or vice versa)
            elif any(
                word in topic or topic in word
                for word in current_words
                if len(word) > 3
            ):
                related.append(
                    {
                        "topic": topic,
                        "mention_count": len(timestamps),
                        "last_mentioned": timestamps[-1] if timestamps else None,
                        "relevance": 0.7,  # Partial match
                    }
                )

        # Sort by relevance and recency
        related.sort(key=lambda x: (x["relevance"], x["mention_count"]), reverse=True)

        return related[:5]  # Return top 5

    def _get_temporal_context(self) -> Dict[str, Any]:
        """
        Get temporal context from recent queries

        Returns:
            Temporal context information
        """
        recent_queries = self._get_recent_queries()

        if not recent_queries:
            return {"has_recent_context": False}

        # Extract temporal patterns from recent queries
        return {
            "has_recent_context": True,
            "recent_query_count": len(recent_queries),
            "time_since_last_query": (
                datetime.now() - recent_queries[-1].timestamp
            ).total_seconds(),
            "active_session": (
                datetime.now() - recent_queries[-1].timestamp
            ).total_seconds()
            < 300,  # 5 minutes
        }

    def _get_conversation_context(self) -> Dict[str, Any]:
        """
        Get conversation flow context

        Returns:
            Conversation context information
        """
        recent_queries = self._get_recent_queries()

        if not recent_queries:
            return {
                "conversation_depth": 0,
                "modality_shifts": 0,
                "topic_consistency": 0.0,
            }

        # Count modality shifts
        modality_shifts = 0
        for i in range(1, len(recent_queries)):
            prev_modalities = set(recent_queries[i - 1].modalities)
            curr_modalities = set(recent_queries[i].modalities)
            if prev_modalities != curr_modalities:
                modality_shifts += 1

        # Calculate topic consistency
        all_topics = []
        for q in recent_queries:
            words = [w for w in q.query.lower().split() if len(w) > 3]
            all_topics.extend(words)

        if all_topics:
            topic_counts = Counter(all_topics)
            most_common_count = topic_counts.most_common(1)[0][1] if topic_counts else 0
            topic_consistency = most_common_count / len(all_topics)
        else:
            topic_consistency = 0.0

        return {
            "conversation_depth": len(recent_queries),
            "modality_shifts": modality_shifts,
            "topic_consistency": topic_consistency,
            "is_exploratory": modality_shifts > len(recent_queries) / 2,
            "is_focused": topic_consistency > 0.5,
        }

    def _get_session_metrics(self) -> Dict[str, Any]:
        """
        Get session performance metrics

        Returns:
            Session metrics
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()

        return {
            "session_duration_seconds": session_duration,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": (
                self.successful_queries / self.total_queries
                if self.total_queries > 0
                else 0.0
            ),
            "queries_per_minute": (
                self.total_queries / (session_duration / 60)
                if session_duration > 0
                else 0.0
            ),
        }

    def _get_recent_queries(
        self, window: Optional[timedelta] = None
    ) -> List[QueryContext]:
        """
        Get queries within recent time window

        Args:
            window: Time window (defaults to context_window)

        Returns:
            List of recent query contexts
        """
        window = window or self.context_window
        cutoff = datetime.now() - window

        return [q for q in self.conversation_history if q.timestamp > cutoff]

    def get_modality_transition_patterns(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze modality transition patterns

        Returns:
            Dictionary mapping (from_modality, to_modality) -> count
        """
        transitions = {}

        for i in range(1, len(self.conversation_history)):
            prev_modalities = self.conversation_history[i - 1].modalities
            curr_modalities = self.conversation_history[i].modalities

            for prev_mod in prev_modalities:
                if prev_mod not in transitions:
                    transitions[prev_mod] = {}

                for curr_mod in curr_modalities:
                    transitions[prev_mod][curr_mod] = (
                        transitions[prev_mod].get(curr_mod, 0) + 1
                    )

        return transitions

    def suggest_next_modality(self, current_modalities: List[str]) -> Optional[str]:
        """
        Suggest next modality based on patterns

        Args:
            current_modalities: Current query modalities

        Returns:
            Suggested modality or None
        """
        transitions = self.get_modality_transition_patterns()

        suggestions = Counter()
        for modality in current_modalities:
            if modality in transitions:
                suggestions.update(transitions[modality])

        if suggestions:
            return suggestions.most_common(1)[0][0]

        return None

    def clear_context(self):
        """Clear conversation context (start fresh session)"""
        self.conversation_history.clear()
        self.modality_preferences.clear()
        self.topic_tracking.clear()
        self.temporal_patterns.clear()
        self.session_start = datetime.now()
        self.total_queries = 0
        self.successful_queries = 0

    def export_context(self) -> Dict[str, Any]:
        """
        Export context for persistence

        Returns:
            Serializable context dictionary
        """
        return {
            "conversation_history": [
                {
                    "query": q.query,
                    "modalities": q.modalities,
                    "timestamp": q.timestamp.isoformat(),
                    "result_count": q.result_count,
                }
                for q in self.conversation_history
            ],
            "modality_preferences": self.modality_preferences,
            "topic_tracking": {
                topic: [ts.isoformat() for ts in timestamps]
                for topic, timestamps in self.topic_tracking.items()
            },
            "session_start": self.session_start.isoformat(),
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
        }

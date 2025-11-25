"""
Macro Evaluator for strategic reasoning assessment.

Evaluates the agent's macro analysis against expert-authored ground truth,
scoring on semantic similarity and theme coverage.
"""

import re
from typing import Optional

import structlog

from cio_agent.models import MacroScore, GroundTruth

logger = structlog.get_logger()


class MacroEvaluator:
    """
    Evaluates agent's macro analysis quality.

    Scoring criteria (30% of Role Score):
    - Semantic similarity to ground truth thesis: 60%
    - Key theme coverage: 40%

    Key themes include:
    - Sector trends (e.g., "AI chip demand", "cloud growth")
    - Macroeconomic factors (e.g., "Fed policy", "inflation")
    - Industry-specific risks (e.g., "regulation", "competition")
    """

    def __init__(
        self,
        ground_truth: GroundTruth,
        use_embeddings: bool = False,
    ):
        """
        Initialize the macro evaluator.

        Args:
            ground_truth: Ground truth containing macro thesis and key themes
            use_embeddings: Whether to use sentence embeddings for similarity
                           (requires sentence-transformers)
        """
        self.ground_truth = ground_truth
        self.use_embeddings = use_embeddings
        self._embedder = None

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence_transformers not available, using keyword matching")
                self.use_embeddings = False

    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using keyword overlap.

        Simple approach: Jaccard similarity of word sets.
        """
        # Normalize and tokenize
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        # Remove common stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "so", "yet", "both", "either",
            "neither", "not", "only", "own", "same", "than", "too",
            "very", "just", "also", "now", "here", "there", "when",
            "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no",
            "any", "this", "that", "these", "those", "it", "its",
        }

        words1 = words1 - stopwords
        words2 = words2 - stopwords

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using sentence embeddings.
        """
        if not self._embedder:
            return self._calculate_keyword_similarity(text1, text2)

        import numpy as np

        embeddings = self._embedder.encode([text1, text2])
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def _count_themes_mentioned(
        self,
        analysis: str,
        themes: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Count how many key themes are mentioned in the analysis.

        Returns:
            Tuple of (themes_found, themes_missed)
        """
        analysis_lower = analysis.lower()
        found = []
        missed = []

        for theme in themes:
            # Check for theme or related keywords
            theme_lower = theme.lower()
            theme_words = theme_lower.split()

            # Check if theme phrase appears
            if theme_lower in analysis_lower:
                found.append(theme)
                continue

            # Check if most theme words appear
            words_found = sum(1 for w in theme_words if w in analysis_lower)
            if words_found >= len(theme_words) * 0.6:  # 60% of words
                found.append(theme)
                continue

            missed.append(theme)

        return found, missed

    def score(self, agent_macro_analysis: str) -> MacroScore:
        """
        Score the agent's macro analysis.

        Args:
            agent_macro_analysis: The agent's macro analysis text

        Returns:
            MacroScore with detailed breakdown
        """
        if not agent_macro_analysis:
            return MacroScore(
                score=0.0,
                similarity_score=0.0,
                theme_coverage=0.0,
                themes_identified=[],
                themes_missed=self.ground_truth.key_themes,
                feedback="No macro analysis provided."
            )

        # Calculate semantic similarity
        if self.use_embeddings:
            similarity_score = self._calculate_embedding_similarity(
                agent_macro_analysis,
                self.ground_truth.macro_thesis
            ) * 100
        else:
            similarity_score = self._calculate_keyword_similarity(
                agent_macro_analysis,
                self.ground_truth.macro_thesis
            ) * 100

        # Calculate theme coverage
        themes_found, themes_missed = self._count_themes_mentioned(
            agent_macro_analysis,
            self.ground_truth.key_themes
        )

        if self.ground_truth.key_themes:
            theme_coverage = len(themes_found) / len(self.ground_truth.key_themes)
        else:
            theme_coverage = 1.0  # No themes to check

        # Combined score (60% similarity, 40% theme coverage)
        combined_score = 0.6 * similarity_score + 0.4 * theme_coverage * 100

        # Generate feedback
        feedback_parts = []

        if similarity_score >= 70:
            feedback_parts.append("Strong alignment with expected macro thesis.")
        elif similarity_score >= 40:
            feedback_parts.append("Partial alignment with expected macro thesis.")
        else:
            feedback_parts.append("Weak alignment with expected macro thesis.")

        if theme_coverage >= 0.8:
            feedback_parts.append(f"Excellent theme coverage ({len(themes_found)}/{len(self.ground_truth.key_themes)} themes).")
        elif theme_coverage >= 0.5:
            feedback_parts.append(f"Moderate theme coverage ({len(themes_found)}/{len(self.ground_truth.key_themes)} themes).")
        else:
            feedback_parts.append(f"Poor theme coverage ({len(themes_found)}/{len(self.ground_truth.key_themes)} themes).")

        if themes_missed:
            feedback_parts.append(f"Missed themes: {', '.join(themes_missed[:3])}.")

        logger.info(
            "macro_evaluation",
            similarity=similarity_score,
            theme_coverage=theme_coverage,
            themes_found=len(themes_found),
            combined_score=combined_score,
        )

        return MacroScore(
            score=combined_score,
            similarity_score=similarity_score,
            theme_coverage=theme_coverage,
            themes_identified=themes_found,
            themes_missed=themes_missed,
            feedback=" ".join(feedback_parts),
        )

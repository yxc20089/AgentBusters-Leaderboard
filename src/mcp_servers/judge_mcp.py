"""
MCP Judge Server for Semantic/Numeric/Contradiction Evaluation.

Provides evaluation-as-a-service via MCP protocol:
- semantic_evaluate: Compare answers using embedding similarity or LLM
- numeric_evaluate: Compare numerical values with tolerance
- contradiction_detect: Check if answer contradicts reference
- hallucination_detect: Check if answer contains fabricated data
- combined_judge: Run all evaluations and return composite score

This follows the Phase1 architecture pattern of separating
evaluation logic into dedicated MCP services.
"""

import os
import re
import logging
from typing import Any, Optional, Literal
from dataclasses import dataclass

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Judge MCP Server")


@dataclass
class JudgeResult:
    """Result from a judge evaluation."""
    score: float  # 0.0 to 1.0
    passed: bool
    method: str
    details: dict[str, Any]
    confidence: float  # 0.0 to 1.0


# ============================================================================
# NUMERIC JUDGE
# ============================================================================

def extract_number(text: str) -> Optional[float]:
    """Extract numerical value from text."""
    if not text:
        return None
    
    # Try direct parsing
    try:
        return float(text.strip())
    except ValueError:
        pass
    
    # Patterns for various number formats
    patterns = [
        r'(-?\d+\.?\d*)\s*%',  # Percentage
        r'\$\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|B)',  # Billions
        r'\$\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M)',  # Millions
        r'\$\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Currency
        r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Number with commas
        r'(-?\d+\.?\d*)',  # Simple number
    ]
    
    multipliers = {
        'billion': 1e9, 'B': 1e9,
        'million': 1e6, 'M': 1e6,
    }
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                value = float(num_str)
                # Check for multiplier
                for mult_key, mult_val in multipliers.items():
                    if mult_key.lower() in text.lower():
                        value *= mult_val
                        break
                return value
            except ValueError:
                continue
    
    return None


@mcp.tool()
def numeric_evaluate(
    predicted: str,
    expected: str,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """
    Evaluate numerical accuracy with configurable tolerance.
    
    Args:
        predicted: The predicted answer (may contain text around number)
        expected: The expected answer (ground truth)
        tolerance: Relative tolerance (default 1%)
        
    Returns:
        JudgeResult dict with score, passed, and details
    """
    pred_num = extract_number(predicted)
    exp_num = extract_number(expected)
    
    if pred_num is None:
        return {
            "score": 0.0,
            "passed": False,
            "method": "numeric_extraction_failed",
            "details": {
                "error": f"Could not extract number from predicted: '{predicted[:100]}'",
                "predicted_raw": predicted[:200],
                "expected_raw": expected[:200],
            },
            "confidence": 1.0,
        }
    
    if exp_num is None:
        return {
            "score": 0.0,
            "passed": False,
            "method": "numeric_extraction_failed",
            "details": {
                "error": f"Could not extract number from expected: '{expected[:100]}'",
                "predicted_raw": predicted[:200],
                "expected_raw": expected[:200],
            },
            "confidence": 1.0,
        }
    
    # Calculate relative error
    if exp_num == 0:
        is_correct = abs(pred_num) < tolerance
        relative_error = abs(pred_num) if pred_num != 0 else 0
    else:
        relative_error = abs(pred_num - exp_num) / abs(exp_num)
        is_correct = relative_error <= tolerance
    
    # Graduated scoring: partial credit for close answers
    if is_correct:
        score = 1.0
    elif relative_error <= tolerance * 2:
        score = 0.75
    elif relative_error <= tolerance * 5:
        score = 0.5
    elif relative_error <= tolerance * 10:
        score = 0.25
    else:
        score = 0.0
    
    return {
        "score": score,
        "passed": is_correct,
        "method": "numeric_tolerance",
        "details": {
            "predicted_value": pred_num,
            "expected_value": exp_num,
            "relative_error": relative_error,
            "tolerance": tolerance,
            "within_tolerance": is_correct,
        },
        "confidence": 0.95,
    }


# ============================================================================
# SEMANTIC JUDGE
# ============================================================================

def extract_key_elements(text: str) -> set[str]:
    """Extract key semantic elements from text."""
    elements = set()
    text_lower = text.lower()
    
    # Numbers with context
    numbers = re.findall(
        r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|%|bps))?',
        text_lower
    )
    elements.update(numbers)
    
    # Stock tickers
    tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
    elements.update(t.lower() for t in tickers)
    
    # Key financial terms
    fin_terms = re.findall(
        r'\b(?:revenue|earnings|eps|margin|growth|decline|beat|miss|'
        r'bullish|bearish|guidance|forecast|estimate|consensus|'
        r'quarter|q[1-4]|fy\d{2,4}|yoy|qoq)\b',
        text_lower
    )
    elements.update(fin_terms)
    
    # Named entities (capitalized multi-word)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    elements.update(n.lower() for n in names)
    
    return elements


@mcp.tool()
def semantic_evaluate(
    predicted: str,
    expected: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate semantic similarity between predicted and expected answers.
    
    Uses key element extraction for fast, interpretable matching.
    For LLM-based deep semantic comparison, use combined_judge.
    
    Args:
        predicted: The predicted answer
        expected: The expected/reference answer
        threshold: Minimum overlap ratio to pass (default 0.5)
        
    Returns:
        JudgeResult dict with score, passed, and details
    """
    pred_elements = extract_key_elements(predicted)
    exp_elements = extract_key_elements(expected)
    
    if not exp_elements:
        # No extractable elements, fall back to substring check
        pred_lower = predicted.lower()
        exp_lower = expected.lower()
        
        if exp_lower in pred_lower:
            return {
                "score": 1.0,
                "passed": True,
                "method": "substring_match",
                "details": {"match_type": "expected_in_predicted"},
                "confidence": 0.7,
            }
        
        # Word overlap
        pred_words = set(pred_lower.split())
        exp_words = set(exp_lower.split())
        overlap = len(pred_words & exp_words)
        total = len(exp_words)
        
        if total == 0:
            return {
                "score": 0.0,
                "passed": False,
                "method": "no_elements",
                "details": {"error": "No elements to compare"},
                "confidence": 0.5,
            }
        
        score = overlap / total
        return {
            "score": score,
            "passed": score >= threshold,
            "method": "word_overlap",
            "details": {
                "overlap_words": overlap,
                "total_expected_words": total,
                "overlap_ratio": score,
            },
            "confidence": 0.6,
        }
    
    # Element-based matching
    overlap = pred_elements & exp_elements
    missing = exp_elements - pred_elements
    extra = pred_elements - exp_elements
    
    overlap_ratio = len(overlap) / len(exp_elements) if exp_elements else 0
    
    return {
        "score": overlap_ratio,
        "passed": overlap_ratio >= threshold,
        "method": "key_element_matching",
        "details": {
            "matched_elements": list(overlap)[:10],
            "missing_elements": list(missing)[:10],
            "extra_elements": list(extra)[:10],
            "predicted_element_count": len(pred_elements),
            "expected_element_count": len(exp_elements),
            "overlap_ratio": overlap_ratio,
        },
        "confidence": 0.8,
    }


# ============================================================================
# CONTRADICTION JUDGE
# ============================================================================

# Semantic opposites for contradiction detection
OPPOSITES = [
    ("beat", "miss"),
    ("bullish", "bearish"),
    ("buy", "sell"),
    ("increase", "decrease"),
    ("growth", "decline"),
    ("positive", "negative"),
    ("strong", "weak"),
    ("outperform", "underperform"),
    ("above", "below"),
    ("higher", "lower"),
    ("profit", "loss"),
    ("surplus", "deficit"),
    ("expand", "contract"),
    ("rise", "fall"),
    ("gain", "lose"),
]


def detect_stance(text: str) -> Optional[str]:
    """Detect bullish/bearish stance in text."""
    text_lower = text.lower()
    
    bullish_signals = sum(1 for w in ["buy", "bullish", "strong", "outperform", "beat", "growth"] if w in text_lower)
    bearish_signals = sum(1 for w in ["sell", "bearish", "weak", "underperform", "miss", "decline"] if w in text_lower)
    
    if bullish_signals > bearish_signals:
        return "bullish"
    elif bearish_signals > bullish_signals:
        return "bearish"
    return None


@mcp.tool()
def contradiction_detect(
    predicted: str,
    reference: str,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Detect if predicted answer contradicts the reference.
    
    Checks for:
    - Stance reversal (bullish vs bearish)
    - Semantic opposites
    - Numerical contradictions (significant differences)
    
    Args:
        predicted: The predicted answer to check
        reference: The reference that should not be contradicted
        strict: If True, use stricter contradiction detection
        
    Returns:
        JudgeResult dict with contradiction status
    """
    pred_lower = predicted.lower()
    ref_lower = reference.lower()
    
    contradictions_found = []
    
    # 1. Stance contradiction
    pred_stance = detect_stance(predicted)
    ref_stance = detect_stance(reference)
    
    if pred_stance and ref_stance and pred_stance != ref_stance:
        contradictions_found.append({
            "type": "stance_reversal",
            "predicted_stance": pred_stance,
            "reference_stance": ref_stance,
            "severity": "high",
        })
    
    # 2. Semantic opposites
    for word1, word2 in OPPOSITES:
        # Check if reference has word1 and predicted has word2 (or vice versa)
        if word1 in ref_lower and word2 in pred_lower and word1 not in pred_lower:
            contradictions_found.append({
                "type": "semantic_opposite",
                "reference_term": word1,
                "predicted_term": word2,
                "severity": "medium",
            })
        elif word2 in ref_lower and word1 in pred_lower and word2 not in pred_lower:
            contradictions_found.append({
                "type": "semantic_opposite",
                "reference_term": word2,
                "predicted_term": word1,
                "severity": "medium",
            })
    
    # 3. Numerical contradiction (numbers differ by > 50%)
    ref_numbers = re.findall(r'\d+\.?\d*', reference)
    pred_numbers = re.findall(r'\d+\.?\d*', predicted)
    
    for ref_num_str in ref_numbers[:5]:  # Check first 5 numbers
        try:
            ref_num = float(ref_num_str)
            if ref_num == 0:
                continue
            for pred_num_str in pred_numbers[:5]:
                try:
                    pred_num = float(pred_num_str)
                    if pred_num == 0:
                        continue
                    # Check if numbers are in similar magnitude but significantly different
                    if 0.1 < ref_num / pred_num < 10:  # Similar magnitude
                        diff = abs(ref_num - pred_num) / ref_num
                        if diff > 0.5:  # >50% different
                            contradictions_found.append({
                                "type": "numerical_mismatch",
                                "reference_value": ref_num,
                                "predicted_value": pred_num,
                                "difference_pct": diff * 100,
                                "severity": "low" if diff < 1.0 else "medium",
                            })
                except ValueError:
                    continue
        except ValueError:
            continue
    
    # Calculate score
    high_severity = sum(1 for c in contradictions_found if c.get("severity") == "high")
    medium_severity = sum(1 for c in contradictions_found if c.get("severity") == "medium")
    low_severity = sum(1 for c in contradictions_found if c.get("severity") == "low")
    
    total_penalty = high_severity * 1.0 + medium_severity * 0.5 + low_severity * 0.2
    score = max(0.0, 1.0 - total_penalty)
    
    has_contradiction = len(contradictions_found) > 0
    if strict:
        passed = not has_contradiction
    else:
        passed = high_severity == 0 and medium_severity < 2
    
    return {
        "score": score,
        "passed": passed,
        "method": "multi_signal_contradiction",
        "details": {
            "has_contradiction": has_contradiction,
            "contradiction_count": len(contradictions_found),
            "contradictions": contradictions_found[:5],  # Limit output
            "high_severity_count": high_severity,
            "medium_severity_count": medium_severity,
            "low_severity_count": low_severity,
        },
        "confidence": 0.85 if contradictions_found else 0.7,
    }


# ============================================================================
# HALLUCINATION JUDGE
# ============================================================================

@mcp.tool()
def hallucination_detect(
    predicted: str,
    context: str = "",
    known_data: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Detect potential hallucinations in the predicted answer.
    
    Checks for:
    - Specific numbers not supported by context/known data
    - Confident claims without evidence
    - Fabricated entity names or dates
    
    Args:
        predicted: The answer to check for hallucinations
        context: The source context (if available)
        known_data: Dict of known financial values for verification
        
    Returns:
        JudgeResult dict with hallucination analysis
    """
    known_data = known_data or {}
    hallucination_signals = []
    
    # Extract specific numbers from prediction
    pred_numbers = re.findall(
        r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|million|%|bps)?',
        predicted,
        re.IGNORECASE
    )
    
    # Extract numbers from context
    context_numbers = set()
    if context:
        context_nums = re.findall(
            r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            context
        )
        context_numbers = {n.replace(',', '') for n in context_nums}
    
    # Check for unsupported specific numbers
    for num_str in pred_numbers:
        num_clean = num_str.replace(',', '')
        try:
            num_val = float(num_clean)
            # Skip small numbers (percentages, ratios)
            if num_val < 100:
                continue
            
            # Check if number appears in context
            if num_clean not in context_numbers:
                # Check if it matches any known data
                is_known = False
                for key, value in known_data.items():
                    if isinstance(value, (int, float)):
                        if abs(num_val - value) / max(value, 1) < 0.1:
                            is_known = True
                            break
                
                if not is_known and num_val > 1000:  # Significant number
                    hallucination_signals.append({
                        "type": "unsupported_number",
                        "value": num_val,
                        "severity": "medium" if num_val > 1000000 else "low",
                    })
        except ValueError:
            continue
    
    # Check for confident assertions without citations
    confident_patterns = [
        r'according to (?:the|their) (?:latest|recent)? ?(?:10-k|10-q|filing|report)',
        r'management (?:stated|confirmed|announced)',
        r'the company (?:reported|disclosed)',
    ]
    
    for pattern in confident_patterns:
        if re.search(pattern, predicted, re.IGNORECASE):
            if not context or pattern.split()[1] not in context.lower():
                hallucination_signals.append({
                    "type": "unverifiable_claim",
                    "pattern": pattern,
                    "severity": "low",
                })
    
    # Calculate score
    medium_count = sum(1 for s in hallucination_signals if s.get("severity") == "medium")
    low_count = sum(1 for s in hallucination_signals if s.get("severity") == "low")
    
    total_penalty = medium_count * 0.3 + low_count * 0.1
    score = max(0.0, 1.0 - total_penalty)
    
    return {
        "score": score,
        "passed": medium_count == 0,
        "method": "hallucination_detection",
        "details": {
            "signals_found": len(hallucination_signals),
            "signals": hallucination_signals[:5],
            "has_context": bool(context),
            "known_data_keys": list(known_data.keys())[:5] if known_data else [],
        },
        "confidence": 0.6,  # Hallucination detection is inherently uncertain
    }


# ============================================================================
# COMBINED JUDGE
# ============================================================================

@mcp.tool()
def combined_judge(
    predicted: str,
    expected: str,
    question: str = "",
    context: str = "",
    evaluation_type: Literal["numeric", "semantic", "both"] = "both",
    weights: dict[str, float] = None,
) -> dict[str, Any]:
    """
    Run comprehensive evaluation combining all judge methods.
    
    Args:
        predicted: The predicted answer
        expected: The expected/ground truth answer
        question: The original question (optional)
        context: Source context (optional)
        evaluation_type: Type of primary evaluation
        weights: Custom weights for score components
            Default: {"numeric": 0.4, "semantic": 0.3, "contradiction": 0.2, "hallucination": 0.1}
        
    Returns:
        Combined JudgeResult with all component scores
    """
    default_weights = {
        "numeric": 0.4,
        "semantic": 0.3,
        "contradiction": 0.2,
        "hallucination": 0.1,
    }
    weights = weights or default_weights
    
    results = {}
    
    # Run numeric evaluation
    if evaluation_type in ("numeric", "both"):
        numeric_result = numeric_evaluate(predicted, expected)
        results["numeric"] = numeric_result
    
    # Run semantic evaluation
    if evaluation_type in ("semantic", "both"):
        semantic_result = semantic_evaluate(predicted, expected)
        results["semantic"] = semantic_result
    
    # Always run contradiction detection
    contradiction_result = contradiction_detect(predicted, expected)
    results["contradiction"] = contradiction_result
    
    # Run hallucination detection if context provided
    hallucination_result = hallucination_detect(predicted, context)
    results["hallucination"] = hallucination_result
    
    # Calculate weighted score
    total_score = 0.0
    total_weight = 0.0
    
    for key, result in results.items():
        weight = weights.get(key, 0.1)
        # For contradiction/hallucination, we want high scores (no issues found)
        if key in ("contradiction", "hallucination"):
            total_score += result["score"] * weight
        else:
            total_score += result["score"] * weight
        total_weight += weight
    
    final_score = total_score / total_weight if total_weight > 0 else 0.0
    
    # Determine if passed overall
    passed = (
        (results.get("numeric", {}).get("passed", True) or evaluation_type == "semantic") and
        (results.get("semantic", {}).get("passed", True) or evaluation_type == "numeric") and
        results.get("contradiction", {}).get("passed", True)
    )
    
    return {
        "score": final_score,
        "passed": passed,
        "method": "combined_judge",
        "details": {
            "component_results": results,
            "weights_used": weights,
            "evaluation_type": evaluation_type,
        },
        "confidence": min(r.get("confidence", 0.5) for r in results.values()),
    }


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8107, help="Port to listen on")
    
    args = parser.parse_args()
    
    print(f"🧑‍⚖️ Starting Judge MCP Server on {args.host}:{args.port}")
    print("Available tools:")
    print("  - numeric_evaluate: Numerical matching with tolerance")
    print("  - semantic_evaluate: Key element semantic matching")
    print("  - contradiction_detect: Multi-signal contradiction detection")
    print("  - hallucination_detect: Fabricated data detection")
    print("  - combined_judge: Comprehensive combined evaluation")
    
    mcp.run(transport="sse", host=args.host, port=args.port)

import math
from .errors import ProviderError


def normalize_prediction(raw, text, model):
    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ProviderError("invalid_response", "Provider must return a finite score between zero and one")
    def label(s):
        return "human" if s < model["lower_threshold"] else "ai" if s >= model["upper_threshold"] else "ai_assisted"
    segments = []
    end_previous = 0
    raw_segments = raw.get("segments", [])
    if not isinstance(raw_segments, list) or len(raw_segments) > 5000:
        raise ProviderError("invalid_response", "Provider returned invalid segments")
    for segment in raw_segments:
        if not isinstance(segment, dict):
            raise ProviderError("invalid_response", "Provider returned invalid segments")
        start, end, s = segment.get("start"), segment.get("end"), segment.get("score")
        if (type(start) is not int or type(end) is not int or start < end_previous or not start < end <= len(text)
            or type(s) not in {int, float} or not math.isfinite(s) or not 0 <= s <= 1):
            raise ProviderError("invalid_response", "Provider returned invalid or overlapping segment offsets")
        segments.append({"start": start, "end": end, "score": s, "label": label(s)})
        end_previous = end
    return {"score": score, "score_type": "ai_intervention", "label": label(score), "segments": segments,
            "thresholds": {"human_below": model["lower_threshold"], "ai_at_or_above": model["upper_threshold"],
                           "calibration": "workspace_policy_not_benchmark_calibrated"},
            "notice": "A model score is not a probability of authorship or a percentage of AI-written words."}

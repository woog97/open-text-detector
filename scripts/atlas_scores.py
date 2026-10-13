"""Compact score summaries shared by publication and catalogue backfills."""
import math

def summarize(segments, text):
    histogram = [0] * 21
    excluded = 0
    for segment in segments:
        if not any(c.isalnum() for c in text[segment['start']:segment['end']]):
            excluded += 1
            continue
        score = segment.get('score')
        if isinstance(score,(int,float)) and math.isfinite(score) and 0 <= score <= 1:
            histogram[math.ceil(score * 20)] += 1
    return {'histogram':histogram,'total':sum(histogram), 'excluded':excluded, 'policy':2}

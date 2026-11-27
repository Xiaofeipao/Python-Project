"""Candidate proposal and resource allocation; estimator construction lives in models.factory."""
from .allocation import allocate_candidates, candidate_key, metric_candidates
from .candidates import next_candidate

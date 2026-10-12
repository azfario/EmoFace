"""Development-only threshold selection and genuinely held-out evaluation.

Raw pair data must remain private. Independent impostor comparisons are selected
deterministically before looking at scores, using each subject at most once.
The order-statistic tolerance bound prevents a large dependent pair matrix from
being mistaken for hundreds of independent people.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta
from .schemas import Status, VerificationError

@dataclass(frozen=True)
class Pair:
    score: float
    genuine: bool
    subject_a: str
    subject_b: str
    enrollment_hashes: tuple[str, ...]
    probe_hash: str
    enrollment_session: str
    probe_session: str

def validate_pairs(pairs):
    seen = set()
    for p in pairs:
        if not np.isfinite(p.score) or not -1 <= p.score <= 1 or type(p.genuine) is not bool:
            raise ValueError('Invalid score or label.')
        if not p.subject_a or not p.subject_b or p.genuine != (p.subject_a == p.subject_b):
            raise ValueError('Pair identity labels disagree.')
        if len(p.enrollment_hashes) < 3 or not p.probe_hash or p.probe_hash in p.enrollment_hashes or len(set(p.enrollment_hashes)) != len(p.enrollment_hashes):
            raise ValueError('Enrollment/probe leakage or duplicate enrollment frames.')
        if not p.enrollment_session or not p.probe_session or p.enrollment_session == p.probe_session:
            raise ValueError('Independent enrollment and probe sessions are required.')
        key = (p.subject_a, p.subject_b, p.enrollment_hashes, p.probe_hash)
        if key in seen:
            raise ValueError('Duplicate comparison.')
        seen.add(key)
    # A capture may recur as one side of multiple comparisons but never in both roles.
    enrollment = {h for p in pairs for h in p.enrollment_hashes}
    probes = {p.probe_hash for p in pairs}
    if enrollment & probes:
        raise ValueError('An enrollment image is also used as a probe.')

def subjects(pairs):
    return {s for p in pairs for s in (p.subject_a, p.subject_b)}

def independent_impostors(pairs):
    chosen, used = [], set()
    # Ordering is independent of scores and labels beyond the impostor filter.
    for p in sorted((p for p in pairs if not p.genuine), key=lambda p: (p.subject_a, p.subject_b, p.probe_hash)):
        if p.subject_a not in used and p.subject_b not in used:
            chosen.append(p)
            used.update((p.subject_a, p.subject_b))
    return chosen

def interval(errors, trials, confidence=0.95):
    if not trials:
        return None
    alpha = 1-confidence
    lo = 0.0 if errors == 0 else float(beta.ppf(alpha/2, errors, trials-errors+1))
    hi = 1.0 if errors == trials else float(beta.ppf(1-alpha/2, errors+1, trials-errors))
    return [lo, hi]

def confusion_metrics(pairs, threshold):
    genuine = [p for p in pairs if p.genuine]
    impostors = [p for p in pairs if not p.genuine]
    false_matches = sum(p.score >= threshold for p in impostors)
    false_nonmatches = sum(p.score < threshold for p in genuine)
    return {'genuine_pairs': len(genuine), 'impostor_pairs': len(impostors),
        'false_matches': false_matches, 'true_nonmatches': len(impostors)-false_matches,
        'false_nonmatches': false_nonmatches, 'true_matches': len(genuine)-false_nonmatches,
        'fmr': false_matches/len(impostors) if impostors else None,
        'fnmr': false_nonmatches/len(genuine) if genuine else None}

def calibrate_development(pairs, model_id, config, *, data_kind='real_consented'):
    """There is deliberately no test-set argument. Returns no cutoff if inadequate."""
    pairs = list(pairs)
    validate_pairs(pairs)
    independent = independent_impostors(pairs)
    n = len(independent)
    needed = math.ceil(math.log(1-config.confidence)/math.log(1-config.target_fmr))
    result = {'schema_version': 1, 'status': Status.INSUFFICIENT_DATA.value, 'threshold': None,
        'model_id': model_id, 'data_kind': data_kind, 'target_fmr': config.target_fmr,
        'confidence': config.confidence, 'development_subjects': len(subjects(pairs)),
        'genuine_pairs': sum(p.genuine for p in pairs), 'impostor_pairs': sum(not p.genuine for p in pairs),
        'independent_impostor_pairs': n, 'minimum_independent_impostors_for_zero_errors': needed,
        'method': 'subject-disjoint impostor order-statistic tolerance bound',
        'test_data_used_for_selection': False, 'production_ready': False}
    if len(subjects(pairs)) < config.min_development_subjects or result['genuine_pairs'] < config.min_genuine_pairs or n < needed:
        result['reason'] = 'Insufficient independent development identities/captures to support the prespecified FMR target.'
        return result
    # Select the permissible error rank by n and the prespecified policy alone.
    permissible = [k for k in range(n) if float(beta.ppf(config.confidence, k+1, n-k)) <= config.target_fmr]
    if not permissible:
        return result
    k = max(permissible)
    ordered = sorted(p.score for p in independent)
    threshold = float(np.nextafter(ordered[n-k-1], math.inf))
    if threshold > 1:
        result['reason'] = 'No valid cosine cutoff can satisfy the policy.'
        return result
    result.update(status='CALIBRATED_RESEARCH' if data_kind == 'real_consented' else 'SYNTHETIC_TEST_ONLY',
        threshold=threshold, independent_impostor_fmr_upper=float(beta.ppf(config.confidence, k+1, n-k)),
        development_metrics=confusion_metrics(pairs, threshold),
        development_protocol_digest=hashlib.sha256(json.dumps([asdict(p) for p in pairs], sort_keys=True).encode()).hexdigest())
    return result

def evaluate_held_out(development_pairs, test_pairs, calibration):
    dev, test = list(development_pairs), list(test_pairs)
    validate_pairs(dev)
    validate_pairs(test)
    if subjects(dev) & subjects(test):
        raise ValueError('Development and final-test subjects overlap.')
    dev_hashes = {h for p in dev for h in (*p.enrollment_hashes, p.probe_hash)}
    test_hashes = {h for p in test for h in (*p.enrollment_hashes, p.probe_hash)}
    if dev_hashes & test_hashes:
        raise ValueError('Development and test images overlap.')
    threshold = calibration.get('threshold')
    if threshold is None:
        return {'status': Status.INSUFFICIENT_DATA.value, 'threshold': None, 'metrics': None}
    digest = hashlib.sha256(json.dumps([asdict(p) for p in dev], sort_keys=True).encode()).hexdigest()
    if calibration.get('development_protocol_digest') != digest:
        raise ValueError('Development protocol differs from the frozen calibration artifact.')
    metrics = confusion_metrics(test, threshold)
    independent = independent_impostors(test)
    selected_genuine, used = [], set()
    for p in sorted((p for p in test if p.genuine), key=lambda p: (p.subject_a, p.probe_hash)):
        if p.subject_a not in used:
            selected_genuine.append(p)
            used.add(p.subject_a)
    metrics['independent_impostor_pairs'] = len(independent)
    metrics['independent_genuine_pairs'] = len(selected_genuine)
    confidence = calibration['confidence']
    metrics['independent_fmr_interval'] = interval(sum(p.score >= threshold for p in independent), len(independent), confidence)
    metrics['independent_fnmr_interval'] = interval(sum(p.score < threshold for p in selected_genuine), len(selected_genuine), confidence)
    metrics['interval_confidence'] = confidence
    target = calibration['target_fmr']
    independent_needed = math.ceil(math.log((1-confidence)/2)/math.log(1-target))
    sufficient = len(independent) >= independent_needed and len(selected_genuine) >= 30
    return {'status': 'HELD_OUT_RESEARCH_EVALUATION' if sufficient else Status.INSUFFICIENT_DATA.value,
        'threshold': threshold, 'metrics': metrics, 'independent_impostors_needed_for_zero_error_two_sided_bound': independent_needed,
        'production_ready': False, 'uncertainty_note': 'Exact intervals apply to the disjoint-subject subsets under independent representative sampling; all-pair point estimates contain dependent comparisons.'}

def save_calibration(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

def load_threshold(path, model_id, config):
    """Fail closed. Artifacts are local trusted configuration, not signed evidence."""
    try:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        threshold = report['threshold']
        needed = math.ceil(math.log(1-config.confidence)/math.log(1-config.target_fmr))
        if (report['schema_version'] != 1 or report['status'] != 'CALIBRATED_RESEARCH' or report['data_kind'] != 'real_consented'
            or report['model_id'] != model_id or report['test_data_used_for_selection'] is not False
            or report['target_fmr'] != config.target_fmr or report['confidence'] != config.confidence
            or report['development_subjects'] < config.min_development_subjects or report['genuine_pairs'] < config.min_genuine_pairs
            or report['independent_impostor_pairs'] < needed or report['independent_impostor_fmr_upper'] > config.target_fmr
            or not report['development_protocol_digest'] or type(threshold) not in (float, int)
            or not np.isfinite(threshold) or not -1 <= threshold <= 1):
            return None
        return float(threshold)
    except (OSError, ValueError, KeyError, TypeError):
        return None

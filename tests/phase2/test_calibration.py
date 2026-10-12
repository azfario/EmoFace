from dataclasses import replace
import numpy as np
import pytest
from face_verification.calibration import Pair, calibrate_development, evaluate_held_out, confusion_metrics, save_calibration, load_threshold

def synthetic_pairs(prefix='dev', n=310):
    result = []
    for i in range(n):
        a, b = f'{prefix}_{i}_a', f'{prefix}_{i}_b'
        hashes = tuple(f'{a}_enrollment_{j}' for j in range(3))
        result.append(Pair(float(i/n/3), False, a,b,hashes,f'{b}_probe','enrollment','probe'))
        result.append(Pair(.9,True,a,a,hashes,f'{a}_probe','enrollment','probe'))
    return result

def test_empty_insufficient(config):
    report = calibrate_development([], 'model',config)
    assert report['threshold'] is None and report['status'] == 'INSUFFICIENT_DATA'
    assert report['minimum_independent_impostors_for_zero_errors'] == 299

def test_synthetic_never_authorizes(config):
    report = calibrate_development(synthetic_pairs(),'model',config,data_kind='synthetic')
    assert report['status'] == 'SYNTHETIC_TEST_ONLY'
    save_calibration(report,config.path(config.threshold_file))
    assert load_threshold(config.path(config.threshold_file),'model',config) is None

def test_dependent_pairs_do_not_inflate_n(config):
    pairs = synthetic_pairs(n=20)
    many = [replace(p,probe_hash=f'{p.probe_hash}_{i}') for p in pairs for i in range(30)]
    report = calibrate_development(many,'model',config)
    assert report['impostor_pairs'] == 600
    assert report['independent_impostor_pairs'] == 20
    assert report['threshold'] is None

def test_test_labels_cannot_change_threshold(config):
    dev = synthetic_pairs()
    report = calibrate_development(dev,'model',config,data_kind='synthetic')
    threshold = report['threshold']
    test = synthetic_pairs('test')
    evaluate_held_out(dev,test,report)
    altered = [replace(p,score=-.5) for p in test]
    evaluate_held_out(dev,altered,report)
    assert calibrate_development(dev,'model',config,data_kind='synthetic')['threshold'] == threshold

def test_subject_leakage(config):
    dev = synthetic_pairs()
    report = calibrate_development(dev,'model',config,data_kind='synthetic')
    with pytest.raises(ValueError,match='subjects overlap'):
        evaluate_held_out(dev,dev,report)

@pytest.mark.parametrize('change', ['probe', 'session', 'label', 'score'])
def test_invalid_protocol(config,change):
    pair = synthetic_pairs(n=1)[0]
    invalid = {'probe': replace(pair,probe_hash=pair.enrollment_hashes[0]),
        'session': replace(pair,probe_session=pair.enrollment_session),
        'label': replace(pair,genuine=True), 'score': replace(pair,score=float('nan'))}[change]
    with pytest.raises(ValueError):
        calibrate_development([invalid],'model',config)

def test_correct_denominators():
    pairs = synthetic_pairs(n=2)
    pairs[0] = replace(pairs[0],score=.9)
    pairs[1] = replace(pairs[1],score=.1)
    m = confusion_metrics(pairs,.8)
    assert m['fmr'] == .5 and m['fnmr'] == .5
    assert m['false_matches'] == 1 and m['false_nonmatches'] == 1

def test_pure_zero_error_tolerance_bound(config):
    # 299 independent impostors suffice for the configured 1% one-sided 95% bound.
    report = calibrate_development(synthetic_pairs(n=299),'m',config,data_kind='synthetic')
    assert report['independent_impostor_fmr_upper'] <= .01
    assert report['threshold'] > max(p.score for p in synthetic_pairs(n=299) if not p.genuine)

def test_small_final_test_is_insufficient(config):
    dev = synthetic_pairs()
    report = calibrate_development(dev,'model',config,data_kind='synthetic')
    evaluation = evaluate_held_out(dev,synthetic_pairs('tiny_test',n=2),report)
    assert evaluation['status'] == 'INSUFFICIENT_DATA'
    assert evaluation['metrics']['impostor_pairs'] == 2

def test_frozen_development_digest(config):
    dev = synthetic_pairs()
    report = calibrate_development(dev,'model',config,data_kind='synthetic')
    changed = [replace(p,score=p.score-.1) for p in dev]
    with pytest.raises(ValueError,match='frozen calibration'):
        evaluate_held_out(changed,synthetic_pairs('test'),report)

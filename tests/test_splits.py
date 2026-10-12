import pytest
from faceattend.prepare_data import audit_dataset, make_manifests, assert_no_overlap


def test_reproducible_stratified_and_untouched_test(source):
    root, mapping = source
    frame, _ = audit_dataset(root / "raw", root, mapping)
    first = make_manifests(frame, root / "processed")
    second = make_manifests(frame, root / "processed2")
    for name in first:
        assert first[name].path.tolist() == second[name].path.tolist()
    assert len(first["val"]) == 21 and first["val"].class_id.value_counts().eq(3).all()
    assert set(first["test"].path) == set(frame.loc[frame.source_partition == "test", "path"])


def test_hash_leakage_rejected(source):
    root, mapping = source
    frame, _ = audit_dataset(root / "raw", root, mapping)
    train, test = frame[frame.source_partition == "train"].copy(), frame[frame.source_partition == "test"].copy()
    test.loc[test.index[0], "sha256"] = train.iloc[0].sha256
    with pytest.raises(ValueError, match="sha256 overlap"):
        assert_no_overlap({"train": train, "test": test})

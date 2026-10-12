import numpy as np
import pytest
from face_verification.quality import normalize, cosine_similarity
from face_verification.enrollment import aggregate_embeddings

def test_identical_orthogonal_opposite():
    assert cosine_similarity([1,0], [2,0]) == 1
    assert cosine_similarity([1,0], [0,1]) == 0
    assert cosine_similarity([1,0], [-1,0]) == -1

@pytest.mark.parametrize('vector', [[0,0], [float('nan'),1], [float('inf'),0], [], [[1,0]]])
def test_invalid_vector(vector):
    with pytest.raises(ValueError):
        normalize(vector)

def test_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1,0], [1,0,0])

def test_aggregation():
    vectors = [[3,0], [4,1], [4,-1]]
    np.testing.assert_allclose(aggregate_embeddings(vectors), [1,0], atol=1e-6)

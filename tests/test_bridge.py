from cosinefusion import core_backend as core_init
import numpy as np
import sys
import os
# Ensure local src/ is on path so tests import the package in-place
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


def test_cosine_similarity_shape():
    """Ensure the returned similarity matrix has the correct shape."""
    A = np.random.rand(3, 4)
    B = np.random.rand(5, 4)
    res = core_init.cosine_similarity(A, B)
    sim = res["similarity_matrix"]
    meta = res["meta"]

    # Assert matrix shape
    assert sim.shape == (3, 5)
    # Assert metadata matches
    assert meta["n_a"] == 3
    assert meta["n_b"] == 5
    assert meta["dim"] == 4
    assert meta["metric"] == "cosine_similarity"


def test_cosine_similarity_values():
    """Check that identical vectors return a similarity of 1."""
    A = np.array([[1, 2, 3]])
    B = np.array([[1, 2, 3], [3, 2, 1]])
    res = core_init.cosine_similarity(A, B)
    sim = res["similarity_matrix"]

    # The first comparison (same vectors) should be 1.0
    np.testing.assert_almost_equal(sim[0, 0], 1.0, decimal=6)
    # The second should be less than 1.0
    assert sim[0, 1] < 1.0


def test_user_item_recommendations():
    """Validate that top-ranked items match expected feature preferences."""
    items = np.array([
        [0, 1, 0, 0, 1],  # Tea
        [0, 1, 0, 0, 1],  # Coffee
        [1, 0, 1, 1, 0],  # Jaffa Cake
        [1, 0, 1, 0, 0],  # Biscuit
        [1, 0, 1, 1, 1],  # Chocolate Bar
        [0, 1, 0, 0, 1],  # Espresso
    ])
    item_names = ["Tea", "Coffee", "Jaffa Cake",
                  "Biscuit", "Chocolate Bar", "Espresso"]

    # User likes sweet + chocolate
    user = np.array([[0, 0, 0, 1, 0]])

    res = core_init.cosine_similarity(user, items)
    sim = res["similarity_matrix"]

    top_indices = np.argsort(sim[0])[::-1]
    top_item = item_names[top_indices[0]]

    # Expect Jaffa Cake to be top match (matches cosine similarity scoring)
    assert top_item == "Jaffa Cake"
    # The similarity should be positive and within range
    assert 0.0 <= sim[0][top_indices[0]] <= 1.0

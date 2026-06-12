import numpy as np
from cosinefusion import core_backend as core_init


def recommend(user, items, item_names=None, top_k=None):
    """Compute similarity and return ranking info.

    Parameters
    - user: numpy array of shape (1, D)
    - items: numpy array of shape (N, D)
    - item_names: optional list of names for items
    - top_k: optional int to limit returned results

    Returns a dict containing:
    - "similarity_matrix": numpy array of shape (1, N)
    - "top_indices": numpy array of ranked indices (desc)
    - "meta": optional metadata from the backend
    """
    res = core_init.cosine_similarity(user, items)
    sim = res["similarity_matrix"]
    top_indices = np.argsort(sim[0])[::-1]
    if top_k is not None:
        top_indices = top_indices[:top_k]
    return {"similarity_matrix": sim, "top_indices": top_indices, "meta": res.get("meta", {})}


def main():
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

    user = np.array([[0, 0, 0, 1, 0]])

    out = recommend(user, items, item_names)
    sim = out["similarity_matrix"]
    top_indices = out["top_indices"]

    print("Top recommendations for user:")
    for i in top_indices:
        name = item_names[i] if item_names is not None else str(i)
        print(f"{name}: {sim[0][i]:.2f}")


if __name__ == "__main__":
    main()

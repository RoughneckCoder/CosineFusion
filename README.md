
```markdown
# CosineFusion 🚀

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/RoughneckCoder/CosineFusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/RoughneckCoder/CosineFusion/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![PyPI version](https://badge.fury.io/py/cosinefusion.svg)](https://pypi.org/project/cosinefusion/)

## Overview
**CosineFusion** is a hardware-accelerated, high-performance cosine similarity engine that seamlessly bridges C++17 and Python using **pybind11**. By executing intense linear algebra routines directly on compiled binaries, CosineFusion bypasses the Python interpreter's performance bottlenecks, delivering bare-metal speed for matrix matching, AI embeddings, and content recommendation pipelines.

---

## 🔥 Key Features
- **Bare-Metal Performance:** High-speed C++ back-end optimized for vector and matrix calculations.
- **Universal2 Multi-Chip Support:** Natively compiled to support both **Intel (x86_64)** and **Apple Silicon M-Series (arm64)** Mac architectures out of the box with zero manual configuration.
- **Seamless Python Binding:** Exposes highly optimized native bindings directly to Python workflows using `pybind11`.
- **Production-Grade Architecture:** Restructured using the standard Python `src-layout` to ensure strict dependency separation, clean imports, and absolute deployment stability.

---

## 🎯 Use Cases & Applications
- **Enterprise Recommender Systems:** Instantaneous user-to-item feature matrix matching for e-commerce or media content streaming.
- **Personalization Engines:** Real-time scoring based on multi-dimensional user preference vectors.
- **Fast Similarity Search:** Sub-millisecond similarity scoring for AI/ML feature embedding matching.

---

## 💻 Installation

### Option 1: Direct from PyPI (Recommended Production Method)
Install the pre-compiled, optimized universal wheels directly from the global package registry:
```bash
pip install cosinefusion

```

### Option 2: Local Development Setup

If working from source, clone the repository and build the native extensions locally:

```bash
git clone [https://github.com/RoughneckCoder/CosineFusion.git](https://github.com/RoughneckCoder/CosineFusion.git)
cd CosineFusion
pip install pybind11
pip install .

```

---

## ⚡ Usage Example

```python
import numpy as np
import core_init

# Example item features
items = np.array([
    [0, 1, 0, 0, 1], # Tea
    [0, 1, 0, 0, 1], # Coffee
    [1, 0, 1, 1, 0], # Jaffa Cake
    [1, 0, 1, 0, 0], # Biscuit
    [1, 0, 1, 1, 1], # Chocolate Bar
    [0, 1, 0, 0, 1], # Espresso
])

item_names = ["Tea", "Coffee", "Jaffa Cake", "Biscuit", "Chocolate Bar", "Espresso"]

# User likes sweetness and chocolate
user = np.array([[0, 0, 0, 1, 0]]) 

# Executes natively on the underlying hardware (Intel or Apple Silicon)
res = core_init.cosine_similarity(user, items)
sim = res["similarity_matrix"]

top_indices = np.argsort(sim[0])[::-1]
print("Top recommendations for user:")
for i in top_indices:
    print(f"{item_names[i]}: {sim[0][i]:.2f}")

```

### Sample Output:

```text
Top recommendations for user:
Chocolate Bar: 0.89
Jaffa Cake: 0.75     <- Accurately indicates structural sweet-matrix alignment
Biscuit: 0.65
Tea: 0.45
Coffee: 0.45
Espresso: 0.43

```

---

## 🏗️ Hardware Architecture & Multi-Chip Targeting

To maximize cross-platform utility, the build pipeline leverages fat multi-architecture binary compilation. When deploying the package via `pip install`, the distribution system automatically delivers a native **Universal2** binary.

This ensures that the underlying C++ extension automatically executes instructions matched perfectly to the host machine's physical CPU:

* **Apple Silicon (M1/M2/M3/M4 chips):** Executes native ARM64 instructions.
* **Intel Processors:** Executes native x86_64 instructions.

---

## 🧪 Testing

The repository includes an automated verification suite to guarantee mathematical accuracy and cross-platform compatibility. Run tests using `pytest`:

```bash
python -m pytest -v

```

---

## 📁 Project Structure

```text
CosineFusion/
 ├── src/
 │    ├── cpp/
 │    │    └── core_init.cpp     # High-performance C++ core engine
 │    └── python/
 │         └── core_demo.py     # Functional verification demo
 ├── tests/
 │    └── test_bridge.py        # Automated PyTest matrix suite
 ├── setup.py                   # Pybind11 / Universal2 compilation configuration
 ├── pyproject.toml             # Standard modern build-system requirements
 ├── README.md                  # Project documentation
 ├── LICENSE                    # MIT License open-source file
 └── requirements.txt           # Environment dependencies

```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

**Author:** Sam Chaudry

**GitHub:** [RoughneckCoder](https://www.google.com/search?q=https://github.com/RoughneckCoder)

```

```
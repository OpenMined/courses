# Matrix Factorization Example

This example shows how to train a basic matrix factorization model in the manner of federated learning.

[MovieLens data](https://files.grouplens.org/datasets/movielens/ml-100k/) is utilized.


## Setup

```
pyenv install $(cat .python-version)

pip install -r requirements.txt
```

PyTorch1.8 does not work (https://github.com/OpenMined/PySyft/issues/5258), so PyTorch1.7 is used instead.


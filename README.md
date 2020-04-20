# RelEx

A simple framework for Relation Extraction built on AllenNLP.

---

## 🔭&nbsp; Overview

| Path     	               | Description                         	|
|------------------------- |------------------------------	|
| [configs/](configs/)     | This directory contains model configurations for relation classification. |
| [scripts/](scripts/)     | This directory contains scripts, e.g., for evaluating a model with a dataset-specific scorer.|


## ✅&nbsp; Requirements

RelEx is tested with:

- Python 3.7


## 🚀&nbsp; Installation

### With pip

```bash
<TBD>
```

### From source
```bash
git clone https://github.com/DFKI-NLP/RelEx
cd RelEx
pip install .
```

## 🔧&nbsp; Usage

### Training

```bash
allennlp train \
    ./configs/relation_classification/tacred/baseline_cnn_tacred.jsonnet \
    -s <RESULTS DIR> \
    --include-package relex
```


## 👾&nbsp; Models

| Model     	| Link    | Description        |
|----------	| :---------: | :-----------------: |
|           | [[Link]()]  |  |


## 📚&nbsp; Citation

```
@misc{alt-etal-2020-relex,
  author = {Christoph Alt and Marc H\"{u}bner and Leonhard Hennig},
  title = {RelEx},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DFKI-NLP/RelEx}}
}
```
Also, please consider cite the following paper when using RelEx:
```
@inproceedings{alt-etal-2020-probing,
    title={Probing Linguistic Features of Sentence-level Representations in Neural Relation Extraction},
    author={Christoph Alt and Aleksandra Gabryszak and Leonhard Hennig},
    year={2020},
    booktitle={Proceedings of ACL},
    url={https://arxiv.org/abs/}
}
```

## 📘&nbsp; Licence
RelEx is released under the under terms of the [Apache 2.0 License](LICENCE).

# Dynamic-Classification

Code from the paper:
[Metric Learning for Dynamic Text Classification](https://arxiv.org/abs/1911.01026)

## Usage

First install the requirements in `requirements.txt`

- The `distance` folder contains the code for the euclidean and hyperbolic metrics.
- `model.py` file contains code for the RNN encoder and the Prototypical model.
- `sampler.pt` contains the code for creating episodes.

See `train.py` for an example on how to train a model.

## Cite

```sh
@inproceedings{wohlwend-etal-2019-metric,
    title = "Metric Learning for Dynamic Text Classification",
    author = "Wohlwend, Jeremy  and
      Elenberg, Ethan R.  and
      Altschul, Sam  and
      Henry, Shawn  and
      Lei, Tao",
    booktitle = "Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-6116",
    doi = "10.18653/v1/D19-6116",
    pages = "143--152"
}
```

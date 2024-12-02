# Co–TES: Learning noisy labels with a Co-Teaching Exchange Student method

Pattern Recognition Letters: [Co–TES: Learning noisy labels with a Co-Teaching Exchange Student method](https://www.sciencedirect.com/science/article/abs/pii/S0167865524001028)

Some code is adapted from [Co-teaching+](https://github.com/xingruiyu/coteaching_plus/).


## Run
1. edit some arguments in ```main.py```.
2. run ```python main.py```


## Cite
```bash
@article{SHIN202417,
  title = {Co–TES: Learning noisy labels with a Co-Teaching Exchange Student method},
  journal = {Pattern Recognition Letters},
  volume = {182},
  pages = {17-23},
  year = {2024},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2024.04.001},
  url = {https://www.sciencedirect.com/science/article/pii/S0167865524001028},
  author = {Chan Ho Shin and Seong-jun Oh},
  keywords = {Learning with noisy labels, Co-teaching, Multi-network learning},
  abstract = {The performance of a machine-learning model is influenced by two main factors: the structure of the model, and the quality of the dataset it processes. As high-quality labeled data in substantial size is often difficult to obtain, there are ongoing efforts to develop machine learning algorithms that are robust with noisy datasets. Among these algorithms, multi-network learning utilizes learning from a noisy dataset by the selection and filtering of samples through multiple learning networks. We propose an improved co-teaching algorithm termed Co-TES that leverages different models with various architectures. Co-TES extracts different features from each iteration of data selection and makes the model more robust with the same quality dataset. Numerical results show that the proposed method can lead to faster performance gains in the early to mid-range.}
}
```

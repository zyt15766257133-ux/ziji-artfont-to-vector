---
{}
---

<a href="" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-StarVector-red?logo=arxiv" height="20" />
</a>
<a href="https://starvector.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-StarVector-blue.svg" height="20" />
</a>
<a href="https://github.com/joanrod/star-vector" target="_blank" style="display: inline-block; margin-right: 10px;">
    <img alt="GitHub Code" src="https://img.shields.io/badge/Code-StarVector-white?&logo=github&logoColor=white" />
</a>

# Dataset Card for svg-fonts

## Dataset Description

This dataset contains SVG code examples for training and evaluating SVG models for image vectorization.

## Dataset Structure


### Features

The dataset contains the following fields:

| Field Name | Description |
| :--------- | :---------- |
| `Filename` | Unique ID for each SVG |
| `Svg` | SVG code |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("starvector/svg-fonts")
```

## Benchmark Evaluation

To evaluate your model on this dataset, please refer to the [README.md](https://github.com/joanrod/star-vector/blob/main/README.md) file in the [StarVector GitHub repository](https://github.com/joanrod/star-vector).


## Citation

```bibtex
@article{rodriguez2023starvector,
    title={{StarVector: Generating Scalable Vector Graphics Code from Images and Text}},
    author={Juan A. Rodriguez and Abhay Puri and Shubham Agarwal and Issam H. Laradji and Pau Rodriguez and Sai Rajeswar and David Vazquez and Christopher Pal and Marco Pedersoli},
    year={2023},
    journal={arXiv preprint arXiv:2312.11556},
}
```

## Tags

- scalable vector graphics (SVG)
- vision language models
- multimodal
- code

# DE-CLIP: Description-Enhanced CLIP for Zero-Shot Dermatology Image Classification

[Nguyen Duy Dung](mailto:bravend.dev@gmail.com)<sup>1</sup>, Tu Minh Phuong<sup>1</sup>, Hoang Van Thanh<sup>2</sup>

<sup>1</sup> Posts and Telecommunications Institute of Technology, Hanoi, Vietnam
<sup>2</sup> Quang Binh University, Quang Tri, Vietnam

> Manuscript under review at *Biomedical Signal Processing and Control*.

## Overview

DE-CLIP is a training-free method that improves zero-shot dermatology image classification with
CLIP-style vision-language models by improving only the text side. For each disease class, an LLM
generates short clinical descriptions; DE-CLIP scores them by their similarity to the class-name
embedding, groups near-duplicate descriptions, and blends the curated description embedding with
the class-name embedding into one text embedding per class. Inference is unchanged standard CLIP:
no fine-tuning, no extra inference cost, and one fixed configuration for all datasets and
backbones.

Evaluated on ten dermatology datasets and five backbones (DermLIP, MAKE, BiomedCLIP, PMC-CLIP and
OpenAI CLIP), DE-CLIP improves the class-name prompting baseline on every backbone under both the
Derm1M accuracy protocol and a nine-dataset weighted-F1 protocol.

## Code release

The code, configuration files, LLM-generated description pools and result files used in the paper
will be released in this repository upon acceptance of the manuscript. The release will include:

- the DE-CLIP implementation (`declip/`) and the configuration of every experiment (`configs/`);
- the description pools for all tested LLM generators (`descriptions/`);
- scripts that reproduce every table of the paper from the released pools (`scripts/`, `tools/`),
  including the statistical tests of Section 4.7;
- the result files behind the reported numbers (`results/`).

Watch or star the repository to be notified of the release.

## Citation

If you find this work useful, please cite:

```bibtex
@article{nguyen2026declip,
  title   = {{DE-CLIP}: Description-Enhanced {CLIP} for Zero-Shot Dermatology Image Classification},
  author  = {Nguyen, Duy Dung and Tu, Minh Phuong and Hoang, Van Thanh},
  journal = {Biomedical Signal Processing and Control},
  note    = {Under review},
  year    = {2026}
}
```

## Contact

For questions about the paper or the code, please open an issue or contact
Nguyen Duy Dung (bravend.dev@gmail.com).

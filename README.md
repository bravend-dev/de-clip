# DE-CLIP: Description-Enhanced CLIP for Zero-Shot Dermatology Image Classification

[Nguyen Duy Dung](mailto:bravend.dev@gmail.com)<sup>1</sup>, Tu Minh Phuong<sup>1</sup>, Hoang Van Thanh<sup>2</sup>

<sup>1</sup> Posts and Telecommunications Institute of Technology, Hanoi, Vietnam
<sup>2</sup> Quang Binh University, Quang Tri, Vietnam

> Accepted for publication in *Biomedical Signal Processing and Control*.

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

## Installation

```bash
conda create -n declip python=3.10 && conda activate declip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

The backbones are downloaded from the Hugging Face Hub on first use.

## Data

The datasets are not redistributed. Place (or symlink) each image folder at `data/<Dataset>/` and
list its images in `data/meta/<protocol>/<Dataset>.csv` with the columns `image_path` (e.g.
`data/PAD/PAT_999_20_401.png`) and `label` (index into the class list in `declip/classnames.py`):

- `accuracy` (Derm1M test splits, Table 3): `HAM`, `F17K`, `PAD`, `Daffodil`
- `weighted_f1` (all images, Table 4 and the ablations): `PAD`, `HAM`, `Daffodil`, `BCN20000`,
  `DDI`, `HIBA`, `MSKCC`, `Dermnet`, `Derm7pt`

## Usage

```bash
# DE-CLIP and the Default / CuPL baselines on one backbone
python -m declip.run --config configs/methods.yaml --backbone dermlip --out results/demo.csv

# every experiment of the paper, then the tables
bash scripts/run_all.sh
python tools/make_tables.py      # Tables 3-10
python tools/stat_tests.py       # Table 11
```

Each experiment has its own script in `scripts/` and configuration in `configs/`. The
LLM-generated description pools used in the paper are in `descriptions/`, so no LLM call is needed;
`tools/generate_descriptions.py` documents the generation prompt.

## Citation

If you find this work useful, please cite:

```bibtex
@article{nguyen2026declip,
  title   = {{DE-CLIP}: Description-Enhanced {CLIP} for Zero-Shot Dermatology Image Classification},
  author  = {Nguyen, Duy Dung and Tu, Minh Phuong and Hoang, Van Thanh},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026}
}
```

## License

The code, configuration files and description pools are released under the [MIT License](LICENSE).

## Contact

For questions about the paper or the code, please open an issue or contact
Nguyen Duy Dung (bravend.dev@gmail.com).

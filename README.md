# EndoCLIP

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenCLIP](https://img.shields.io/badge/built%20on-OpenCLIP%20v3.3.0-8a2be2.svg)](https://github.com/mlfoundations/open_clip)

**EndoCLIP** is a contrastive vision–language framework for **gastrointestinal endoscopy**.
It adapts open-vocabulary image–text pretraining to endoscopic video frames and their
paired clinical reports, so that a single model can be used for zero-shot lesion
recognition, frame–report retrieval, and transferable endoscopic representation learning.

This repository contains the **full training and evaluation codebase** (vendored from
[OpenCLIP](https://github.com/mlfoundations/open_clip) v3.3.0, see
[Acknowledgements](#acknowledgements)) together with the release information for the
**EndoReport100** dataset.

---

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data: EndoReport100](#data-endoreport100)
- [Usage](#usage)
  - [Loading a model](#loading-a-model)
  - [Training](#training)
  - [Zero-shot evaluation](#zero-shot-evaluation)
- [Upstream OpenCLIP Documentation](#upstream-openclip-documentation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Contrastive image–text pretraining** over endoscopic frames and de-identified report text.
- **Wide model coverage** — ResNet, ViT, ConvNeXt, CoCa, SigLIP and timm-backed towers, all
  configurable through JSON model configs in [`src/open_clip/model_configs/`](src/open_clip/model_configs).
- **Distributed training** with PyTorch DDP / FSDP, mixed precision (`amp`, `amp_bf16`),
  gradient checkpointing, and gradient accumulation.
- **Webdataset and CSV pipelines**, so both sharded video-frame archives and simple
  `image_path,caption` tables can be used without code changes.
- **Zero-shot classification and retrieval evaluation** built into the training loop.

---

## Repository Layout

```text
EndoCLIP/
├── src/
│   ├── open_clip/            # model definitions, tokenizer, losses, pretrained registry
│   │   └── model_configs/    # JSON architecture configs
│   └── open_clip_train/      # training entry point, data pipelines, schedulers, zero-shot eval
├── scripts/                  # helper launch scripts
├── docs/                     # upstream documentation, pretrained model tables, notebooks
├── tests/                    # unit / regression tests
├── tutorials/                # notebook tutorials
├── requirements.txt          # inference dependencies
├── requirements-training.txt # additional training dependencies
├── requirements-test.txt     # test dependencies
├── OPENCLIP_README.md        # unmodified upstream OpenCLIP README
├── CITATION_openclip.cff     # unmodified upstream OpenCLIP citation metadata
└── LICENSE                   # MIT license inherited from OpenCLIP
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Jia7878/EndoCLIP.git
cd EndoCLIP

# 2. Create an environment
conda create -n endoclip python=3.10 -y
conda activate endoclip

# 3. Install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch torchvision

# 4. Install EndoCLIP in editable mode
pip install -e .

# 5. Extra dependencies for training
pip install -r requirements-training.txt
```

Verify the installation:

```bash
python -c "import open_clip; print(open_clip.__version__)"
```

---

## Data: EndoReport100

The de-identified **EndoReport100** release contains **100 cases** and **7,003 frames**,
together with frame-level annotations and instance metadata.

> **Download:** [https://doi.org/10.6084/m9.figshare.33085637](https://doi.org/10.6084/m9.figshare.33085637)

Access is restricted and subject to approval by the dataset owners. All frames and reports
are de-identified; any attempt at re-identification is prohibited.

### Expected format

The training scripts consume either a CSV table or webdataset shards.

**CSV** — one row per frame:

```csv
filepath,caption
frames/case001/000123.jpg,"Sessile polyp in the sigmoid colon, approximately 8 mm."
frames/case001/000124.jpg,"Normal mucosa with clear vascular pattern."
```

**Webdataset** — `.tar` shards where each sample pairs an image (`.jpg`) with its text (`.txt`)
under a shared key.

---

## Usage

### Loading a model

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="/path/to/endoclip_checkpoint.pt"
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-16")

image = preprocess(Image.open("frames/case001/000123.jpg")).unsqueeze(0)
text = tokenizer(["a colonoscopy frame of a polyp",
                  "a colonoscopy frame of normal mucosa"])

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("label probs:", probs)
```

### Training

Single node, 4 GPUs, CSV input:

```bash
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --model ViT-B-16 \
    --pretrained openai \
    --train-data /path/to/endoreport100/train.csv \
    --val-data /path/to/endoreport100/val.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --dataset-type csv \
    --batch-size 128 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 30 \
    --warmup 500 \
    --workers 8 \
    --precision amp \
    --logs ./logs \
    --name endoclip-vitb16 \
    --report-to tensorboard
```

Webdataset shards:

```bash
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --model ViT-B-16 \
    --dataset-type webdataset \
    --train-data "/path/to/endoreport100/shards/train-{000000..000031}.tar" \
    --train-num-samples 7003 \
    --batch-size 128 --epochs 30 --precision amp \
    --logs ./logs --name endoclip-vitb16-wds
```

All available flags are documented in
[`src/open_clip_train/params.py`](src/open_clip_train/params.py).

### Zero-shot evaluation

```bash
python -m open_clip_train.main \
    --model ViT-B-16 \
    --resume ./logs/endoclip-vitb16/checkpoints/epoch_30.pt \
    --imagenet-val /path/to/endoreport100/zeroshot_val \
    --batch-size 128 \
    --precision amp
```

For broader benchmarking we recommend the
[CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) suite, which consumes
OpenCLIP-format checkpoints directly.

---

## Upstream OpenCLIP Documentation

Because this repository vendors the OpenCLIP source tree, the upstream documentation
applies verbatim:

| Topic | Document |
| --- | --- |
| Full upstream README | [`OPENCLIP_README.md`](OPENCLIP_README.md) |
| Pretrained model zoo | [`docs/PRETRAINED.md`](docs/PRETRAINED.md) |
| DataComp models | [`docs/datacomp_models.md`](docs/datacomp_models.md) |
| CLIPA training | [`docs/clipa.md`](docs/clipa.md) |
| Conceptual Captions walkthrough | [`docs/clip_conceptual_captions.md`](docs/clip_conceptual_captions.md) |
| Interactive notebook | [`docs/Interacting_with_open_clip.ipynb`](docs/Interacting_with_open_clip.ipynb) |
| Release history | [`HISTORY.md`](HISTORY.md) |

---

## Citation

If EndoCLIP or EndoReport100 is useful for your research, please cite this repository:

```bibtex
@misc{endoclip,
  title        = {EndoCLIP: Endoscopy Vision-Language Modeling},
  author       = {{FDU-MICCAI}},
  year         = {2026},
  howpublished = {\url{https://github.com/Jia7878/EndoCLIP}}
}
```

Please **also** cite OpenCLIP, whose implementation this work is built on. The upstream
citation metadata is preserved in [`CITATION_openclip.cff`](CITATION_openclip.cff):

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author    = {Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and
               Gordon, Cade and Carlini, Nicholas and Taori, Rohan and
               Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and
               Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and
               Schmidt, Ludwig},
  title     = {OpenCLIP},
  month     = jul,
  year      = 2021,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.5143773},
  url       = {https://doi.org/10.5281/zenodo.5143773}
}
```

---

## License

The code in this repository is distributed under the **MIT license** inherited from
OpenCLIP — see [`LICENSE`](LICENSE). The original copyright notice of the OpenCLIP authors
is retained unmodified.

The **EndoReport100** dataset is *not* covered by that license. It is released separately
under restricted access and remains subject to the terms stated on its
[FigShare record](https://doi.org/10.6084/m9.figshare.33085637).

---

## Acknowledgements

**This project is built on [OpenCLIP](https://github.com/mlfoundations/open_clip).**

The entire `src/`, `docs/`, `scripts/`, `tests/` and `tutorials/` trees in this repository are
copied from OpenCLIP release **v3.3.0**
([`mlfoundations/open_clip@v3.3.0`](https://github.com/mlfoundations/open_clip/tree/v3.3.0)),
with EndoCLIP-specific configurations and endoscopy adaptations layered on top. The upstream
README and citation file are preserved unmodified as [`OPENCLIP_README.md`](OPENCLIP_README.md)
and [`CITATION_openclip.cff`](CITATION_openclip.cff), and the upstream MIT
[`LICENSE`](LICENSE) is retained in full.

We thank the OpenCLIP authors and the wider LAION / mlfoundations community for releasing
their work openly. Please follow the upstream
[OpenCLIP citation guidance](https://github.com/mlfoundations/open_clip#citing) when using
this code.

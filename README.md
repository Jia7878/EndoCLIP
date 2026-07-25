# EndoCLIP

EndoCLIP is developed on top of [OpenCLIP](https://github.com/mlfoundations/open_clip). The upstream OpenCLIP source is included in [`open_clip/`](open_clip/) as a Git submodule and pinned to the `v3.3.0` release for reproducibility.

## Clone

Clone this repository together with OpenCLIP:

```bash
git clone --recurse-submodules https://github.com/Jia7878/EndoCLIP.git
```

If the repository was cloned without submodules, initialize OpenCLIP with:

```bash
git submodule update --init --recursive
```

## EndoReport100

The de-identified EndoReport100 release contains 100 cases and 7,003 frames, together with frame-level annotations and instance metadata.

[Request access to EndoReport100 on FigShare](https://doi.org/10.6084/m9.figshare.33085637)

## Acknowledgment

This project builds upon OpenCLIP. Please follow the upstream [OpenCLIP citation guidance](https://github.com/mlfoundations/open_clip#citing) when using this code. OpenCLIP is distributed under its own [MIT license](open_clip/LICENSE).

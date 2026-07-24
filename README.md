# EndoCLIP

EndoCLIP is developed on top of [OpenCLIP](https://github.com/mlfoundations/open_clip). The upstream OpenCLIP source is included in [`open_clip/`](open_clip/) as a Git submodule and pinned to the `v3.3.0` release for reproducibility.

## Clone

Clone this repository together with OpenCLIP:

```bash
git clone --recurse-submodules https://github.com/FDU-MICCAI/EndoCLIP.git
```

If the repository was cloned without submodules, initialize OpenCLIP with:

```bash
git submodule update --init --recursive
```

## EndoReport100

The de-identified EndoReport100 release contains 100 cases and 7,003 frames, together with frame-level annotations and instance metadata.

[Request access to EndoReport100 on Google Drive](https://drive.google.com/file/d/1Q8RhuoajghvtNZ7-xvWKnnGOix4ZI65k/view?usp=sharing)

Access is restricted and subject to approval by the dataset owners. The SHA-256 checksum of the current `EndoReport100.tar` release is:

```text
dcc173d139c6b3f1478160f79a52ce72488930abd971097f829fdb985628b529
```

## Acknowledgment

This project builds upon OpenCLIP. Please follow the upstream [OpenCLIP citation guidance](https://github.com/mlfoundations/open_clip#citing) when using this code. OpenCLIP is distributed under its own [MIT license](open_clip/LICENSE).

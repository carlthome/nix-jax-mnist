# nix+JAX=❤️
**A toy example of training MNIST with JAX in a nix shell.**

I'm dreaming of scientifically reproducible model development between multiple past/present/future collaborators in long-running R&D projects. Historically, I've been bit by past colleagues asking why they're getting `nan` when running my old training code on a new system, and learning that simply pinning Python packages in a requirements.txt with `pip freeze` is not enough because ML dependencies usually dynamically link to system-wide software such as CUDA/CuDNN, and different versions of those can result in different ML models.

## Build
> **Note:** Every instruction that follows assumes `nix` is installed on your system. If it isn't, go and do that first at https://nixos.org/download.html

Install required Python dependencies into a nix package by running
```sh
nix-build
```

## Run
After `nix-build`, the build artifacts are available in result/ and can be started as:
```sh
./result/bin/train
```

## Develop
Start a development shell with
```sh
nix-shell
```
within which `python` has the packages specified in [shell.nix](./shell.nix) as well as a virtual environment for `pip install`:ing additional dependencies when needed.

Thus you can simply run `python train.py` as per usual.

{ pkgs ? import <nixpkgs> { } }:
with pkgs;
let
  python = python3.withPackages (p: with p; [
    pip
    ipython
    black
    mypy
    tqdm
    jax
    jaxlib
    tensorflow
    tensorflow-datasets
  ]);
in
mkShell {
  buildInputs = [
    python
  ];

  shellHook = ''
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${python.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
  '';
}

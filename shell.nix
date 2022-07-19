{ pkgs ? import <nixpkgs> { } }:
with pkgs;
let
  python = python3.withPackages (p: with p; [
    ipython
    jax
    jaxlib
    black
    tensorflow
    tensorflow-datasets
  ]);
in
mkShell {
  buildInputs = [
    python
  ];
}

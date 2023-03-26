{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (p: with p; [
      pip
      ipython
      black
      mypy
      tqdm
      jax
      jaxlib
      tensorflow
      tensorflow-datasets
    ]))
  ];
}

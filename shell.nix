{ pkgs ? import <nixpkgs> { } }:
with pkgs;
let
  python = python3.withPackages (p: with p; [
    pip
    ipython
    black
    mypy
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
    PYTHONPATH="${python.sitePackages}:$PYTHONPATH"
    VIRTUAL_ENV=.venv

    ${python}/bin/python -m venv $VIRTUAL_ENV
    source $VIRTUAL_ENV/bin/activate
  '';
}

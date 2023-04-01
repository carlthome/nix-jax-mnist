{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (p: with p; [
      pip
      ipython
      black
      mypy
    ] ++ [
      tqdm
      jax
      jaxlib
      tensorflow
      tensorflow-datasets
    ]))
  ];
  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export PIP_PREFIX=$(pwd)/.pip
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}

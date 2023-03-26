{ pkgs ? import <nixpkgs> { } }:
with pkgs;

stdenv.mkDerivation {
  name = "train";

  src = ./.;

  buildInputs = [
    (python3.withPackages (ps: with ps; [
      jax
      jaxlib
      tensorflow
      tensorflow-datasets
    ]))
  ];

  buildPhase = ''
    echo Building!
  '';

  installPhase = ''
    mkdir -p $out/bin
    cp train.py $out/bin/train
    chmod +x $out/bin/train
  '';

  checkInputs = [
    pytest
  ];

  checkPhase = ''
    pytest
  '';
}

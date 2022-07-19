{ pkgs ? import <nixpkgs> { } }:
with pkgs;
stdenv.mkDerivation {
  inherit (packages.zucker) version src;
  pname = "zucker-docs";

  buildInputs = [
    (python39.withPackages (ps: with ps; [
      sphinx
      sphinx_rtd_theme
    ]))
  ];
  buildPhase = ''
    echo building
  '';

  installPhase = ''
    echo installing > $out
  '';
};

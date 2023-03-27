{
  description = "A toy example of training a ML model with JAX";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
  };
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      packages.${system}.default = import ./default.nix { inherit pkgs; };
      devShells.${system}.default = import ./shell.nix { inherit pkgs; };
    };
}

{
  description = "Toy example of using JAX with nix flakes.";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system}; in
      rec {
        checks.pytest = pkgs.python3Packages.pytest;

        packages = {
          train = import ./default.nix { inherit pkgs; };
        };

        defaultPackage = packages.train;

        apps.train = flake-utils.lib.mkApp { drv = self.defaultPackage.${system}; };

        devShells.default = import ./shell.nix { inherit pkgs; };

      }
    );
}

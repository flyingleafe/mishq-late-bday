{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    bun2nix.url = "github:nix-community/bun2nix";
    bun2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  outputs = { self, nixpkgs, bun2nix }: {
    packages.x86_64-linux.default = bun2nix.mkDerivation {
      pname = "mishq-late-bday";
      version = "1.0.0";
      src = ./.;
      bunNix = ./bun.nix;
      module = "index.ts";
    };

    devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      packages = with nixpkgs.legacyPackages.x86_64-linux; [
        bun
        bun2nix.packages.x86_64-linux.default
      ];
    };
  };
}   

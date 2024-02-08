{
  description = "A build environment for this project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/22.11";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, ... }@inputs: inputs.utils.lib.eachSystem [
    "x86_64-linux"
  ] (system: let
    pkgs = import nixpkgs {

      inherit system;

      overlays = [];

    };
  in {
    devShells.default = pkgs.mkShell rec {
      packages = with pkgs; [
        nodePackages.terser
        nodePackages.svgo
        nodePackages.clean-css-cli
      ];

      shellHook = let
        icon = "f121";
      in ''
        export PS1="$(echo -e '\u${icon}') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} \\$ \[$(tput sgr0)\]"
      '';
    };

  });
}

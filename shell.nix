{
    system ? builtins.currentSystem,
    nixpkgs ? fetchTarball "https://github.com/NixOS/nixpkgs/archive/057f9aecfb71c4437d2b27d3323df7f93c010b7e.tar.gz"
}:
let
    # The haskell nixpkgs requires the build of node from scratch :)
    pkgs = import nixpkgs {
        inherit system;
        config = {};
        overlays = [];
    };

    sources = import ./nix/sources.nix;
    haskellpkgs = import sources.nixpkgs {
        inherit system;
        config = {};
        overlays = [];
    };

    stack-wrapped = haskellpkgs.symlinkJoin {
        name = "stack";
        paths = [ haskellpkgs.stack ];
        buildInputs = [ haskellpkgs.makeWrapper ];
        postBuild = ''
        wrapProgram $out/bin/stack \
            --add-flags "\
            --nix \
            --no-nix-pure \
            --nix-shell-file=nix/stack-integration.nix \
            "
        '';
    };

in
haskellpkgs.mkShell {
    buildInputs =  [
        stack-wrapped
        haskellpkgs.niv
        pkgs.nodePackages.terser
        pkgs.nodePackages.svgo
        pkgs.nodePackages.clean-css-cli
    ];
    NIX_PATH = "nixpkgs=" + pkgs.path;
}

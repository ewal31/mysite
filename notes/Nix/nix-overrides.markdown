---
title: Override vs OverrideAttrs
---

Given a simple derivation, such as

```nix
{
    stdenv,
    bar,
    baz
}:
stdenv.mkDerivation {
  pname = "test";
  version = "0.0.1";
  buildInputs = [bar baz];
  phases = ["installPhase"];
  installPhase = "touch $out";
}
```

We can overide the arguments/inputs via `override`. For example,

```nix
example.override {
    baz = customBaz;
}
```

would use a different derivation in place of `baz`.

If, however, we want to change one of the build phases, or the version of the
resulting derivation, we can run

```nix
example.overrideAttrs (_: {
    version = "0.0.2";
    installPhase = ''
        echo "Some Information" > $out
    ''
})
```

For many programming lanugages there are also special wrappers and possibilites
for overriding derivations. Documentation for specific languages is available
[here](https://github.com/NixOS/nixpkgs/tree/master/doc/languages-frameworks).

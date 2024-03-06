---
title: Hashing Files for the Nix Store
---

To manually add a file to the store, use the following command

```bash
nix-store --add-fixed sha256 <file-name>.tar.gz
```

The hash can be calculated using

```bash
nix-hash --type sha256 --flat <file-name>.tar.gz
```

Files retrieved by one of the nix builtin functions, typically don't add the
`--flat` option, which in this case would hash the compressed file, but instead
hash the decompressed files.

We could then make use of this artifact, for example, as the `src` for another
derivation.

```nix
stdenv.mkDerivation rec {
  name = "test";
  version = "0.0.1";
  pname = "${name}-${version}";
  src = prev.requireFile {
    name = "<file-name>.tar.gz";
    url = "<url-to-retrieve-file-if-not-available-in-store>";
    sha256 = "<hash-from-nix-hash-command>";
  };
}
```

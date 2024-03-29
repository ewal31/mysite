---
title: Integration Test Template
---

This template spins up a virtual machine containing a Python webserver with a
simple API. It then spins up a second virtual machine that utilises the API and
checks that it returns the expected string `"Nix is cool!"`. To run, we create a
file `default.nix` and `main.py` and copy the Nix and Python code from below
into the respective file and run `nix-build`.

```nix
# default.nix
{
  system ? builtins.currentSystem,
  nixpkgs ? (fetchTarball "https://github.com/NixOS/nixpkgs/archive/057f9aecfb71c4437d2b27d3323df7f93c010b7e.tar.gz"),
}:

let

pkgs = import nixpkgs {
  inherit system;
  config = {};
  overlays = [];
};

isNix = path : let
  file = baseNameOf path;
  suffixMatch = builtins.match "^.*(\\.nix)$" file;
in
  suffixMatch != null;

pathsToFilter = map builtins.toString [
  ./result # a previous nix-build derivation result
];

toFilter = path : type : !(
  isNix path ||
  builtins.any (x : x == builtins.toString path) pathsToFilter
);

# Pack the source code into a derivation.
app-source = pkgs.stdenv.mkDerivation rec {
  name = "api-code";

  src = builtins.path {
    name = "${name}";
    # He we take all source files in the currenty directory
    # that aren't listed in pathsToFiler or end in .nix.
    path = builtins.filterSource toFilter ./.;
  };

  phases = [ "installPhase" ];
  installPhase = "mkdir -p $out && cp -rT $src $out";
};

webserverPort = 4000;

# Build a runnable derivation
app = pkgs.writeShellApplication {
  name = "api";
  runtimeInputs = [ python app-source ];

  # Shell Applications are checked by shellcheck at build time
  text = ''
    export UVICORN_HOST=0.0.0.0
    export UVICORN_PORT=${builtins.toString webserverPort}
    uvicorn main:app
  '';
};

# The python packages and interpreter required to run
# our application
python = pkgs.python311.withPackages (ps: [
  ps.fastapi
  ps.uvicorn
]);

in

pkgs.testers.runNixOSTest {
  name = "api-test";

  # Define Virtual Machines (names such as server and client
  # are arbitrary)
  # In this case we are starting two separate virtual machines
  nodes = {

    # 1. Our simple Python API
    server = {

      # The server is started as a systemd service
      systemd.services.app = {
        wantedBy = [ "multi-user.target" ];
        serviceConfig = {
          # Specify which app to start
          ExecStart = "${app}/bin/api";

          # In the case of a Python application we
          # also specify the working directory as the
          # location of the source code
          WorkingDirectory = "${app-source}";
        };
      };

      # The server is started as a systemd service
      networking = {
        firewall = {
          # specify ports that should be opened
          allowedTCPPorts = [ webserverPort ];
        };
      };

      environment.systemPackages = [ ];
    };

    # 2. The client testing the API
    client = {
      # we use curl to check the api response
      environment.systemPackages = [ pkgs.curl ];
    };
  };

  # Wait for the VMs to start (default.target)
  # and the server to start (app.service)
  # the make use of the API and check the expected
  # string is received
  testScript = ''
    server.wait_for_unit("default.target")
    server.wait_for_unit("app.service")
    client.wait_for_unit("default.target")
    client.succeed(
        "curl http://server:${builtins.toString webserverPort}/" + \
        "| grep -o \"Nix is cool!\""
    )
    print("Test was successful")
  '';
}
```

Our simple API just returns "Nix is cool!" at its root path.

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Nix is cool!"}
```

We use `nix-build` to start building the derivation, which runs the test. We
see that it starts a VM `server` and `client`, the two VMs defined in the test
above. At the end, it uses `curl` to test the API endpoint. If we then rerun
the test, we should see that it has cached the result, and finishes instantly.
Try breaking the test by changing the text that the Python application returns.

```bash
$> nix-build
these 13 derivations will be built:
  /nix/store/q4c7ayi5vaalvh90ig9i7y6l3a8jkpqj-api.drv
  /nix/store/9r04vm7w0kjdqnh7445gx0i6p4xjk561-unit-app.service.drv
  /nix/store/bdaa7r9q98z3lzkxyz94z4v2g7jzmsjf-firewall-start.drv
...
...
...
...
server: waiting for unit default.target
server: waiting for the VM to finish booting
server: starting vm
server: QEMU running (pid 7)
...
...
...
...
(finished: waiting for unit default.target, in 43.43 seconds)
server: waiting for unit app.service
(finished: waiting for unit app.service, in 0.64 seconds)
client: waiting for unit default.target
client: waiting for the VM to finish booting
client: starting vm
client: QEMU running (pid 37)
...
...
...
...
client: Guest shell says: b'Spawning backdoor root shell...\n'
client: connected to guest root shell
client: (connecting took 36.04 seconds)
(finished: waiting for the VM to finish booting, in 36.18 seconds)
...
...
...
...
(finished: waiting for unit default.target, in 44.43 seconds)
client: must succeed: curl http://server:4000/ | grep -o "Nix is cool!"
client # [   43.938431] AVX2 version of gcm_enc/dec engaged.
client # [   43.941429] AES CTR mode by8 optimization enabled
client #   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
client #                                  Dload  Upload   Total   Spent    Left  Speed
server # [   88.207195] api[582]: INFO:     192.168.1.1:47708 - "GET / HTTP/1.1" 200 OK
client #   0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100    26  100    26    0     0    193      0 --:--:-- --:--:-- --:--:--   213
(finished: must succeed: curl http://server:4000/ | grep -o "Nix is cool!", in 0.71 seconds)
Test was successful
(finished: run the VM test script, in 89.23 seconds)
...
...
...
...
test script finished in 89.69s
cleanup
kill machine (pid 37)
client # qemu-kvm: terminating on signal 15 from pid 4 (/nix/store/qp5zys77biz7imbk6yy85q5pdv7qk84j-python3-3.11.6/bin/python3.11)
kill machine (pid 7)
server # qemu-kvm: terminating on signal 15 from pid 4 (/nix/store/qp5zys77biz7imbk6yy85q5pdv7qk84j-python3-3.11.6/bin/python3.11)
(finished: cleanup, in 0.03 seconds)
```

## Debugging

It is also possible to start an interactive environment for debugging. This can
be done by running

```bash
$(nix-build -A driverInteractive default.nix)/bin/nixos-test-driver
```

This doesn't automatically start the virtual machines. Start all of them with

```bash
start_all()
```

or start individual machines with

```bash
<virtual-machine-name>.start()
```

In this case we coudld run

```bash
server.start()
client.start()
```

to start the individual machines.

We could then start an interactive session on one of the machines via

```bash
<virtual-machine-name>.shell_interact()
```

We can also run the steps from the test script one after another and inspect
the machines.

The entire test script can be run via

```bash
test_script()
```

## More Information

* [nix.dev](https://nix.dev/tutorials/nixos/integration-testing-using-virtual-machines)
* [official documentation](https://nixos.org/manual/nixos/stable/index.html#sec-call-nixos-test-in-nixos)

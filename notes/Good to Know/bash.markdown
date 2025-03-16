# Preferred Bash Shebang

```bash
#!/usr/bin/env bash
```

`env` runs a program in a modified environment, the command in this case being
the first copy of `bash` found within `${PATH}`.

[source](https://en.wikipedia.org/wiki/Shebang_(Unix)#Portability)

# Location of Bash Script

```bash
SCRIPT_DIR="$( cd -P -- "$( dirname -- "$(readlink -f "${BASH_SOURCE[0]}")" )" >/dev/null 2>&1 && pwd )"
```

[source](https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script?page=1&tab=scoredesc#tab-top)

# Bash Strict Mode

```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
```

* `-e`: immediately exit if a command has a non-zero exit status
* `-u`: references to an undefined variable cause the script to exit
* `-o pipefail`: returns non-zero exit statuses from any command in a set of
  piped commands instead of the exit code of the final command
* `IFS`: each character is a delimiter used by bash for separating arguments,
  so we avoid arguments with spaces, for example, being incorrectly parsed in
  bash loops

[source](http://redsymbol.net/articles/unofficial-bash-strict-mode/)

# Bash Exit Traps for Cleanup

At the start of your file set up a trap with a cleanup function that should be
run.

```bash
#!/usr/bin/env bash

function finish {
  # cleanup
}
trap finish EXIT
```

[source](http://redsymbol.net/articles/bash-exit-traps/)

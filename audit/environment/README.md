# Verification environment accommodation

The official Lean 4.19.0 Linux binary queries its own executable through
`/proc/<getpid()>/exe`. This container exposes that same executable through
`/proc/self/exe`, while its numbered process paths use a different PID view.
The included `self_exe_compat.c` translates only the calling process's own
executable lookup to `/proc/self/exe`. All other `readlink` calls are delegated
unchanged. No Lean source, kernel, proof term or Mathlib source was modified
for this accommodation.

The local verification process built the shim with:

```sh
gcc -shared -fPIC self_exe_compat.c -o self_exe_compat.so -ldl
```

It set `LD_PRELOAD` to that local shared object and put the official Lean
4.19.0 `bin` directory on `PATH`. Normal Lean installations can use the
commands in `lean/COMPILE.md` directly and do not require this shim.
The official release and cache archives were extracted without restoring
archive ownership (`tar --no-same-owner`).

The toolchain was downloaded from:
https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst

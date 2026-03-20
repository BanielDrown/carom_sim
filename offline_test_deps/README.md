# Offline test dependencies

This directory contains repo-vendored fallbacks that are used only when a
required third-party dependency is unavailable in the active Python
environment.

## Current fallback

- `numpy/`: a minimal compatibility shim that covers the subset of NumPy APIs
  exercised by this repository's runtime logic and unit tests.

## Activation

`src/sitecustomize.py` prepends this directory to `sys.path` only when the real
package cannot be imported. In a normal environment with real NumPy installed,
the real package is used instead.

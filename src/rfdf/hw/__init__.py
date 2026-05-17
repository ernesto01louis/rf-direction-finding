"""Hardware-support utilities for rfdf.

Pure-Python helpers that surround the hardware backends — udev-rule generation
and installation, and the extended ``rfdf hw selftest`` runner. Nothing here
imports a vendor SDK; the modules are safe to import on a base install.
"""

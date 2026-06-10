"""Standalone inference sidecars — single-file HTTP services shipped as
their own container images (see deploy/<name>/Dockerfile). Each module
must stay free of cogniverse imports so the image can COPY it alone."""

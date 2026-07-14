from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from urllib.request import urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_conf_url = "https://tiker.net/sphinxconfig-v0.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
]

project = "namedisl"
copyright = "2025- University of Illinois Board of Trustees"
author = "Andreas Kloeckner"

release = metadata.version("namedisl")
version = ".".join(release.split(".")[:2])

intersphinx_mapping = {
    "islpy": ("https://documen.tician.de/islpy", None),
    "constantdict": ("https://matthiasdiener.github.io/constantdict/", None),
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore_regex = [
    ["py:class", r"IslObjectT.*"],
    ["py:class", r"namedisl\.core\.IslObjectT.*"],
    ["py:class", r"namedisl\.core\.NamedIslObjectT.*"],
]

sphinxconfig_missing_reference_aliases = {
    "dim_type": "class:islpy.dim_type",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821

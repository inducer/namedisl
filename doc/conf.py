from __future__ import annotations

from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2025- University of Illiois Board of Trustees"
author = "Andreas Kloeckner"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
ver_dic = {}
with open("../namedisl/__init__.py") as vfile:
    exec(compile(vfile.read(), "../namedisl/__init__.py", "exec"), ver_dic)

version = ".".join(str(x) for x in ver_dic["__version__"])
release = ver_dic["__version__"]

intersphinx_mapping = {
    "islpy": ("https://documen.tician.de/islpy", None),
    "constantdict": ("https://matthiasdiener.github.io/constantdict/", None),
}

nitpicky = True

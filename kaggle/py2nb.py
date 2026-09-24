"""Convert a percent-format script (cells split by '# %%', markdown by '# %% [markdown]') to .ipynb.

Usage: python kaggle/py2nb.py kaggle/prep/prep.py   -> writes kaggle/prep/prep.ipynb
"""
import json, re, sys

src_path = sys.argv[1]
text = open(src_path).read()
cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in header:
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(src_path[:-3] + ".ipynb", "w"), indent=1)

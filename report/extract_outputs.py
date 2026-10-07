"""Extract every image and text/table output from the executed notebook, for writing the report.

Images go to report/outputs/cell<NN>_<k>.<ext>; all text and table outputs to report/outputs/outputs.md,
in cell order, so every number quoted in the report can be traced to a notebook output.

Usage:
    .venv/bin/python report/extract_outputs.py notebooks/uvh26_signal_counting.ipynb
"""

import base64
import html
import re
import sys
from pathlib import Path

import nbformat

nb = nbformat.read(sys.argv[1], 4)
out = Path(__file__).parent / "outputs"
out.mkdir(exist_ok=True)
md = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        first = cell.source.splitlines()[0] if cell.source else ""
        if first.startswith("#"):
            md.append(f"\n{first}\n")
        continue
    md.append(f"\n### cell {i}: `{cell.source.splitlines()[0][:90]}`\n")
    for k, o in enumerate(cell.get("outputs", [])):
        if o.output_type == "stream":
            text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", o.text)
            lines = [ln for ln in text.splitlines() if "it/s" not in ln and "━" not in ln]
            md.append("```\n" + "\n".join(lines)[-6000:] + "\n```")
        elif o.output_type in ("display_data", "execute_result"):
            d = o.get("data", {})
            for mime, ext in (("image/png", "png"), ("image/jpeg", "jpg")):
                if mime in d:
                    p = out / f"cell{i:02d}_{k}.{ext}"
                    p.write_bytes(base64.b64decode(d[mime]))
                    md.append(f"![{p.name}]({p.name})")
                    break
            else:
                if "text/plain" in d:
                    md.append("```\n" + d["text/plain"] + "\n```")
        elif o.output_type == "error":
            md.append(f"**ERROR** {o.ename}: {o.evalue}")
(out / "outputs.md").write_text("\n".join(md))
print(f"{len(list(out.glob('cell*')))} images, outputs.md {len(''.join(md)) // 1000} kB")

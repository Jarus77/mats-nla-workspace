"""Inline results/viewer_data.json into viewer/template.html -> results/readout_viewer.html (self-contained)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = (HERE.parent / "results" / "viewer_data.json").read_text()
json.loads(data)  # validate
safe = data.replace("<", "\\u003c")  # keeps </script> and <!-- out of the script block
safe = safe.replace("\ufffd", "\\ufffd")  # partial-byte tokens: keep the JSON escape, not a literal U+FFFD
html = (HERE / "template.html").read_text().replace("__DATA__", safe)
out = HERE.parent / "results" / "readout_viewer.html"
out.write_text(html)
print(out, f"{len(html.encode()) / 1e6:.2f} MB")

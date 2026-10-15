"""Static index.html for browsing a directory of generated plots."""
import html
import re
from pathlib import Path

PHASES = ("Excitation", "Free_Vibration", "Full_Simulation")
_NODE_RE = re.compile(r"^run_\d+_node_(\d+)_(" + "|".join(PHASES) + r")_(.+)$")
_PHASE_RE = re.compile(r"^run_\d+_(" + "|".join(PHASES) + r")_(.+)$")
_RUN_RE = re.compile(r"^run_\d+_(.+)$")


def _classify(stem):
    """(section title, sort key, caption) for one output file stem."""
    m = _NODE_RE.match(stem)
    if m:
        node, phase, rest = m.groups()
        return (f"Node {int(node)} · {phase.replace('_', ' ')}",
                (2, int(node), PHASES.index(phase)), rest)
    m = _PHASE_RE.match(stem)
    if m:
        phase, rest = m.groups()
        return f"Energy · {phase.replace('_', ' ')}", (1, PHASES.index(phase)), rest
    m = _RUN_RE.match(stem)
    return "Overview", (0,), m.group(1) if m else stem


def write_gallery(directory, title="Plot Preview"):
    """Write <directory>/index.html linking every png/wav/json in it."""
    directory = Path(directory)
    sections = {}
    for f in sorted(directory.iterdir()):
        if f.suffix not in (".png", ".wav", ".json"):
            continue
        name, key, caption = _classify(f.stem)
        sec = sections.setdefault(name, {"key": key, "png": [], "wav": [], "json": []})
        sec[f.suffix[1:]].append((f.name, caption.replace("_", " ")))

    ordered = sorted(sections.items(), key=lambda kv: kv[1]["key"])
    nav, body = [], []
    for i, (name, sec) in enumerate(ordered):
        sid = f"s{i}"
        nav.append(f'<a href="#{sid}">{html.escape(name)}</a>')
        parts = [f'<section id="{sid}"><h2>{html.escape(name)}</h2>']
        if sec["wav"] or sec["json"]:
            parts.append('<div class="extras">')
            for fname, cap in sec["wav"]:
                parts.append(f'<label>{html.escape(cap)}<audio controls preload="none" '
                             f'src="{html.escape(fname)}"></audio></label>')
            for fname, cap in sec["json"]:
                parts.append(f'<a href="{html.escape(fname)}">{html.escape(cap)} (json)</a>')
            parts.append('</div>')
        parts.append('<div class="grid">')
        for fname, cap in sec["png"]:
            parts.append(f'<figure><img loading="lazy" src="{html.escape(fname)}" '
                         f'alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>')
        parts.append('</div></section>')
        body.append("".join(parts))

    page = _TEMPLATE.format(title=html.escape(title), nav="".join(nav), body="\n".join(body))
    out = directory / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg:#f6f6f4; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#dcdcd8; --accent:#2563eb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#141416; --card:#1f1f22; --fg:#ececee; --muted:#9a9aa0; --line:#33333a; --accent:#7aa2ff; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }}
  header {{ position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
           padding:10px 16px; }}
  header h1 {{ margin:0 0 6px; font-size:17px; }}
  nav {{ display:flex; flex-wrap:wrap; gap:6px; }}
  nav a {{ color:var(--fg); text-decoration:none; font-size:13px; padding:3px 9px; border:1px solid var(--line);
          border-radius:999px; background:var(--card); }}
  nav a:hover {{ border-color:var(--accent); color:var(--accent); }}
  .size {{ float:right; font-size:13px; color:var(--muted); }}
  main {{ padding:8px 16px 40px; }}
  section {{ scroll-margin-top:110px; }}
  h2 {{ font-size:16px; margin:26px 0 10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(var(--min, 420px), 1fr)); gap:12px; }}
  figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; cursor:zoom-in; }}
  figure img {{ display:block; width:100%; height:auto; background:#fff; }}
  figcaption {{ padding:6px 10px; font-size:13px; color:var(--muted); }}
  .extras {{ display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin-bottom:12px; font-size:13px; }}
  .extras label {{ display:flex; align-items:center; gap:8px; }}
  .extras audio {{ height:30px; }}
  .extras a {{ color:var(--accent); }}
  #lb {{ position:fixed; inset:0; background:rgba(0,0,0,.88); display:none; flex-direction:column;
        align-items:center; justify-content:center; z-index:10; padding:16px; cursor:zoom-out; }}
  #lb.open {{ display:flex; }}
  #lb img {{ max-width:100%; max-height:calc(100vh - 70px); background:#fff; }}
  #lb p {{ color:#ddd; margin:10px 0 0; font-size:14px; }}
</style>
</head>
<body>
<header>
  <label class="size">Thumbnail size <input id="size" type="range" min="260" max="900" value="420"></label>
  <h1>{title}</h1>
  <nav>{nav}</nav>
</header>
<main>
{body}
</main>
<div id="lb"><img alt=""><p></p></div>
<script>
  const figs = [...document.querySelectorAll('figure')];
  const lb = document.getElementById('lb'), lbImg = lb.querySelector('img'), lbCap = lb.querySelector('p');
  let cur = -1;
  function show(i) {{
    cur = (i + figs.length) % figs.length;
    const f = figs[cur];
    lbImg.src = f.querySelector('img').src;
    lbCap.textContent = f.closest('section').querySelector('h2').textContent + ' \\u2014 '
      + f.querySelector('figcaption').textContent + '  (' + (cur + 1) + '/' + figs.length + ', \\u2190/\\u2192 to browse, Esc to close)';
    lb.classList.add('open');
  }}
  figs.forEach((f, i) => f.addEventListener('click', () => show(i)));
  lb.addEventListener('click', () => lb.classList.remove('open'));
  document.addEventListener('keydown', e => {{
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') lb.classList.remove('open');
    if (e.key === 'ArrowRight') show(cur + 1);
    if (e.key === 'ArrowLeft') show(cur - 1);
  }});
  const size = document.getElementById('size');
  try {{ const s = localStorage.getItem('thumb'); if (s) size.value = s; }} catch (e) {{}}
  const apply = () => document.documentElement.style.setProperty('--min', size.value + 'px');
  size.addEventListener('input', () => {{ apply(); try {{ localStorage.setItem('thumb', size.value); }} catch (e) {{}} }});
  apply();
</script>
</body>
</html>
"""

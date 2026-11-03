"""Ekstrak gambar/tabel dari output notebook tersimpan; tidak melakukan fit model."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import re
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]

def _source(value):
    return value if isinstance(value, str) else "".join(value)

def _flat_table(html):
    frame = pd.read_html(io.StringIO(html), flavor="lxml")[0]
    columns = []
    for column in frame.columns:
        parts = column if isinstance(column, tuple) else [column]
        parts = [str(x) for x in parts if not str(x).startswith("Unnamed:")]
        columns.append(" / ".join(dict.fromkeys(parts)) if parts else "index")
    frame.columns = columns
    if len(frame.columns) and frame.columns[0] == "index":
        if frame.iloc[:, 0].tolist() == list(range(len(frame))):
            frame = frame.iloc[:, 1:]
    truncated = any(str(c) in {"...", "…"} for c in frame.columns)
    frame = frame.drop(columns=[c for c in frame.columns if str(c) in {"...", "…"}])
    return frame, truncated

def _md_table(frame, max_rows=14, max_cols=9):
    shown = frame.iloc[:max_rows, :max_cols]
    def cell(value):
        if pd.isna(value):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, shown.columns)) + " |",
             "| " + " | ".join(["---"] * len(shown.columns)) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in shown.itertuples(index=False, name=None))
    if frame.shape != shown.shape:
        lines.append(f"\nCuplikan {len(shown)} dari {len(frame)} baris dan {len(shown.columns)} dari {len(frame.columns)} kolom tersimpan.")
    return "\n".join(lines)

def table_image(frame, title, target, subtitle):
    shown = frame.iloc[:16, :8].copy()
    selected = [c for c in ["model", "AP_proxy", "precision_proxy", "recall_proxy", "F1_proxy", "alarm_rate"] if c in frame]
    if len(selected) >= 5:
        shown = frame[selected].iloc[:16].copy()
    regular_path = font_manager.findfont("DejaVu Sans")
    bold_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    body = ImageFont.truetype(regular_path, 19)
    header = ImageFont.truetype(bold_path, 19)
    title_font = ImageFont.truetype(bold_path, 28)
    small = ImageFont.truetype(regular_path, 16)
    width, margin = 1700, 35
    probe = ImageDraw.Draw(Image.new("RGB", (width, 100)))
    strings = [["—" if pd.isna(v) else str(v) for v in row] for row in shown.itertuples(index=False, name=None)]
    headings = [str(c).replace("_", " ") for c in shown.columns]
    weights = [min(42, max(12, len(h), *(len(row[j]) for row in strings))) for j,h in enumerate(headings)]
    available = width - 2 * margin
    widths = [int(available * v / sum(weights)) for v in weights]
    widths[-1] += available - sum(widths)
    def wrapped(text, max_width, font):
        lines, line = [], ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if probe.textbbox((0,0), trial, font=font)[2] <= max_width:
                line = trial
            else:
                if line:
                    lines.append(line)
                while probe.textbbox((0,0), word, font=font)[2] > max_width:
                    cut = len(word)-1
                    while cut > 1 and probe.textbbox((0,0), word[:cut], font=font)[2] > max_width:
                        cut -= 1
                    lines.append(word[:cut]); word = word[cut:]
                line = word
        if line:
            lines.append(line)
        return lines or [""]
    all_rows = [headings] + strings
    rendered = [[wrapped(v, widths[j]-22, header if i==0 else body) for j,v in enumerate(row)] for i,row in enumerate(all_rows)]
    heights = [max(len(v) for v in row)*27 + 22 for row in rendered]
    caption = "Tabel dirender dari output notebook yang tersimpan; presisi mengikuti output sumber."
    if frame.shape != shown.shape:
        caption += " Cuplikan; tabel tersimpan tersedia melalui tautan CSV."
    title_lines = wrapped(title, available, title_font)
    top = 35 + len(title_lines)*36 + 38
    bottom_lines = wrapped(caption, available, small)
    canvas = Image.new("RGB", (width, top+sum(heights)+len(bottom_lines)*23+35), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    y=25
    for line in title_lines:
        draw.text((margin,y),line,font=title_font,fill="#16425b"); y+=36
    draw.text((margin,y+6),subtitle,font=small,fill="#56636f")
    y=top
    for i,(row,height) in enumerate(zip(rendered,heights)):
        x=margin
        color = "#16425b" if i==0 else ("#eef4f6" if i%2 else "#ffffff")
        for j,lines in enumerate(row):
            draw.rectangle((x,y,x+widths[j],y+height),fill=color,outline="#d8e1e6")
            for k,line in enumerate(lines):
                draw.text((x+11,y+11+k*27),line,font=header if i==0 else body,fill="#ffffff" if i==0 else "#243440")
            x+=widths[j]
        y+=height
    y+=14
    for line in bottom_lines:
        draw.text((margin,y),line,font=small,fill="#56636f"); y+=23
    canvas.save(target)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook",type=Path,default=ROOT/"notebooks/UTS_ML_B10_Kelompok5.ipynb")
    args=parser.parse_args()
    nb=json.loads(args.notebook.read_text(encoding="utf-8"))
    image_dir=ROOT/"docs/images"
    table_dir=ROOT/"results/recorded/tables"
    stage_dir=ROOT/"docs/stages"
    for path in [image_dir,table_dir,stage_dir]: path.mkdir(parents=True,exist_ok=True)
    stages={}; inventory=[]; current=0
    for cell_index,cell in enumerate(nb["cells"]):
        source=_source(cell.get("source",[]))
        if cell["cell_type"]=="markdown":
            match=re.search(r"\*\*SECTION\s+(\d+)\s*:\s*(.+?)\*\*",source,re.I)
            if match:
                current=int(match.group(1))
                stages[current]={"title":match.group(2),"intro":source,"blocks":[],"first_table":None,"images":[]}
            continue
        for output_index,output in enumerate(cell.get("outputs",[])):
            if output.get("output_type")=="error": raise RuntimeError(f"Error output at cell {cell_index}")
            data=output.get("data",{})
            tag=f"s{current:02d}_c{cell_index:03d}_o{output_index:02d}"
            record={"section":current,"cell_index_zero_based":cell_index,"output_index_zero_based":output_index,
                    "execution_count":cell.get("execution_count"),"mime_types":list(data)}
            if "image/png" in data:
                payload=base64.b64decode(_source(data["image/png"]))
                target=image_dir/(tag+".png");target.write_bytes(payload)
                with Image.open(target) as im: record["image_size"]=list(im.size)
                record.update(kind="plot",path=target.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(payload).hexdigest())
                if current:
                    stages[current]["blocks"].append(f"![Output grafik Section {current}](../images/{target.name})")
                    stages[current]["images"].append(target.name)
            elif "text/html" in data and "<table" in _source(data["text/html"]) and current:
                frame,truncated=_flat_table(_source(data["text/html"]))
                target=table_dir/(tag+".csv");frame.to_csv(target,index=False)
                record.update(kind="table",path=target.relative_to(ROOT).as_posix(),rows=len(frame),
                              columns=list(map(str,frame.columns)),display_excerpt=bool(truncated))
                stages[current]["blocks"].append(f"[Tabel CSV dari output tersimpan](../../results/recorded/tables/{target.name})\n\n"+_md_table(frame))
                if stages[current]["first_table"] is None:
                    stages[current]["first_table"]=(frame,tag,cell_index,output_index)
            elif "text/markdown" in data and current:
                stages[current]["blocks"].append(_source(data["text/markdown"]))
                record["kind"]="explanation"
            else:
                continue
            inventory.append(record)
    gallery=["# Galeri hasil eksperimen per tahap", "",
        "Grafik PNG berikut diekstrak dari output notebook final. Gambar tabel dirender dari nilai yang tersimpan pada output tersebut. "
        "Ini merupakan urutan eksperimen dalam notebook, dengan provenance pada setiap section.", "",
        "[Kembali ke README](../README.md) · [Ringkasan eksperimen](EXPERIMENTS.md)", ""]
    for number,stage in sorted(stages.items()):
        page=f"{number:02d}.md"
        content=[f"# Section {number} {stage['title']}","",stage["intro"],"",
            "[Galeri seluruh tahap](../GALLERY.md) · [Notebook sumber](../../notebooks/UTS_ML_B10_Kelompok5.ipynb)",""]
        item=stage["first_table"]
        if item:
            frame,tag,cell_index,output_index=item
            png=image_dir/f"section_{number:02d}_table.png"
            table_image(frame,f"Section {number} — {stage['title']}",png,
                        f"B10 · Kelompok 5 · cell {cell_index}, output {output_index} (indeks mulai 0)")
            content.extend([f"![Cuplikan tabel Section {number}](../images/{png.name})",""])
            inventory.append({"section":number,"kind":"rendered_table","path":png.relative_to(ROOT).as_posix(),"derived_from":tag})
        content.extend(stage["blocks"])
        (stage_dir/page).write_text("\n\n".join(content)+"\n",encoding="utf-8")
        gallery.extend([f"## {number}. {stage['title']}","",f"[Lihat penjelasan dan seluruh output tahap ini](stages/{page})",""])
        first_image=stage["images"][0] if stage["images"] else (f"section_{number:02d}_table.png" if item else None)
        if first_image: gallery.extend([f"![Hasil Section {number}](images/{first_image})",""])
    (ROOT/"docs/GALLERY.md").write_text("\n".join(gallery)+"\n",encoding="utf-8")
    manifest={"notebook":args.notebook.name,"notebook_sha256":hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
              "source":"saved notebook outputs; no model training by this exporter", "items":inventory}
    (ROOT/"results/recorded/output_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"plots":sum(r["kind"]=="plot" for r in inventory),
        "tables":sum(r["kind"]=="table" for r in inventory),"table_images":sum(r["kind"]=="rendered_table" for r in inventory),
        "stages":len(stages)},indent=2))

if __name__=="__main__": main()

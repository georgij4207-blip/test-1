"""Write docs/turnaround_preview.gif and docs/qa_contact_sheet.png for quick review."""
import json
import os

from PIL import Image, ImageDraw

ASSETS = "web/assets"
os.makedirs("docs", exist_ok=True)
man = json.load(open(f"{ASSETS}/manifest.json", encoding="utf-8"))
fw, fh = man["frameWidth"], man["frameHeight"]
BG = (74, 80, 90, 255)
COL = {"ok": (90, 170, 90), "jump": (220, 70, 50), "near-duplicate": (230, 180, 40)}

frames = []
sheet = Image.new("RGB", (fw * 6, (fh + 22) * 6), (40, 43, 50))
d = ImageDraw.Draw(sheet)
for k, f in enumerate(man["frames"]):
    img = Image.open(f"{ASSETS}/frames/coyote_{f['angle']:03d}.png")
    bg = Image.new("RGBA", (fw, fh), BG)
    bg.alpha_composite(img)
    frames.append(bg.convert("P", palette=Image.ADAPTIVE))
    x, y = (k % 6) * fw, (k // 6) * (fh + 22)
    sheet.paste(bg.convert("RGB"), (x, y))
    t = f["to_next"]
    d.text((x + 4, y + 4), f"{f['angle']}°" + ("  interp" if f["interpolated"] else "  redrawn" if f.get("redrawn") else ""), fill="white")
    d.rectangle([x, y + fh, x + fw - 1, y + fh + 21], fill=COL[t["flag"]])
    d.text((x + 4, y + fh + 5), f"-> next: IoU {t['iou']:.2f} {t['flag']}", fill="black")
frames[0].save("docs/turnaround_preview.gif", save_all=True, append_images=frames[1:],
               duration=90, loop=0, optimize=True, disposal=2)
sheet.save("docs/qa_contact_sheet.png", optimize=True)

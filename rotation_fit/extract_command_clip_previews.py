"""Export one real frame per prepared recording, plus labeled contact sheets."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps


def extract(source: Path, output: Path):
    manifest = json.loads((source / "command_clips_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETE":
        raise ValueError("Prepared dataset is not complete")
    output.mkdir(parents=True, exist_ok=False)
    previews = []
    for number, recording in enumerate(manifest["recordings"], 1):
        prefix = recording["prefix"]
        with (source / f"{prefix}_inference_timing.csv").open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["raw_video_frame_index"] != ""]
        windows = recording["command_windows"]
        # Prefer a command with actual retained frames during its drive interval.
        candidates = []
        for window in windows:
            drive = [row for row in rows if window["command_host_s"] <=
                     float(row["camera_input_host_mono_ms"]) / 1000 <= window["stop_host_s"]]
            if drive:
                candidates.append((window, drive))
        if candidates:
            window, candidates_rows = max(candidates, key=lambda pair:
                                         pair[0]["stop_host_s"] - pair[0]["command_host_s"])
            midpoint = (window["command_host_s"] + window["stop_host_s"]) / 2
            row = min(candidates_rows, key=lambda item:
                      abs(float(item["camera_input_host_mono_ms"]) / 1000 - midpoint))
            selection = "middle of longest CAN rotation command with retained drive frames"
        else:
            row = rows[len(rows) // 2]
            selection = "middle retained frame; no frame lies inside a command hold"
        index = int(row["raw_video_frame_index"])
        capture = cv2.VideoCapture(str(source / recording["video_file"]))
        try:
            # Sequential decoding avoids depending on AVI seek accuracy.
            for _ in range(index + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {prefix}, index {index}")
        finally:
            capture.release()
        name = f"{number:02d}_{prefix.removeprefix('forklift_v4_recording_')}.png"
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(output / name)
        previews.append(dict(number=number, recording=prefix, image=name,
                             clip_frame_index=index, source_frame_i=int(row["frame_i"]),
                             captured_at=row.get("t_iso", ""), selection=selection))

    font_file = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(font_file), 21) if font_file.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_file), 17) if font_file.exists() else font
    sheets = []
    for page, start in enumerate(range(0, len(previews), 12), 1):
        group = previews[start:start + 12]
        tile_w, tile_h, gap = 400, 356, 16
        row_count = (len(group) + 2) // 3
        sheet = Image.new("RGB", (gap + 3 * (tile_w + gap), 62 + row_count * (tile_h + gap)), "#101820")
        draw = ImageDraw.Draw(sheet)
        draw.text((gap, 17), f"CAN rotation recordings | {start + 1:02d}-{start + len(group):02d} / {len(previews)}", font=font, fill="white")
        for slot, item in enumerate(group):
            x = gap + (slot % 3) * (tile_w + gap)
            y = 62 + (slot // 3) * (tile_h + gap)
            with Image.open(output / item["image"]) as original:
                thumb = ImageOps.contain(original, (tile_w, 300))
                sheet.paste(thumb, (x + (tile_w - thumb.width) // 2, y + (300 - thumb.height) // 2))
            stamp = item["recording"].removeprefix("forklift_v4_recording_")
            label = datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            draw.text((x, y + 305), f'{item["number"]:02d}  {label}', font=font, fill="white")
            draw.text((x, y + 333), f'Source frame {item["source_frame_i"]} | clip frame {item["clip_frame_index"]}', font=small, fill="#b4c4d0")
        name = f"contact_sheet_{page:02d}.jpg"
        sheet.save(output / name, quality=94)
        sheets.append(name)
    (output / "preview_manifest.json").write_text(json.dumps(
        dict(source=str(source.resolve()), count=len(previews), contact_sheets=sheets, frames=previews),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), images=len(previews), sheets=sheets)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "out/command_clips")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "out/command_clip_previews")
    args = parser.parse_args()
    extract(args.source, args.output)

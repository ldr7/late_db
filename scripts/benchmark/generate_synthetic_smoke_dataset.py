#!/usr/bin/env python3
"""Generate a small synthetic smoke dataset for retrieval pipeline testing.

This is not the final public benchmark. It creates simple, readable images that
exercise common retrieval shapes while public-source curation is still pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 900
HEIGHT = 650


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    return image, ImageDraw.Draw(image)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int = 24) -> None:
    draw.text(xy, value, fill="black", font=font(size))


def form_invoice() -> Image.Image:
    image, draw = canvas()
    text(draw, (50, 40), "ACME INFRASTRUCTURE INVOICE", 32)
    text(draw, (50, 95), "Invoice ID: INV-7429", 24)
    text(draw, (50, 130), "Payment due date: 2026-08-15", 24)
    text(draw, (50, 165), "Project: Riverside Bridge Inspection", 24)
    for y in [230, 280, 330, 380]:
        draw.line((50, y, 830, y), fill="black", width=2)
    for x in [50, 420, 620, 830]:
        draw.line((x, 230, x, 380), fill="black", width=2)
    text(draw, (70, 245), "Service")
    text(draw, (440, 245), "Hours")
    text(draw, (650, 245), "Amount")
    text(draw, (70, 295), "Crack mapping")
    text(draw, (440, 295), "12")
    text(draw, (650, 295), "$2400")
    text(draw, (70, 345), "Report writing")
    text(draw, (440, 345), "4")
    text(draw, (650, 345), "$800")
    text(draw, (620, 430), "Subtotal: $3200", 28)
    return image


def form_lab() -> Image.Image:
    image, draw = canvas()
    text(draw, (50, 45), "MATERIAL TEST REQUEST", 32)
    labels = [
        ("Specimen ID", "CYL-18B"),
        ("Concrete mix", "C40 high early strength"),
        ("Slump", "75 mm"),
        ("Compressive test date", "2026-07-18"),
    ]
    y = 125
    for key, value in labels:
        draw.rectangle((50, y, 820, y + 55), outline="black", width=2)
        text(draw, (70, y + 14), key)
        text(draw, (420, y + 14), value)
        y += 70
    draw.rectangle((50, 450, 820, 575), outline="black", width=2)
    text(draw, (70, 470), "Notes: core sample from north retaining wall")
    return image


def chart_revenue() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 40), "Quarterly Revenue by Region", 32)
    draw.line((90, 540, 820, 540), fill="black", width=3)
    draw.line((90, 120, 90, 540), fill="black", width=3)
    data = [180, 260, 220, 360]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    for i, val in enumerate(data):
        x0 = 150 + i * 150
        draw.rectangle((x0, 540 - val, x0 + 70, 540), fill=colors[i])
        text(draw, (x0 + 12, 555), labels[i])
    text(draw, (600, 140), "West region peaks in Q4", 24)
    return image


def chart_temperature() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 40), "Bridge Deck Temperature Trend", 32)
    draw.line((90, 540, 820, 540), fill="black", width=3)
    draw.line((90, 120, 90, 540), fill="black", width=3)
    points = [(120, 480), (250, 430), (380, 360), (510, 300), (640, 250), (770, 210)]
    draw.line(points, fill="#dc2626", width=5)
    for point in points:
        draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill="#dc2626")
    text(draw, (600, 565), "Time")
    text(draw, (110, 130), "Temperature rises steadily", 24)
    return image


def science_diagram() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 40), "Photosynthesis Inputs and Outputs", 32)
    draw.ellipse((350, 220, 560, 420), fill="#22c55e", outline="black", width=3)
    text(draw, (410, 305), "Leaf", 28)
    items = [("Sunlight", (90, 130), (350, 260)), ("CO2", (95, 400), (350, 340)), ("Water", (390, 550), (430, 420))]
    for label, start, end in items:
        text(draw, start, label, 24)
        draw.line((start[0] + 120, start[1] + 20, end[0], end[1]), fill="black", width=3)
    text(draw, (650, 250), "Oxygen", 24)
    draw.line((560, 300, 650, 275), fill="black", width=3)
    text(draw, (650, 360), "Glucose", 24)
    draw.line((560, 360, 650, 380), fill="black", width=3)
    return image


def circuit_diagram() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 40), "Sensor Circuit Diagram", 32)
    draw.rectangle((120, 250, 260, 360), outline="black", width=3)
    text(draw, (155, 290), "MCU", 26)
    draw.rectangle((580, 245, 760, 365), outline="black", width=3)
    text(draw, (615, 290), "TEMP\nSENSOR", 22)
    draw.line((260, 280, 580, 280), fill="black", width=3)
    draw.line((260, 330, 580, 330), fill="black", width=3)
    text(draw, (370, 245), "I2C SDA", 22)
    text(draw, (370, 345), "I2C SCL", 22)
    draw.rectangle((120, 460, 260, 530), outline="black", width=3)
    text(draw, (155, 480), "3.3V", 26)
    draw.line((190, 460, 190, 360), fill="black", width=3)
    return image


def floorplan() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 35), "Two Bedroom Floor Plan", 32)
    draw.rectangle((100, 110, 800, 570), outline="black", width=6)
    draw.line((430, 110, 430, 570), fill="black", width=5)
    draw.line((100, 330, 430, 330), fill="black", width=5)
    draw.line((430, 300, 800, 300), fill="black", width=5)
    text(draw, (190, 200), "Bedroom 1", 26)
    text(draw, (190, 430), "Bedroom 2", 26)
    text(draw, (560, 180), "Kitchen", 26)
    text(draw, (560, 420), "Living Room", 26)
    draw.arc((410, 300, 490, 380), start=270, end=360, fill="#2563eb", width=4)
    text(draw, (455, 320), "Door", 18)
    return image


def ui_settings() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 40), "Mobile Settings Screen", 32)
    draw.rounded_rectangle((250, 90, 650, 590), radius=30, outline="black", width=4)
    text(draw, (300, 135), "Settings", 30)
    rows = ["Notifications", "Dark Mode", "Location Access", "Backup over Wi-Fi"]
    y = 210
    for row in rows:
        text(draw, (295, y), row, 22)
        draw.rounded_rectangle((540, y, 615, y + 32), radius=16, outline="black", width=2)
        draw.ellipse((582, y + 4, 610, y + 28), fill="#22c55e")
        y += 75
    return image


def product_lamp() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 40), "Product Catalog: Desk Lamp", 32)
    draw.ellipse((350, 470, 560, 540), fill="#9ca3af", outline="black", width=3)
    draw.rectangle((445, 250, 465, 480), fill="#374151")
    draw.polygon([(340, 230), (575, 230), (520, 150), (395, 150)], fill="#facc15", outline="black")
    text(draw, (575, 320), "matte black stem", 24)
    text(draw, (575, 360), "yellow shade", 24)
    return image


def natural_scene() -> Image.Image:
    image, draw = canvas()
    draw.rectangle((0, 0, WIDTH, 390), fill="#bfdbfe")
    draw.rectangle((0, 390, WIDTH, HEIGHT), fill="#86efac")
    text(draw, (60, 40), "Outdoor Scene", 32)
    draw.rectangle((180, 360, 420, 450), fill="#dc2626", outline="black", width=3)
    draw.ellipse((210, 435, 260, 485), fill="black")
    draw.ellipse((340, 435, 390, 485), fill="black")
    draw.rectangle((560, 260, 590, 450), fill="#854d0e")
    draw.ellipse((500, 170, 650, 300), fill="#16a34a")
    text(draw, (170, 500), "red vehicle near a tree", 24)
    return image


def remote_sensing() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 35), "Synthetic Aerial Land Use", 32)
    colors = ["#16a34a", "#84cc16", "#64748b", "#0ea5e9", "#a3a3a3"]
    for row in range(6):
        for col in range(8):
            x0 = 80 + col * 90
            y0 = 110 + row * 80
            draw.rectangle((x0, y0, x0 + 85, y0 + 75), fill=colors[(row + col) % len(colors)])
    draw.rectangle((360, 250, 630, 330), fill="#0ea5e9")
    text(draw, (370, 350), "river beside industrial blocks", 24)
    return image


def table_report() -> Image.Image:
    image, draw = canvas()
    text(draw, (60, 40), "Bridge Inspection Defect Table", 32)
    headers = ["Element", "Defect", "Severity"]
    rows = [
        ["Pier 2", "cracking", "medium"],
        ["Deck joint", "corrosion", "high"],
        ["Abutment", "spalling", "low"],
    ]
    x_positions = [70, 330, 590, 820]
    y0 = 140
    for x in x_positions:
        draw.line((x, y0, x, y0 + 240), fill="black", width=2)
    for i in range(5):
        draw.line((70, y0 + i * 60, 820, y0 + i * 60), fill="black", width=2)
    for i, header in enumerate(headers):
        text(draw, (x_positions[i] + 15, y0 + 15), header, 22)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            text(draw, (x_positions[c] + 15, y0 + 75 + r * 60), cell, 20)
    return image


IMAGES = [
    ("smoke_form_invoice", "forms_ocr_docs", form_invoice, "Find the invoice with payment due date 2026-08-15.", "Find the document about Riverside Bridge Inspection."),
    ("smoke_form_lab", "forms_ocr_docs", form_lab, "Find the material test request for specimen CYL-18B.", "Find the form mentioning the north retaining wall."),
    ("smoke_chart_revenue", "charts_tables", chart_revenue, "Find the chart where the West region peaks in Q4.", "Find the bar chart about quarterly revenue."),
    ("smoke_chart_temperature", "charts_tables", chart_temperature, "Find the line chart where bridge deck temperature rises steadily.", "Find the plot about temperature over time."),
    ("smoke_science_diagram", "diagrams", science_diagram, "Find the diagram showing sunlight, CO2, and water entering a leaf.", "Find the photosynthesis diagram with oxygen output."),
    ("smoke_circuit_diagram", "diagrams", circuit_diagram, "Find the circuit diagram with I2C SDA and SCL lines.", "Find the sensor connected to an MCU."),
    ("smoke_floorplan", "floorplans_civil", floorplan, "Find the floorplan with two bedrooms beside a kitchen and living room.", "Find the drawing with a door between rooms."),
    ("smoke_ui_settings", "ui_screens", ui_settings, "Find the mobile settings screen with notification and location toggles.", "Find the UI screen with Backup over Wi-Fi."),
    ("smoke_product_lamp", "product_catalog", product_lamp, "Find the product image of a desk lamp with a yellow shade.", "Find the item with a matte black stem."),
    ("smoke_natural_scene", "natural_images", natural_scene, "Find the outdoor image with a red vehicle near a tree.", "Find the natural scene with grass and blue sky."),
    ("smoke_remote_sensing", "remote_sensing", remote_sensing, "Find the aerial image with a river beside industrial blocks.", "Find the remote sensing scene with mixed land use."),
    ("smoke_table_report", "enterprise_docs", table_report, "Find the defect table where deck joint corrosion severity is high.", "Find the bridge inspection table with pier cracking."),
]


def record(image_id: str, slice_name: str, path: Path, q1: str, q2: str) -> dict:
    return {
        "image_id": image_id,
        "slice": slice_name,
        "source": "synthetic_smoke",
        "license": "project-generated",
        "local_path": path.as_posix(),
        "source_url": "synthetic://late_db/universal_retrieval_smoke",
        "notes": "Generated smoke-test image for pipeline validation; not part of final public benchmark.",
        "queries": [
            {
                "query_id": f"{image_id}_q_direct",
                "kind": "direct",
                "text": q1,
                "positive_image_ids": [image_id],
                "hard_negative_image_ids": [],
                "tags": [slice_name, "synthetic_smoke"],
            },
            {
                "query_id": f"{image_id}_q_fine",
                "kind": "fine_grained",
                "text": q2,
                "positive_image_ids": [image_id],
                "hard_negative_image_ids": [],
                "tags": [slice_name, "synthetic_smoke", "fine_grained"],
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/universal_retrieval/images/smoke_synthetic"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/universal_retrieval/smoke_manifest.jsonl"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for image_id, slice_name, fn, q1, q2 in IMAGES:
        path = args.out_dir / f"{image_id}.png"
        fn().save(path)
        records.append(record(image_id, slice_name, path, q1, q2))

    with args.manifest.open("w") as handle:
        for item in records:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    print(f"wrote {len(records)} images to {args.out_dir}")
    print(f"wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()

"""
Conversor de anotações do Make Sense AI (YOLO .txt ou Pascal VOC .xml) para o
formato COCO, usado pelo `benchmark_pid_models.py`.

O documento do projeto deixa em aberto qual formato o Make Sense AI vai
exportar. Este utilitário cobre os dois formatos mais comuns exportados por
ele, para que o benchmark funcione independentemente da escolha.

USO
---
# Se o Make Sense AI exportou no formato YOLO (um .txt por imagem +
# um arquivo de classes, ex.: classes.txt / labels.txt):
python convert_annotations.py yolo \
    --images-dir dataset_bruto/images/train \
    --labels-dir dataset_bruto/labels/train \
    --classes-file dataset_bruto/classes.txt \
    --output dataset/annotations/train.json

# Se o Make Sense AI exportou no formato Pascal VOC (um .xml por imagem):
python convert_annotations.py voc \
    --images-dir dataset_bruto/images/train \
    --annotations-dir dataset_bruto/annotations_voc/train \
    --output dataset/annotations/train.json

Rode uma vez para o split de treino e uma vez para o de validação. As
imagens também precisam ficar organizadas em `dataset/images/train` e
`dataset/images/val` (é essa a estrutura que `benchmark_pid_models.py`
espera) - copie/mova as imagens para lá antes de rodar o benchmark.

IMPORTANTE: no formato YOLO os ids de classe são a ORDEM das linhas do
`classes.txt`. Garanta que o MESMO `classes.txt` (mesma ordem) seja usado
para converter tanto o split de treino quanto o de validação, senão os
ids de classe dos dois splits ficam inconsistentes entre si.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_classes(classes_file):
    with open(classes_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def convert_yolo(images_dir, labels_dir, classes_file, output):
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    classes = load_classes(classes_file)
    categories = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]

    images, annotations = [], []
    ann_id = 1
    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    for img_id, img_path in enumerate(img_paths, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_idx, xc, yc, bw, bh = parts
                cls_idx = int(cls_idx)
                xc, yc = float(xc) * w, float(yc) * h
                bw, bh = float(bw) * w, float(bh) * h
                x, y = xc - bw / 2, yc - bh / 2
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_idx + 1,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    _write_coco(images, annotations, categories, output)


def convert_voc(images_dir, annotations_dir, output):
    annotations_dir = Path(annotations_dir)
    xml_paths = sorted(annotations_dir.glob("*.xml"))

    class_names = []
    for xp in xml_paths:
        root = ET.parse(xp).getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_names:
                class_names.append(name)
    class_names = sorted(class_names)
    name_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    categories = [{"id": i, "name": n} for n, i in name_to_id.items()]

    images, annotations = [], []
    ann_id = 1
    for img_id, xp in enumerate(xml_paths, start=1):
        root = ET.parse(xp).getroot()
        filename = root.find("filename").text
        size = root.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)
        images.append({"id": img_id, "file_name": filename, "width": w, "height": h})

        for obj in root.findall("object"):
            name = obj.find("name").text
            bnd = obj.find("bndbox")
            xmin, ymin = float(bnd.find("xmin").text), float(bnd.find("ymin").text)
            xmax, ymax = float(bnd.find("xmax").text), float(bnd.find("ymax").text)
            bw, bh = xmax - xmin, ymax - ymin
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_id[name],
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    _write_coco(images, annotations, categories, output)


def _write_coco(images, annotations, categories, output):
    coco = {"images": images, "annotations": annotations, "categories": categories}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(
        f"OK: {len(images)} imagens, {len(annotations)} anotações, "
        f"{len(categories)} classes -> {output}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="formato", required=True)

    p_yolo = sub.add_parser("yolo", help="Converte anotações no formato YOLO (.txt)")
    p_yolo.add_argument("--images-dir", required=True)
    p_yolo.add_argument("--labels-dir", required=True)
    p_yolo.add_argument("--classes-file", required=True)
    p_yolo.add_argument("--output", required=True)

    p_voc = sub.add_parser("voc", help="Converte anotações no formato Pascal VOC (.xml)")
    p_voc.add_argument("--images-dir", required=True)
    p_voc.add_argument("--annotations-dir", required=True)
    p_voc.add_argument("--output", required=True)

    args = p.parse_args()
    if args.formato == "yolo":
        convert_yolo(args.images_dir, args.labels_dir, args.classes_file, args.output)
    elif args.formato == "voc":
        convert_voc(args.images_dir, args.annotations_dir, args.output)


if __name__ == "__main__":
    main()

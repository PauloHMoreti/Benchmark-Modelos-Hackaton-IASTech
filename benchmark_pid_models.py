"""
Benchmark de detectores de objetos para reconhecimento de equipamentos/símbolos em P&IDs
==========================================================================================

Compara três arquiteturas de detecção de objetos (as sugeridas no documento do
projeto) no MESMO dataset, com fine-tuning a partir de pesos pré-treinados:

    1) RT-DETR      - Transformer, via HuggingFace `transformers`
    2) Faster R-CNN - via torchvision (`fasterrcnn_resnet50_fpn_v2`)
    3) RetinaNet    - via torchvision (`retinanet_resnet50_fpn_v2`)

O PaddleOCR fica FORA deste benchmark: ele cuida do reconhecimento de texto
(tags) separadamente. Este script decide apenas qual arquitetura reconhece
melhor os equipamentos/símbolos (o objeto do "detector" no pipeline
P&ID -> equipamentos + OCR -> Excel).

--------------------------------------------------------------------------------
FORMATO DE DADOS ESPERADO
--------------------------------------------------------------------------------
O script espera anotações em formato COCO (json) - é o formato mais universal
e o único que os três modelos conseguem consumir sem conversões escondidas.
Estrutura esperada:

    dataset/
        images/
            train/*.jpg  (ou .png)
            val/*.jpg
        annotations/
            train.json
            val.json

Se o Make Sense AI exportar em outro formato (YOLO .txt ou Pascal VOC .xml),
use o utilitário `convert_annotations.py` (neste mesmo diretório) para
convertê-las para COCO antes de rodar este script.

--------------------------------------------------------------------------------
COMO RODAR
--------------------------------------------------------------------------------
1) Instale as dependências:
       pip install -r requirements.txt

2) (Recomendado na primeira vez) Rode um teste de fumaça com um dataset
   sintético, só para validar que o pipeline inteiro roda ponta a ponta -
   sem precisar do dataset real nem de GPU:
       python benchmark_pid_models.py --smoke-test --epochs 1

3) Quando o dataset real (já em formato COCO) estiver pronto:
       python benchmark_pid_models.py \
           --data-dir ./dataset \
           --epochs 15 \
           --batch-size 4 \
           --output-dir ./resultados_benchmark

O script vai, para cada modelo:
    - Carregar pesos pré-treinados (COCO) da arquitetura;
    - Fazer fine-tuning no dataset de vocês pelo mesmo número de épocas
      (usando o otimizador/LR "de fábrica" recomendado para cada arquitetura -
      ver a explicação sobre isso no final do módulo e no chat);
    - Avaliar no conjunto de validação: mAP@[.5:.95], mAP@.5, mAP@.75,
      AP por tamanho de objeto (small/medium/large), AP por classe, tempo de
      inferência médio (ms/imagem) e FPS;
    - Salvar uma tabela comparativa (CSV + JSON) e um gráfico de barras em
      `--output-dir`.

Rode `python benchmark_pid_models.py --help` para ver todas as opções.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ==================================================================================
# Utilidades gerais
# ==================================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_collate(batch):
    """Collate simples e comum a todos os modelos: cada wrapper decide,
    a partir dessa dupla (imagens PIL, anotações cru), o que fazer."""
    images, raw_targets = zip(*batch)
    return list(images), list(raw_targets)


# ==================================================================================
# Dataset (formato COCO)
# ==================================================================================


class CocoDetectionRaw(Dataset):
    """Lê um dataset em formato COCO e devolve (PIL.Image, anotação cru).

    A anotação cru de cada imagem é um dict com a lista de anotações no
    formato COCO original (bbox em [x, y, w, h], category_id = id ORIGINAL
    do json de categorias). Cada wrapper de modelo (Faster R-CNN / RetinaNet
    / RT-DETR) converte esse formato cru para o que a respectiva API espera -
    isso evita duplicar a leitura do dataset três vezes.
    """

    def __init__(self, images_dir, ann_file):
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(ann_file))
        self.img_ids = sorted(self.coco.getImgIds())

        # categorias originais do COCO json -> índice contíguo 0..C-1
        cat_ids = sorted(self.coco.getCatIds())
        self.catid2contig = {cid: i for i, cid in enumerate(cat_ids)}
        self.contig2catid = {i: cid for cid, i in self.catid2contig.items()}
        self.classes = [self.coco.cats[cid]["name"] for cid in cat_ids]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        raw_target = {
            "image_id": img_id,
            "width": img_info["width"],
            "height": img_info["height"],
            "annotations": anns,  # bbox em xywh, category_id ORIGINAL
        }
        return image, raw_target


def to_torchvision_target(raw_target, catid2contig):
    """Converte a anotação cru para o formato esperado por Faster R-CNN e
    RetinaNet no torchvision: boxes em xyxy absoluto e labels 1..C (0 é
    reservado, seguindo a convenção dos scripts de referência do
    torchvision para detecção)."""
    boxes, labels, areas, iscrowd = [], [], [], []
    for ann in raw_target["annotations"]:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(catid2contig[ann["category_id"]] + 1)
        areas.append(ann.get("area", w * h))
        iscrowd.append(ann.get("iscrowd", 0))

    boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
    areas_t = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
    iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)

    return {
        "boxes": boxes_t,
        "labels": labels_t,
        "image_id": torch.tensor([raw_target["image_id"]]),
        "area": areas_t,
        "iscrowd": iscrowd_t,
    }


def to_hf_coco_annotation(raw_target, catid2contig):
    """Converte a anotação cru para o formato que o RTDetrImageProcessor
    espera: lista de anotações estilo COCO, mas com category_id 0-indexado
    (sem classe de "background")."""
    anns = []
    for ann in raw_target["annotations"]:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        anns.append(
            {
                "image_id": raw_target["image_id"],
                "category_id": catid2contig[ann["category_id"]],
                "bbox": [x, y, w, h],
                "area": ann.get("area", w * h),
                "iscrowd": ann.get("iscrowd", 0),
            }
        )
    return {"image_id": raw_target["image_id"], "annotations": anns}


# ==================================================================================
# Wrappers de modelo - interface comum:
#   .build(num_classes, device)
#   .train_one_epoch(loader, optimizer, device) -> loss médio
#   .prepare_eval_batch(images, raw_targets, device) -> model_input
#   .predict_batch(model_input, raw_targets, device) -> resultados por imagem
#   .num_trainable_params()
# ==================================================================================


class TorchvisionDetectorBase:
    def __init__(self, catid2contig, contig2catid):
        self.catid2contig = catid2contig
        self.contig2catid = contig2catid
        self.model = None

    def prepare_train_batch(self, images, raw_targets, device):
        tv_images = [torchvision.transforms.functional.to_tensor(img).to(device) for img in images]
        tv_targets = [
            {k: v.to(device) for k, v in to_torchvision_target(rt, self.catid2contig).items()} for rt in raw_targets
        ]
        return tv_images, tv_targets

    def prepare_eval_batch(self, images, raw_targets, device):
        return [torchvision.transforms.functional.to_tensor(img).to(device) for img in images]

    def train_one_epoch(self, loader, optimizer, device):
        self.model.train()
        total_loss, n = 0.0, 0
        for images, raw_targets in tqdm(loader, desc=f"train [{self.name}]", leave=False):
            tv_images, tv_targets = self.prepare_train_batch(images, raw_targets, device)
            loss_dict = self.model(tv_images, tv_targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def predict_batch(self, model_input, raw_targets, device):
        self.model.eval()
        outputs = self.model(model_input)
        results = []
        for out in outputs:
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()
            keep = labels > 0  # descarta a classe 0 ("background"), por segurança
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            orig_cat_ids = [self.contig2catid[int(l) - 1] for l in labels]
            results.append({"boxes": boxes, "scores": scores, "category_ids": orig_cat_ids})
        return results

    def num_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class FasterRCNNDetector(TorchvisionDetectorBase):
    name = "Faster R-CNN"

    def build(self, num_classes, device):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # +1 pela classe "background", exigida pelo Faster R-CNN
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        self.model = model.to(device)
        return self.model


class RetinaNetDetector(TorchvisionDetectorBase):
    name = "RetinaNet"

    def build(self, num_classes, device):
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(weights=weights)
        out_channels = model.backbone.out_channels
        num_anchors = model.head.classification_head.num_anchors
        # Mantemos o mesmo espaço de rótulos "+1" usado no Faster R-CNN (o
        # índice 0 fica sem uso) só para reaproveitar `to_torchvision_target`
        # sem duplicar código - isso é inofensivo para o RetinaNet.
        model.head = RetinaNetHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,
        )
        self.model = model.to(device)
        return self.model


class RTDetrDetector:
    name = "RT-DETR"

    def __init__(self, catid2contig, contig2catid, classes, checkpoint="PekingU/rtdetr_r50vd_coco_o365"):
        if not _HF_AVAILABLE:
            raise RuntimeError("Pacote `transformers` não encontrado. Rode: pip install transformers")
        self.catid2contig = catid2contig
        self.contig2catid = contig2catid
        self.classes = classes
        self.checkpoint = checkpoint
        self.processor = None
        self.model = None

    def build(self, num_classes, device):
        id2label = {i: name for i, name in enumerate(self.classes)}
        label2id = {name: i for i, name in id2label.items()}
        self.processor = RTDetrImageProcessor.from_pretrained(self.checkpoint)
        self.model = RTDetrForObjectDetection.from_pretrained(
            self.checkpoint,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        ).to(device)
        return self.model

    def train_one_epoch(self, loader, optimizer, device):
        self.model.train()
        total_loss, n = 0.0, 0
        for images, raw_targets in tqdm(loader, desc=f"train [{self.name}]", leave=False):
            hf_targets = [to_hf_coco_annotation(rt, self.catid2contig) for rt in raw_targets]
            encoding = self.processor(images=list(images), annotations=hf_targets, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in encoding["labels"]]
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n += 1
        return total_loss / max(n, 1)

    def prepare_eval_batch(self, images, raw_targets, device):
        encoding = self.processor(images=list(images), return_tensors="pt")
        return {k: v.to(device) for k, v in encoding.items()}

    @torch.no_grad()
    def predict_batch(self, model_input, raw_targets, device):
        self.model.eval()
        outputs = self.model(**model_input)
        target_sizes = torch.tensor([[rt["height"], rt["width"]] for rt in raw_targets], device=device)
        # threshold baixo (mesma filosofia do box_score_thresh=0.05 do
        # torchvision) para não truncar a curva precisão-recall usada no mAP
        processed = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.05
        )
        results = []
        for p in processed:
            boxes = p["boxes"].detach().cpu()
            scores = p["scores"].detach().cpu()
            labels = p["labels"].detach().cpu()
            orig_cat_ids = [self.contig2catid[int(l)] for l in labels]
            results.append({"boxes": boxes, "scores": scores, "category_ids": orig_cat_ids})
        return results

    def num_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def build_wrapper(name, dataset_train, args):
    catid2contig = dataset_train.catid2contig
    contig2catid = dataset_train.contig2catid
    classes = dataset_train.classes
    if name == "faster_rcnn":
        return FasterRCNNDetector(catid2contig, contig2catid)
    if name == "retinanet":
        return RetinaNetDetector(catid2contig, contig2catid)
    if name == "rtdetr":
        return RTDetrDetector(catid2contig, contig2catid, classes, checkpoint=args.rtdetr_checkpoint)
    raise ValueError(f"Modelo desconhecido: {name}")


def build_optimizer(wrapper, args):
    """Usa o otimizador tipicamente recomendado para cada família de
    arquitetura, em vez de forçar hiperparâmetros idênticos entre CNN e
    Transformer - ver a explicação sobre "comparação justa" no chat."""
    params = [p for p in wrapper.model.parameters() if p.requires_grad]
    if isinstance(wrapper, RTDetrDetector):
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)


# ==================================================================================
# Avaliação (mAP via pycocotools + latência/FPS)
# ==================================================================================


def evaluate_model(wrapper, loader, device, warmup=3):
    all_preds = []
    latencies = []
    batch_idx = 0
    # Em datasets pequenos (ex.: --smoke-test) o nº de batches pode ser menor
    # que o warmup padrão - sem isso, a latência/FPS ficariam sempre NaN.
    try:
        n_batches = len(loader)
        warmup = min(warmup, max(n_batches - 1, 0))
    except TypeError:
        pass  # loader sem __len__ (não deve acontecer aqui, mas por segurança)

    for images, raw_targets in tqdm(loader, desc=f"eval [{wrapper.name}]", leave=False):
        model_input = wrapper.prepare_eval_batch(images, raw_targets, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = wrapper.predict_batch(model_input, raw_targets, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        batch_idx += 1
        if batch_idx > warmup:  # ignora os primeiros batches (warmup de CUDA/cache)
            latencies.append(dt / len(raw_targets))

        for res, rt in zip(results, raw_targets):
            for box, score, cat_id in zip(res["boxes"].tolist(), res["scores"].tolist(), res["category_ids"]):
                x1, y1, x2, y2 = box
                all_preds.append(
                    {
                        "image_id": int(rt["image_id"]),
                        "category_id": int(cat_id),
                        "bbox": [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
                        "score": float(score),
                    }
                )

    if latencies:
        avg_latency_ms = 1000.0 * (sum(latencies) / len(latencies))
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else float("nan")
    else:
        avg_latency_ms, fps = float("nan"), float("nan")

    return all_preds, avg_latency_ms, fps


def per_class_ap(coco_eval, class_names):
    """Extrai AP@[.5:.95] por classe a partir do objeto COCOeval já
    acumulado (coco_eval.accumulate() precisa já ter rodado)."""
    precisions = coco_eval.eval["precision"]  # shape [T, R, K, A, M]
    results = {}
    for k, name in enumerate(class_names):
        precision = precisions[:, :, k, 0, -1]  # todos IoU / recall, área=all, maxDets=100
        precision = precision[precision > -1]
        results[name] = float(precision.mean()) if precision.size else float("nan")
    return results


def run_coco_eval(coco_gt, predictions, class_names):
    if not predictions:
        print("  [AVISO] Nenhuma detecção gerada - métricas de mAP não puderam ser calculadas.")
        return None

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats

    metrics = {
        "AP@[.5:.95]": float(stats[0]),
        "AP@.5": float(stats[1]),
        "AP@.75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR@100": float(stats[8]),
    }
    metrics["per_class_AP@[.5:.95]"] = per_class_ap(coco_eval, class_names)
    return metrics


# ==================================================================================
# Dataset sintético (para --smoke-test, sem depender do dataset real)
# ==================================================================================


def make_synthetic_dataset(root: Path, n_train=12, n_val=6, img_size=256):
    root.mkdir(parents=True, exist_ok=True)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    classes = ["valvula", "bomba", "tanque"]
    categories = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    colors = [(200, 50, 50), (50, 150, 50), (50, 50, 200)]

    def gen_split(split, n):
        images, annotations = [], []
        ann_id = 1
        for i in range(n):
            img = Image.new("RGB", (img_size, img_size), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(1, 4)):
                cls_idx = random.randrange(len(classes))
                w, h = random.randint(20, 50), random.randint(20, 50)
                x = random.randint(0, img_size - w)
                y = random.randint(0, img_size - h)
                draw.rectangle([x, y, x + w, y + h], outline=colors[cls_idx], width=3)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": i + 1,
                        "category_id": cls_idx + 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
            fname = f"{split}_{i:03d}.png"
            img.save(root / "images" / split / fname)
            images.append({"id": i + 1, "file_name": fname, "width": img_size, "height": img_size})
        return {"images": images, "annotations": annotations, "categories": categories}

    for split, n in [("train", n_train), ("val", n_val)]:
        with open(root / "annotations" / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(gen_split(split, n), f)

    print(f"[smoke-test] dataset sintético criado em: {root.resolve()}")
    return root


# ==================================================================================
# Relatório final
# ==================================================================================


def save_checkpoint(wrapper, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)  # MB


def save_and_report(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resultados_completos.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    rows = [{k: v for k, v in r.items() if k != "per_class_AP@[.5:.95]"} for r in all_results]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "resultados_resumo.csv", index=False)

    print("\n" + "=" * 88)
    print("RESUMO COMPARATIVO")
    print("=" * 88)
    if not df.empty:
        print(df.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if "AP@[.5:.95]" in df.columns:
            axes[0].bar(df["model"], df["AP@[.5:.95]"])
        axes[0].set_title("mAP@[.5:.95]")
        axes[0].set_ylabel("mAP")
        axes[1].bar(df["model"], df["fps"])
        axes[1].set_title("Velocidade (FPS)")
        axes[1].set_ylabel("FPS")
        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(output_dir / "comparacao.png", dpi=150)
        plt.close(fig)

    print(f"\nResultados salvos em: {output_dir.resolve()}")
    print("  - resultados_resumo.csv     (tabela comparativa)")
    print("  - resultados_completos.json (inclui AP por classe)")
    print("  - comparacao.png            (gráfico mAP x FPS)")
    print("  - <modelo>.pt               (pesos de cada modelo treinado)")


# ==================================================================================
# main
# ==================================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, default="./dataset", help="Pasta com images/ e annotations/ em COCO")
    p.add_argument("--output-dir", type=str, default="./resultados_benchmark")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005, help="LR do SGD usado por Faster R-CNN / RetinaNet")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models",
        nargs="+",
        choices=["faster_rcnn", "retinanet", "rtdetr"],
        default=None,
        help="Subconjunto de modelos a rodar (padrão: os 3 sugeridos no documento)",
    )
    p.add_argument("--rtdetr-checkpoint", type=str, default="PekingU/rtdetr_r50vd_coco_o365")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Roda com um dataset sintético só para validar o pipeline (não usa --data-dir)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    set_seed(args.seed)

    if args.smoke_test:
        data_dir = make_synthetic_dataset(Path(args.output_dir) / "_smoke_dataset")
    else:
        data_dir = Path(args.data_dir)

    ds_train = CocoDetectionRaw(data_dir / "images" / "train", data_dir / "annotations" / "train.json")
    ds_val = CocoDetectionRaw(data_dir / "images" / "val", data_dir / "annotations" / "val.json")
    num_classes = len(ds_train.classes)
    print(f"Classes ({num_classes}): {ds_train.classes}")
    print(f"Treino: {len(ds_train)} imagens | Validação: {len(ds_val)} imagens")

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=default_collate, num_workers=args.num_workers
    )

    model_names = args.models or ["faster_rcnn", "retinanet", "rtdetr"]
    all_results = []

    for name in model_names:
        print("\n" + "-" * 88)
        print(f"Modelo: {name}")
        print("-" * 88)
        try:
            wrapper = build_wrapper(name, ds_train, args)
            wrapper.build(num_classes, device)
        except Exception as e:  # falta de pacote, sem internet para baixar pesos, etc.
            print(f"[AVISO] Não foi possível carregar '{name}': {e}")
            print("Pulando este modelo no benchmark.")
            continue

        optimizer = build_optimizer(wrapper, args)

        t_train0 = time.perf_counter()
        train_losses = []
        for epoch in range(args.epochs):
            loss = wrapper.train_one_epoch(train_loader, optimizer, device)
            train_losses.append(loss)
            print(f"[{wrapper.name}] época {epoch + 1}/{args.epochs} - loss médio: {loss:.4f}")
        train_time_s = time.perf_counter() - t_train0

        predictions, latency_ms, fps = evaluate_model(wrapper, val_loader, device)
        metrics = run_coco_eval(ds_val.coco, predictions, ds_train.classes)

        model_size_mb = save_checkpoint(wrapper, Path(args.output_dir) / f"{name}.pt")

        result = {
            "model": wrapper.name,
            "n_params_milhoes": wrapper.num_trainable_params() / 1e6,
            "tamanho_disco_mb": model_size_mb,
            "tempo_treino_s": train_time_s,
            "loss_final_treino": train_losses[-1] if train_losses else None,
            "latencia_ms_por_imagem": latency_ms,
            "fps": fps,
        }
        if metrics:
            result.update(metrics)
        all_results.append(result)

    save_and_report(all_results, args.output_dir)


if __name__ == "__main__":
    main()

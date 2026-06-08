# -*- coding: utf-8 -*-
"""
GXX_mamografia_lcc_unico_windows_v7_medical_ui.py

Script unico para o trabalho de Segmentacao e Classificacao de Imagens Mamograficas
Dataset utilizado: LCC
Redes utilizadas: EfficientNet + ResNet
 Segmentações utilizadas: Otsu, Filtro Conexo (Attribute Filtering)

"""

import os
import re
import csv
import time
import shutil
import zipfile
import threading
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

THIRD_PARTY_IMPORT_ERROR = None
try:
    import numpy as np
    from PIL import Image, ImageTk
    import cv2
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
except ImportError as e:
    THIRD_PARTY_IMPORT_ERROR = e

"""
-----------------------------------------------------
- Configurações gerais e declaração de constantes
------------------------------------------------------
"""

APP_TITLE = "MammoClass AI - LCC | Interface Médica | EfficientNet + ResNet | V7"
PROCESSED_DIR = Path("dataset_lcc_processado")
MODELS_DIR = Path("modelos_treinados")
RESULTS_DIR = Path("resultados")
TEMP_DIR = Path("_temp_lcc")

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

CLASS_MAP = {
    "D": {"folder": "D_BIRADS_I", "birads": "BIRADS I", "label_4": 0, "label_bin": 0},
    "E": {"folder": "E_BIRADS_II", "birads": "BIRADS II", "label_4": 1, "label_bin": 0},
    "F": {"folder": "F_BIRADS_III", "birads": "BIRADS III", "label_4": 2, "label_bin": 1},
    "G": {"folder": "G_BIRADS_IV", "birads": "BIRADS IV", "label_4": 3, "label_bin": 1},
}

LABEL_NAMES_4 = ["BIRADS I", "BIRADS II", "BIRADS III", "BIRADS IV"]
LABEL_NAMES_BIN = ["I+II", "III+IV"]
AUGMENT_ANGLES = [-20, -10, 0, 10, 20]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3




"""
-----------------------------------------------------
- Utilitários e arquivos de log
------------------------------------------------------
"""

def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS

def safe_stem(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", path.stem)

def get_image_number(filename: str):
    nums = re.findall(r"\d+", filename)
    if not nums:
        return None
    return int(nums[-1])

def infer_class_from_path(path: Path):
    candidates = [path.name.upper()] + [p.name.upper() for p in path.parents]
    for text in candidates:
        text = text.replace("+", " ").replace("-", "_")
        for cls in CLASS_MAP:
            if re.match(rf"^\s*{cls}(\s|_|\.|$)", text):
                return cls
            if f" {cls} " in f" {text} ":
                return cls
    return None

def find_7zip_executable():
    candidates = [
        "7z",
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for c in candidates:
        if shutil.which(c) or Path(c).exists():
            return c
    return None

def check_dependencies_or_show_help():
    if THIRD_PARTY_IMPORT_ERROR is None:
        return True
    msg = (
        "Faltam bibliotecas obrigatorias.\n\n"
        f"Erro original: {THIRD_PARTY_IMPORT_ERROR}"
    )
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependencias ausentes", msg)
    root.destroy()
    return False


"""
---------------------------------------------------------
- Leitura, normalização e segmentação da mama
---------------------------------------------------------
"""
def read_image_as_uint8_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, (1, 99))
    if p99 <= p1:
        mn, mx = float(arr.min()), float(arr.max())
    else:
        mn, mx = float(p1), float(p99)
    arr = np.clip(arr, mn, mx)
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)

def segment_otsu(gray: np.ndarray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.mean(th > 0)
    if white_ratio > 0.85:
        th = cv2.bitwise_not(th)
    h, w = gray.shape
    k = max(5, int(min(h, w) * 0.015))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        mask = (gray > 5).astype(np.uint8) * 255
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        mask = (labels == largest_label).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    return segmented, mask

def segment_morphological_reconstruction(gray: np.ndarray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(th == 255) > 0.85:
        th = cv2.bitwise_not(th)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    th_opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th_opened, connectivity=8)

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        final_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        final_mask = th_opened

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    segmented = cv2.bitwise_and(gray, gray, mask=final_mask)
    return segmented, final_mask

def segment_breast_region(gray: np.ndarray, method="Otsu (Padrão)"):
    if method == "Filtro Conexo (Attribute Filtering)":
        return segment_morphological_reconstruction(gray)
    else:
        return segment_otsu(gray)

def save_gray_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


"""
---------------------------
- Preparação do dataset
---------------------------
"""

def extract_lcc_source(source_path: Path, log=print) -> Path:
    source_path = Path(source_path)
    if source_path.is_dir():
        log(f"[OK] Usando pasta ja extraida: {source_path}")
        return source_path
    if source_path.suffix.lower() != ".zip":
        raise ValueError("Selecione o LCC.zip ou uma pasta ja extraida do LCC.")
    seven_zip = find_7zip_executable()
    if seven_zip is None:
        raise RuntimeError("Nao encontrei o 7-Zip no Windows.")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_out = TEMP_DIR / "zip_extraido"
    zip_out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_path, "r") as z:
        z.extractall(zip_out)
    rar_files = [p for p in zip_out.rglob("*.rar") if "__MACOSX" not in str(p)]
    if not rar_files:
        return zip_out
    rar_out = TEMP_DIR / "rar_extraido"
    rar_out.mkdir(parents=True, exist_ok=True)
    for rar in rar_files:
        cls = infer_class_from_path(rar)
        folder_name = CLASS_MAP[cls]["folder"] if cls else safe_stem(rar)
        class_out = rar_out / folder_name
        class_out.mkdir(parents=True, exist_ok=True)
        cmd = [seven_zip, "x", "-y", str(rar), f"-o{class_out}"]
        subprocess.run(cmd, check=True)
    return rar_out

def prepare_dataset_lcc(source_path: Path, log=print, seg_method="Otsu (Padrão)"):
    ensure_dirs()
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    extracted_root = extract_lcc_source(source_path, log=log)
    manifest = []
    image_paths = [p for p in extracted_root.rglob("*") if p.is_file() and is_image_file(p)]
    image_paths = [p for p in image_paths if "__MACOSX" not in str(p)]
    if not image_paths:
        raise RuntimeError("Nenhuma imagem encontrada.")
    log(f"[INFO] Imagens encontradas: {len(image_paths)}")
    for idx, img_path in enumerate(image_paths, start=1):
        cls = infer_class_from_path(img_path)
        if cls is None: continue
        img_num = get_image_number(img_path.name)
        if img_num is None: continue
        split = "test" if img_num % 4 == 0 else "train"
        class_folder = CLASS_MAP[cls]["folder"]
        gray = read_image_as_uint8_gray(img_path)
        segmented, mask = segment_breast_region(gray, method=seg_method)
        base_name = f"{cls}_{img_num:04d}_{safe_stem(img_path)}.png"
        original_out = PROCESSED_DIR / "original" / split / class_folder / base_name
        segmented_out = PROCESSED_DIR / "segmentado" / split / class_folder / base_name
        mask_out = PROCESSED_DIR / "mascaras" / split / class_folder / base_name
        save_gray_png(gray, original_out)
        save_gray_png(segmented, segmented_out)
        save_gray_png(mask, mask_out)
        manifest.append({
            "original_source": str(img_path),
            "original_processed": str(original_out),
            "segmented_processed": str(segmented_out),
            "mask": str(mask_out),
            "filename": base_name,
            "image_number": img_num,
            "split": split,
            "class_letter": cls,
            "birads": CLASS_MAP[cls]["birads"],
            "label_4classes": CLASS_MAP[cls]["label_4"],
            "label_binary": CLASS_MAP[cls]["label_bin"],
        })
        if idx % 20 == 0:
            log(f"[INFO] Processadas {idx}/{len(image_paths)} imagens...")
    manifest_path = PROCESSED_DIR / "manifest_lcc.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    log("\n[OK] Dataset preparado.")
    return manifest

"""
----------------------------------
- Dataset pytorch e redes neurais
----------------------------------
"""



class MammographyDataset(Dataset):

    def __init__(self, root_dir: Path, task="4classes", train=True, image_size=224):
        self.root_dir = Path(root_dir)
        self.task = task
        self.train = train
        self.samples = []
        for cls in ["D", "E", "F", "G"]:
            folder = self.root_dir / CLASS_MAP[cls]["folder"]
            if not folder.exists(): continue
            label = CLASS_MAP[cls]["label_4"] if task == "4classes" else CLASS_MAP[cls]["label_bin"]
            for p in folder.rglob("*.png"):
                self.samples.append((p, label, cls))
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.samples) * len(AUGMENT_ANGLES) if self.train else len(self.samples)
    

    def __getitem__(self, idx):
        if self.train:
            sample_idx = idx // len(AUGMENT_ANGLES)
            angle = AUGMENT_ANGLES[idx % len(AUGMENT_ANGLES)]
        else:
            sample_idx = idx
            angle = 0
        path, label, _ = self.samples[sample_idx]
        img = Image.open(path).convert("L").convert("RGB")
        if self.train:
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        return self.tf(img), torch.tensor(label, dtype=torch.long), str(path)

def build_model(model_name="efficientnet_b0", num_classes=4, pretrained=True):
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        for p in model.features.parameters(): p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model, "features"
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        for name, p in model.named_parameters():
            if not name.startswith("fc"): p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "layer4"
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        for name, p in model.named_parameters():
            if not name.startswith("fc"): p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "layer4"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_file_name(model_name, task, input_kind, seg_method=""):
    suffix = f"_{safe_stem(Path(seg_method))}" if input_kind == "segmentado" and seg_method else ""
    return MODELS_DIR / f"{model_name}_{task}_{input_kind}{suffix}.pt"


"""
------------------------------------
- Treinamento, avaliação e GRAD-CAM
------------------------------------
"""

def train_selected_model(model_name, task, input_kind, epochs=5, batch_size=8, lr=1e-3, log=print, seg_method=""):
    ensure_dirs()
    input_root = PROCESSED_DIR / input_kind
    train_root, test_root = input_root / "train", input_root / "test"
    if not train_root.exists(): raise RuntimeError("Dataset ainda nao preparado.")
    num_classes = 4 if task == "4classes" else 2
    device = get_device()

    train_ds = MammographyDataset(train_root, task=task, train=True, image_size=DEFAULT_IMAGE_SIZE)
    test_ds = MammographyDataset(test_root, task=task, train=False, image_size=DEFAULT_IMAGE_SIZE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model, _ = build_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history, start_time = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        metrics = evaluate_in_memory(model, test_loader, num_classes, device)
        history.append({
            "epoch": epoch, "train_loss": running_loss/max(total,1),
            "train_acc": correct/max(total,1), "test_acc": metrics["accuracy"], "test_f1": metrics["f1"]
        })
        log(f"[EPOCA {epoch}/{epochs}] loss={running_loss/max(total,1):.4f} test_acc={metrics['accuracy']:.4f}")

    elapsed = time.time() - start_time
    ckpt_path = model_file_name(model_name, task, input_kind, seg_method)
    torch.save({
        "model_name": model_name, "task": task, "input_kind": input_kind, "num_classes": num_classes,
        "state_dict": model.state_dict(), "image_size": DEFAULT_IMAGE_SIZE
    }, ckpt_path)

    suffix = f"_{safe_stem(Path(seg_method))}" if input_kind == "segmentado" and seg_method else ""
    history_path = RESULTS_DIR / f"historico_{model_name}_{task}_{input_kind}{suffix}.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


    log(f"\n[OK] Modelo e historico salvos! Tempo: {elapsed:.2f}s")
    return evaluate_saved_model(model_name, task, input_kind, log=log, seg_method=seg_method)

def evaluate_in_memory(model, loader, num_classes, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())
    
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    else:
        recalls, specs = [], []
        for i in labels:
            tp, fn = cm[i, i], cm[i, :].sum() - cm[i, i]
            fp, tn = cm[:, i].sum() - cm[i, i], cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        sens, spec = float(np.mean(recalls)), float(np.mean(specs))
        prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    return {"confusion_matrix": cm, "accuracy": float(acc), "precision": float(prec), "sensitivity": float(sens), "specificity": float(spec), "f1": float(f1)}

def load_checkpoint_model(model_name, task, input_kind, seg_method=""):
    ckpt_path = model_file_name(model_name, task, input_kind, seg_method)
    if not ckpt_path.exists(): raise RuntimeError(f"Modelo nao encontrado: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model, _ = build_model(checkpoint["model_name"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])

    return model, checkpoint

def evaluate_saved_model(model_name, task, input_kind, log=print, seg_method=""):
    device = get_device()
    model, checkpoint = load_checkpoint_model(model_name, task, input_kind, seg_method)
    model = model.to(device)
    test_ds = MammographyDataset(PROCESSED_DIR / input_kind / "test", task=task, train=False, image_size=checkpoint.get("image_size", 224))
    test_loader = DataLoader(test_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    metrics = evaluate_in_memory(model, test_loader, checkpoint["num_classes"], device)
    names = LABEL_NAMES_4 if checkpoint["num_classes"] == 4 else LABEL_NAMES_BIN
    suffix = f"_{safe_stem(Path(seg_method))}" if input_kind == "segmentado" and seg_method else ""
    result_path = RESULTS_DIR / f"metricas_{model_name}_{task}_{input_kind}{suffix}.txt"
    
    relatorio_txt = [
        "-----------------------------------------",
        "Relatorio da avaliação do modelo treinado",
        "-----------------------------------------",
        f"Rede Neural: {model_name}",
        f"Tarefa: {task}",
        f"Entrada: {input_kind}",
        f"Método de Segmentação: {seg_method if input_kind == 'segmentado' else 'N/A'}",
        "----------------------------------------",
        f"Acurácia Geral: {metrics['accuracy']:.4f}",
        f"Precisão Média: {metrics['precision']:.4f}",
        f"Sensibilidade (Recall): {metrics['sensitivity']:.4f}",
        f"Especificidade: {metrics['specificity']:.4f}",
        f"F1-Score: {metrics['f1']:.4f}",
        "----------------------------------------",
        "Matriz de Confusão:",
        f"Classes: {' | '.join(names)}",
        str(metrics["confusion_matrix"]),
        "----------------------------------------"
    ]
    
    texto_final = "\n".join(relatorio_txt)
    with open(result_path, "w", encoding="utf-8") as f: 
        f.write(texto_final)
        
    log("\n" + texto_final)
    log(f"[OK] Relatório salvo na pasta resultados: {result_path.name}")
    return metrics

def classify_image_with_gradcam(image_path: Path, model_name, task, input_kind, log=print, seg_method="Otsu (Padrão)"):
    device = get_device()
    model, checkpoint = load_checkpoint_model(model_name, task, input_kind, seg_method)
    model = model.to(device).eval()

    gray = read_image_as_uint8_gray(image_path)
    segmented, mask = segment_breast_region(gray, method=seg_method)
    used_img = segmented if input_kind == "segmentado" else gray

    img_pil = Image.fromarray(used_img.astype(np.uint8)).convert("RGB")
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(img_pil).unsqueeze(0).to(device)
    x.requires_grad_(True)

    saved = {"activation": None}
    target_layer = model.layer4[-1] if checkpoint["model_name"].startswith("resnet") else model.features[-1]
    def forward_hook(m, i, o):
        saved["activation"] = o
        if hasattr(o, "retain_grad"): o.retain_grad()
    handle = target_layer.register_forward_hook(forward_hook)


    try:
        out = model(x)
        pred_idx = int(out.argmax(dim=1).item())
        confidence = float(torch.softmax(out, dim=1)[0, pred_idx].item())
        model.zero_grad()
        out[0, pred_idx].backward(retain_graph=True)

        act, grad = saved["activation"].detach(), saved["activation"].grad.detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1).squeeze()).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min()) if cam.max() > cam.min() else np.zeros_like(cam)

        heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (used_img.shape[1], used_img.shape[0]))), cv2.COLORMAP_JET)
        overlay_rgb = cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(used_img, cv2.COLOR_GRAY2BGR), 0.65, heatmap, 0.35, 0), cv2.COLOR_BGR2RGB)

        pred_label = (LABEL_NAMES_4 if checkpoint["num_classes"] == 4 else LABEL_NAMES_BIN)[pred_idx]
        suffix = f"_{safe_stem(Path(seg_method))}" if input_kind == "segmentado" and seg_method else ""
        out_path = RESULTS_DIR / f"gradcam_{checkpoint['model_name']}_{task}_{input_kind}{suffix}_{safe_stem(Path(image_path))}.png"
        Image.fromarray(overlay_rgb).save(out_path)
        log(f"[OK] Predicao: {pred_label} ({confidence:.2f}) | Grad-CAM salvo.")
        return pred_label, confidence, used_img, overlay_rgb, out_path
    finally:
        handle.remove()

"""
------------------------------------
- Interface gráfica com Tkinter
------------------------------------
"""

UI = {"bg": "#EEF6F8", "card": "#FFFFFF", "text": "#12313A", "muted": "#5D7680", "primary": "#0F7C90", "primary_dark": "#0B5F6E", "success": "#138A5B", "warning": "#E28A10", "danger": "#C24141", "border": "#C9DDE3", "canvas": "#0B1F27"}

def apply_modern_theme(root):
    root.configure(bg=UI["bg"])
    style = ttk.Style(root)
    try: style.theme_use("clam")
    except tk.TclError: pass
    font = ("Segoe UI", 10)
    root.option_add("*Font", font)
    style.configure("TFrame", background=UI["bg"])
    style.configure("Card.TFrame", background=UI["card"])
    style.configure("Header.TFrame", background=UI["primary_dark"])
    style.configure("TLabel", background=UI["bg"], foreground=UI["text"], font=font)
    style.configure("Card.TLabel", background=UI["card"], foreground=UI["text"], font=font)
    style.configure("HeaderTitle.TLabel", background=UI["primary_dark"], foreground="white", font=("Segoe UI", 20, "bold"))
    style.configure("HeaderSub.TLabel", background=UI["primary_dark"], foreground="#DDF7FA", font=("Segoe UI", 10))
    style.configure("TLabelframe", background=UI["card"], bordercolor=UI["border"], padding=10)
    style.configure("TLabelframe.Label", background=UI["card"], foreground=UI["text"], font=("Segoe UI", 10, "bold"))
    style.configure("TButton", font=font, padding=(10, 7), background="#E5E7EB", foreground=UI["text"], borderwidth=0)
    style.configure("Primary.TButton", background=UI["primary"], foreground="white")
    style.configure("Success.TButton", background=UI["success"], foreground="white")
    style.configure("Warning.TButton", background=UI["warning"], foreground="white")
    style.configure("Danger.TButton", background=UI["danger"], foreground="white")
    style.configure("TNotebook.Tab", padding=(14, 8), background="#E5E7EB", font=("Segoe UI", 9, "bold"))
    style.map("TNotebook.Tab", background=[("selected", UI["card"])], foreground=[("selected", UI["primary"])])

class ImagePanel(ttk.LabelFrame):
    def __init__(self, parent, title="Imagem"):
        super().__init__(parent, text=title, padding=8)
        self.original_pil = None
        self.zoom, self.fit_to_panel = 1.0, True
        
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(toolbar, text="＋", command=lambda: self._zoom_at(1.25)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="－", command=lambda: self._zoom_at(0.8)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Ajustar", command=self.fit).pack(side=tk.LEFT, padx=2)
        
        self.canvas = tk.Canvas(self, bg=UI["canvas"], cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", lambda e: self._zoom_at(1.15 if e.delta > 0 else 0.85))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<Configure>", lambda e: self.render(center=True) if self.fit_to_panel and self.original_pil else None)

    def set_image(self, arr, info=""):
        self.original_pil = Image.fromarray(arr).convert("RGB") if isinstance(arr, np.ndarray) else arr.convert("RGB")
        self.fit_to_panel, self.zoom = True, 1.0
        self.render(center=True)

    def _current_scale(self):
        if not self.original_pil: return 1.0
        if self.fit_to_panel:
            return max(0.05, min((self.canvas.winfo_width()-24)/self.original_pil.width, (self.canvas.winfo_height()-24)/self.original_pil.height))
        return self.zoom

    def render(self, center=False):
        if not self.original_pil: return
        self.canvas.delete("all")
        scale = self._current_scale()
        new_w, new_h = max(1, int(self.original_pil.width * scale)), max(1, int(self.original_pil.height * scale))
        self.current_tk = ImageTk.PhotoImage(self.original_pil.resize((new_w, new_h), Image.BILINEAR))
        self.canvas.create_image(max(12, int((self.canvas.winfo_width()-new_w)/2)), max(12, int((self.canvas.winfo_height()-new_h)/2)), image=self.current_tk, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, max(self.canvas.winfo_width(), new_w+24), max(self.canvas.winfo_height(), new_h+24)))

    def _zoom_at(self, factor):
        if self.original_pil:
            self.fit_to_panel = False
            self.zoom = max(0.05, min(self._current_scale() * factor, 8.0))
            self.render()

    def fit(self):
        self.fit_to_panel = True
        self.render(center=True)

class MammoApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        apply_modern_theme(self.root)
        self.root.geometry("1400x850")

        self.source_path = tk.StringVar()
        self.model_name = tk.StringVar(value="efficientnet_b0")
        self.task = tk.StringVar(value="binary")
        self.input_kind = tk.StringVar(value="segmentado")
        self.seg_method = tk.StringVar(value="Otsu (Padrão)")
        self.epochs = tk.IntVar(value=DEFAULT_EPOCHS)
        self.batch_size = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        self.lr = tk.DoubleVar(value=DEFAULT_LR)

        self.buttons_to_lock = []
        self.build_layout()
        if Path("LCC").exists(): self.source_path.set(str(Path("LCC").resolve()))

    def build_layout(self):
        header = ttk.Frame(self.root, style="Header.TFrame", padding=14)
        header.pack(fill=tk.X)
        ttk.Label(header, text="MammoClass AI", style="HeaderTitle.TLabel").pack(side=tk.LEFT)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.left = ttk.Frame(main, padding=4)
        self.center = ttk.Frame(main, padding=4)
        main.add(self.left, weight=0)
        main.add(self.center, weight=5)

        self.build_left_controls()
        self.build_center_viewer()

        log_frame = ttk.LabelFrame(self.center, text="Terminal de Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=10, bg="#0F172A", fg="#E5E7EB", font=("Cascadia Mono", 9))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def add_button(self, parent, text, cmd, style="TButton", pady=3):
        btn = ttk.Button(parent, text=text, command=cmd, style=style)
        btn.pack(fill=tk.X, pady=pady)
        self.buttons_to_lock.append(btn)

    def build_left_controls(self):
        tabs = ttk.Notebook(self.left)
        tabs.pack(fill=tk.BOTH, expand=True)
        tab_data = ttk.Frame(tabs, padding=8)
        tab_ai = ttk.Frame(tabs, padding=8)
        tab_test = ttk.Frame(tabs, padding=8)
        tabs.add(tab_data, text="1. Dados")
        tabs.add(tab_ai, text="2. IA")
        tabs.add(tab_test, text="3. Testes")

        # Aba 1
        ds = ttk.LabelFrame(tab_data, text="Dataset LCC")
        ds.pack(fill=tk.X, pady=6)
        ttk.Entry(ds, textvariable=self.source_path).pack(fill=tk.X, padx=6, pady=5)
        ttk.Combobox(ds, textvariable=self.seg_method, values=["Otsu (Padrão)", "Filtro Conexo (Attribute Filtering)"], state="readonly").pack(fill=tk.X, padx=6, pady=3)
        self.add_button(ds, "Testar método em UMA imagem", self.run_preview_seg, style="Warning.TButton")
        self.add_button(ds, "Preparar dataset e segmentar", self.run_prep, style="Success.TButton", pady=8)

        # Aba 2 - IA 
        ai = ttk.LabelFrame(tab_ai, text="Configuração da Rede")
        ai.pack(fill=tk.X, pady=6)
        ttk.Combobox(ai, textvariable=self.model_name, values=["efficientnet_b0", "resnet18"], state="readonly").pack(fill=tk.X, padx=6, pady=3)
        ttk.Combobox(ai, textvariable=self.task, values=["binary", "4classes"], state="readonly").pack(fill=tk.X, padx=6, pady=3)
        ttk.Combobox(ai, textvariable=self.input_kind, values=["original", "segmentado"], state="readonly").pack(fill=tk.X, padx=6, pady=6)
        
        train_frame = ttk.LabelFrame(tab_ai, text="Hiperparâmetros de Treinamento")
        train_frame.pack(fill=tk.X, pady=6)
        
        row = ttk.Frame(train_frame, style="Card.TFrame")
        row.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(row, text="Épocas").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Batch").grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.epochs, width=8).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Spinbox(row, from_=1, to=64, textvariable=self.batch_size, width=8).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=2)
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        
        ttk.Label(train_frame, text="Learning rate (ex: 0.001)").pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Entry(train_frame, textvariable=self.lr).pack(fill=tk.X, padx=6, pady=(2, 6))

        self.add_button(train_frame, "Treinar modelo", self.run_train, style="Warning.TButton")

        # Aba 3 - Testes
        test_cfg = ttk.LabelFrame(tab_test, text="Seleção do Modelo e Segmentação")
        test_cfg.pack(fill=tk.X, pady=6)
        ttk.Combobox(test_cfg, textvariable=self.model_name, values=["efficientnet_b0", "resnet18"], state="readonly").pack(fill=tk.X, padx=6, pady=3)
        ttk.Combobox(test_cfg, textvariable=self.task, values=["binary", "4classes"], state="readonly").pack(fill=tk.X, padx=6, pady=3)
        ttk.Combobox(test_cfg, textvariable=self.input_kind, values=["original", "segmentado"], state="readonly").pack(fill=tk.X, padx=6, pady=3)
        ttk.Combobox(test_cfg, textvariable=self.seg_method, values=["Otsu (Padrão)", "Filtro Conexo (Attribute Filtering)"], state="readonly").pack(fill=tk.X, padx=6, pady=6)

        tt = ttk.LabelFrame(tab_test, text="Testes e Relatórios")
        tt.pack(fill=tk.X, pady=6)
        self.add_button(tt, "Classificar imagem + Grad-CAM", self.run_gradcam, style="Primary.TButton")
        self.add_button(tt, "Avaliar Dataset e Gerar Relatório TXT", self.run_evaluate_and_report, style="Success.TButton", pady=6)

    def build_center_viewer(self):
        view = ttk.PanedWindow(self.center, orient=tk.HORIZONTAL)
        view.pack(fill=tk.BOTH, expand=True)
        self.img_l = ImagePanel(view, "Entrada")
        self.img_r = ImagePanel(view, "Saída")
        view.add(self.img_l, weight=1)
        view.add(self.img_r, weight=1)

    def log(self, msg):
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def run_threaded(self, func):
        def wrapper():
            for b in self.buttons_to_lock: b.config(state=tk.DISABLED)
            try: func()
            except Exception as e: messagebox.showerror("Erro", str(e))
            finally:
                for b in self.buttons_to_lock: b.config(state=tk.NORMAL)
        threading.Thread(target=wrapper, daemon=True).start()

    def run_prep(self):
        self.run_threaded(lambda: prepare_dataset_lcc(Path(self.source_path.get()), self.log, self.seg_method.get()))

    def run_train(self):
        self.run_threaded(lambda: train_selected_model(self.model_name.get(), self.task.get(), self.input_kind.get(), int(self.epochs.get()), int(self.batch_size.get()), float(self.lr.get()), self.log, self.seg_method.get()))

    def run_evaluate_and_report(self):
        def task():
            self.log("\n--- Avaliação e geraçao de relatorio txt ---")
            self.log(f"Testando: {self.model_name.get()} | {self.task.get()} | {self.input_kind.get()} | {self.seg_method.get()}")
            metrics = evaluate_saved_model(
                self.model_name.get(), 
                self.task.get(), 
                self.input_kind.get(), 
                log=self.log, 
                seg_method=self.seg_method.get()
            )
            self.root.after(0, lambda: messagebox.showinfo("Sucesso", "Relatório TXT gerado com sucesso na pasta 'resultados'."))
        self.run_threaded(task)

    def run_preview_seg(self):
        path = filedialog.askopenfilename()
        if not path: return
        gray = read_image_as_uint8_gray(Path(path))
        seg, mask = segment_breast_region(gray, self.seg_method.get())
        self.img_l.set_image(gray)
        self.img_r.set_image(seg)

    def run_gradcam(self):
        path = filedialog.askopenfilename()
        if not path: return
        def task():
            pred, conf, used, overlay, _ = classify_image_with_gradcam(Path(path), self.model_name.get(), self.task.get(), self.input_kind.get(), self.log, self.seg_method.get())
            self.root.after(0, lambda: self.img_l.set_image(used))
            self.root.after(0, lambda: self.img_r.set_image(overlay))
        self.run_threaded(task)


if __name__ == "__main__":
    if check_dependencies_or_show_help():
        ensure_dirs()
        app = tk.Tk()
        MammoApp(app)
        app.mainloop()
# -*- coding: utf-8 -*-
"""
GXX_mamografia_lcc_unico_windows_v7_medical_ui.py

Script unico para o trabalho de Segmentacao e Classificacao de Imagens Mamograficas.
Dataset do grupo: LCC
Redes do grupo: EfficientNet + ResNet

IMPORTANTE:
- Este arquivo foi organizado em secoes, mas tudo fica em um unico .py.
- Nao envie o dataset nem os pesos treinados no ZIP final do Canvas.
- Altere os dados do grupo no bloco abaixo antes da entrega.

Componentes do grupo:
- Nome 1 - Matricula 1 - Curso/Campus
- Nome 2 - Matricula 2 - Curso/Campus
- Nome 3 - Matricula 3 - Curso/Campus
"""

import os
import re
import csv
import json
import time
import shutil
import zipfile
import threading
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# As bibliotecas externas ficam dentro do try para o programa abrir uma
# mensagem amigavel caso o professor/aluno ainda nao tenha instalado algo.
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


# ============================================================
# 1. CONFIGURACOES GERAIS
# ============================================================

APP_TITLE = "MammoClass AI - LCC | Interface Médica | EfficientNet + ResNet | V7"
PROCESSED_DIR = Path("dataset_lcc_processado")
MODELS_DIR = Path("modelos_treinados")
RESULTS_DIR = Path("resultados")
TEMP_DIR = Path("_temp_lcc")

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

# Mapeamento definido no enunciado: D, E, F, G -> BIRADS I, II, III, IV.
CLASS_MAP = {
    "D": {"folder": "D_BIRADS_I", "birads": "BIRADS I", "label_4": 0, "label_bin": 0},
    "E": {"folder": "E_BIRADS_II", "birads": "BIRADS II", "label_4": 1, "label_bin": 0},
    "F": {"folder": "F_BIRADS_III", "birads": "BIRADS III", "label_4": 2, "label_bin": 1},
    "G": {"folder": "G_BIRADS_IV", "birads": "BIRADS IV", "label_4": 3, "label_bin": 1},
}

LABEL_NAMES_4 = ["BIRADS I", "BIRADS II", "BIRADS III", "BIRADS IV"]
LABEL_NAMES_BIN = ["I+II", "III+IV"]
AUGMENT_ANGLES = [-20, -10, 0, 10, 20]

# Para deixar o treino mais leve em notebook comum.
DEFAULT_IMAGE_SIZE = 224
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3


# ============================================================
# 2. UTILITARIOS DE LOG E ARQUIVOS
# ============================================================

def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def safe_stem(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", path.stem)


def get_image_number(filename: str):
    """Extrai o ultimo numero do nome do arquivo para aplicar a regra do multiplo de 4."""
    nums = re.findall(r"\d+", filename)
    if not nums:
        return None
    return int(nums[-1])


def infer_class_from_path(path: Path):
    """Tenta inferir D/E/F/G olhando o nome do arquivo e das pastas acima dele."""
    candidates = [path.name.upper()] + [p.name.upper() for p in path.parents]

    for text in candidates:
        text = text.replace("+", " ").replace("-", "_")
        for cls in CLASS_MAP:
            # Exemplos aceitos: D..., D + left + CC, D_left_CC, D BIRADS etc.
            if re.match(rf"^\s*{cls}(\s|_|\.|$)", text):
                return cls
            if f" {cls} " in f" {text} ":
                return cls
    return None


def find_7zip_executable():
    """Procura 7z.exe no PATH ou nos locais comuns do Windows."""
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
    """Mostra instrucoes claras se faltar alguma biblioteca externa."""
    if THIRD_PARTY_IMPORT_ERROR is None:
        return True

    msg = (
        "Faltam bibliotecas obrigatorias para executar o trabalho.\n\n"
        "Abra o PowerShell na pasta do projeto e rode:\n\n"
        r".\.venv\Scripts\python.exe -m pip install --upgrade pip"
        "\n"
        r".\.venv\Scripts\python.exe -m pip install numpy pillow opencv-python scikit-learn torch torchvision"
        "\n\n"
        "Se nao estiver usando .venv, use:\n"
        "py -m pip install numpy pillow opencv-python scikit-learn torch torchvision\n\n"
        f"Erro original: {THIRD_PARTY_IMPORT_ERROR}"
    )
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dependencias ausentes", msg)
    root.destroy()
    return False


# ============================================================
# 3. LEITURA, NORMALIZACAO E SEGMENTACAO DA MAMA
# ============================================================

def read_image_as_uint8_gray(path: Path) -> np.ndarray:
    """
    Le imagem PNG/TIFF/JPG/BMP, inclusive 16 bits, e retorna escala de cinza uint8.
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3:
        # Converte RGB/RGBA para cinza.
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    arr = arr.astype(np.float32)

    # Normalizacao robusta para lidar com imagens 8/12/16 bits.
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
    """Segmentação padrão usando Otsu e Componentes Conexos."""
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


def segment_region_growing(gray: np.ndarray):
    """Segmentação baseada em Crescimento de Regiões usando Flood Fill."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
    M = cv2.moments(th)
    
    if M["m00"] == 0:
        return gray, th
        
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    h, w = gray.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    cv2.floodFill(blur, flood_mask, (cX, cY), 255, loDiff=5, upDiff=5)
    
    mask = flood_mask[1:-1, 1:-1]
    mask = (mask > 0).astype(np.uint8) * 255
    
    k_size = max(5, int(min(h, w) * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    return segmented, mask


def segment_graph_cut(gray: np.ndarray):
    """Segmentação automática usando GrabCut (Graph Cuts) com Fallback Seguro."""
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape

    # 1. Acha a caixa (Bounding Box) da mama de forma inteligente
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Pega apenas o maior objeto para evitar que a caixa fique numa letra no canto da imagem
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
    else:
        x, y, w, h = 10, 10, W-20, H-20

    # 2. Trava de Segurança Crítica para o GrabCut!
    # Garante que a caixa nunca encoste na borda da imagem (sempre sobra fundo fora dela)
    x = max(2, x - 5)
    y = max(2, y - 5)
    w = min(W - x - 2, w + 10)
    h = min(H - y - 2, h + 10)

    # Se a caixa ficar bizarramente pequena, força um tamanho seguro
    if w < 20 or h < 20:
         x, y, w, h = 2, 2, W - 4, H - 4

    rect = (x, y, w, h)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros(gray.shape, np.uint8)

    # 3. Executa o Grafo. Se a imagem estiver muito ruim e quebrar a matemática, 
    # o 'except' intercepta a queda e usa o Otsu como salva-vidas para não crashar o Dataset.
    try:
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return segment_otsu(gray)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    final_mask = mask2 * 255

    k_size = max(5, int(min(H, W) * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    segmented = cv2.bitwise_and(gray, gray, mask=final_mask)
    return segmented, final_mask


def segment_morphological_reconstruction(gray: np.ndarray):
    """Segmentação por Filtro Conexo simulando Max-Tree (Abertura por Reconstrução)."""
    k_size = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    marker = cv2.erode(gray, kernel)

    mask_img = gray.copy()
    reconstructed = marker.copy()
    recon_kernel = np.ones((3, 3), np.uint8)

    while True:
        dilated = cv2.dilate(reconstructed, recon_kernel)
        proposed = cv2.min(dilated, mask_img)
        if np.array_equal(reconstructed, proposed):
            break
        reconstructed = proposed

    _, th = cv2.threshold(reconstructed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))
        final_mask = (comp_labels == largest_label).astype(np.uint8) * 255
    else:
        final_mask = th

    segmented = cv2.bitwise_and(gray, gray, mask=final_mask)
    return segmented, final_mask


def segment_breast_region(gray: np.ndarray, method="Otsu (Padrão)"):
    """Roteador para os métodos de segmentação."""
    if method == "Region Growing":
        return segment_region_growing(gray)
    elif method == "Graph Cuts (GrabCut)":
        return segment_graph_cut(gray)
    elif method == "Filtro Conexo (Max-Tree)":
        return segment_morphological_reconstruction(gray)
    else:
        return segment_otsu(gray)


def save_gray_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


# ============================================================
# 4. PREPARACAO DO DATASET LCC NO WINDOWS
# ============================================================

def extract_lcc_source(source_path: Path, log=print) -> Path:
    """
    Aceita:
    - LCC.zip contendo arquivos .rar; ou
    - pasta ja extraida manualmente pelo usuario.

    No Windows, para extrair .rar automaticamente, instale o 7-Zip.
    """
    source_path = Path(source_path)

    if source_path.is_dir():
        log(f"[OK] Usando pasta ja extraida: {source_path}")
        return source_path

    if source_path.suffix.lower() != ".zip":
        raise ValueError("Selecione o LCC.zip ou uma pasta ja extraida do LCC.")

    seven_zip = find_7zip_executable()
    if seven_zip is None:
        raise RuntimeError(
            "Nao encontrei o 7-Zip no Windows.\n"
            "Instale o 7-Zip ou extraia manualmente o LCC.zip e os .rar.\n"
            "Depois selecione a pasta extraida no aplicativo."
        )

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    zip_out = TEMP_DIR / "zip_extraido"
    zip_out.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Extraindo ZIP: {source_path}")
    with zipfile.ZipFile(source_path, "r") as z:
        z.extractall(zip_out)

    rar_files = [p for p in zip_out.rglob("*.rar") if "__MACOSX" not in str(p)]
    if not rar_files:
        log("[AVISO] Nenhum .rar encontrado. Vou tentar usar o conteudo extraido do ZIP.")
        return zip_out

    rar_out = TEMP_DIR / "rar_extraido"
    rar_out.mkdir(parents=True, exist_ok=True)

    for rar in rar_files:
        cls = infer_class_from_path(rar)
        folder_name = CLASS_MAP[cls]["folder"] if cls else safe_stem(rar)
        class_out = rar_out / folder_name
        class_out.mkdir(parents=True, exist_ok=True)
        log(f"[INFO] Extraindo RAR: {rar.name}")
        cmd = [seven_zip, "x", "-y", str(rar), f"-o{class_out}"]
        subprocess.run(cmd, check=True)

    return rar_out


def prepare_dataset_lcc(source_path: Path, log=print, seg_method="Otsu (Padrão)"):
    """
    Cria duas versoes locais:
    - dataset_lcc_processado/original/train|test/...
    - dataset_lcc_processado/segmentado/train|test/...

    A divisao segue a regra do enunciado:
    numero multiplo de 4 -> teste; demais -> treino.
    """
    ensure_dirs()

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    extracted_root = extract_lcc_source(source_path, log=log)

    manifest = []
    image_paths = [p for p in extracted_root.rglob("*") if p.is_file() and is_image_file(p)]
    image_paths = [p for p in image_paths if "__MACOSX" not in str(p)]

    if not image_paths:
        raise RuntimeError("Nenhuma imagem encontrada. Verifique se o LCC foi extraido corretamente.")

    log(f"[INFO] Imagens encontradas: {len(image_paths)}")

    for idx, img_path in enumerate(image_paths, start=1):
        cls = infer_class_from_path(img_path)
        if cls is None:
            log(f"[AVISO] Classe D/E/F/G nao identificada: {img_path}")
            continue

        img_num = get_image_number(img_path.name)
        if img_num is None:
            log(f"[AVISO] Numero da imagem nao identificado: {img_path.name}")
            continue

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
    log(f"[OK] Manifesto: {manifest_path}")
    log_dataset_summary(manifest, log=log)
    return manifest


def log_dataset_summary(manifest, log=print):
    summary = {}
    for row in manifest:
        key = (row["split"], row["class_letter"], row["birads"])
        summary[key] = summary.get(key, 0) + 1

    log("\n========== RESUMO ==========")
    for split in ["train", "test"]:
        log(split.upper())
        for cls in ["D", "E", "F", "G"]:
            birads = CLASS_MAP[cls]["birads"]
            log(f"  {cls} - {birads}: {summary.get((split, cls, birads), 0)}")
    log("============================\n")


# ============================================================
# 5. DATASET PYTORCH COM AUMENTO DE DADOS
# ============================================================

class MammographyDataset(Dataset):
    def __init__(self, root_dir: Path, task="4classes", train=True, image_size=224):
        self.root_dir = Path(root_dir)
        self.task = task
        self.train = train
        self.image_size = image_size
        self.samples = []

        if task not in {"4classes", "binary"}:
            raise ValueError("task deve ser '4classes' ou 'binary'.")

        for cls in ["D", "E", "F", "G"]:
            folder = self.root_dir / CLASS_MAP[cls]["folder"]
            if not folder.exists():
                continue
            label = CLASS_MAP[cls]["label_4"] if task == "4classes" else CLASS_MAP[cls]["label_bin"]
            for p in folder.rglob("*.png"):
                self.samples.append((p, label, cls))

        if not self.samples:
            raise RuntimeError(f"Nenhuma amostra encontrada em: {root_dir}")

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        if self.train:
            return len(self.samples) * len(AUGMENT_ANGLES)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.train:
            sample_idx = idx // len(AUGMENT_ANGLES)
            angle_idx = idx % len(AUGMENT_ANGLES)
            angle = AUGMENT_ANGLES[angle_idx]
        else:
            sample_idx = idx
            angle = 0

        path, label, cls = self.samples[sample_idx]
        img = Image.open(path).convert("L")
        img = img.convert("RGB")

        if self.train:
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        x = self.tf(img)
        return x, torch.tensor(label, dtype=torch.long), str(path)


# ============================================================
# 6. MODELOS: EFFICIENTNET E RESNET COM TRANSFER LEARNING
# ============================================================

def build_model(model_name="efficientnet_b0", num_classes=4, pretrained=True):
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        for p in model.features.parameters():
            p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        target_layer_name = "features"

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer_name = "layer4"

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer_name = "layer4"

    else:
        raise ValueError("Modelo invalido. Use: efficientnet_b0, resnet18 ou resnet50.")

    return model, target_layer_name


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_file_name(model_name, task, input_kind):
    return MODELS_DIR / f"{model_name}_{task}_{input_kind}.pt"


# ============================================================
# 7. TREINAMENTO E AVALIACAO
# ============================================================

def train_selected_model(model_name, task, input_kind, epochs=5, batch_size=8, lr=1e-3, log=print):
    ensure_dirs()

    input_root = PROCESSED_DIR / input_kind
    train_root = input_root / "train"
    test_root = input_root / "test"

    if not train_root.exists() or not test_root.exists():
        raise RuntimeError("Dataset ainda nao preparado. Primeiro execute 'Preparar dataset'.")

    num_classes = 4 if task == "4classes" else 2
    device = get_device()

    log(f"[INFO] Dispositivo: {device}")
    log(f"[INFO] Modelo: {model_name} | Tarefa: {task} | Entrada: {input_kind}")

    train_ds = MammographyDataset(train_root, task=task, train=True, image_size=DEFAULT_IMAGE_SIZE)
    test_ds = MammographyDataset(test_root, task=task, train=False, image_size=DEFAULT_IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model, _ = build_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        metrics = evaluate_in_memory(model, test_loader, num_classes, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": metrics["accuracy"],
            "test_f1": metrics["f1"],
        }
        history.append(row)

        log(
            f"[EPOCA {epoch}/{epochs}] "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}"
        )

    elapsed = time.time() - start_time

    ckpt_path = model_file_name(model_name, task, input_kind)
    checkpoint = {
        "model_name": model_name,
        "task": task,
        "input_kind": input_kind,
        "num_classes": num_classes,
        "state_dict": model.state_dict(),
        "image_size": DEFAULT_IMAGE_SIZE,
        "elapsed_train_seconds": elapsed,
        "history": history,
    }
    torch.save(checkpoint, ckpt_path)

    history_path = RESULTS_DIR / f"historico_{model_name}_{task}_{input_kind}.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    log(f"\n[OK] Modelo salvo em: {ckpt_path}")
    log(f"[OK] Historico salvo em: {history_path}")
    log(f"[OK] Tempo de treinamento: {elapsed:.2f} s")

    final_metrics = evaluate_saved_model(model_name, task, input_kind, log=log)
    return final_metrics


def evaluate_in_memory(model, loader, num_classes, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())

    return compute_metrics(y_true, y_pred, num_classes)


def compute_metrics(y_true, y_pred, num_classes):
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    if num_classes == 2:
        # Classe positiva = III+IV.
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    else:
        recalls = []
        specs = []
        for i in labels:
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        sensitivity = float(np.mean(recalls))
        specificity = float(np.mean(specs))
        precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    return {
        "confusion_matrix": cm,
        "accuracy": float(acc),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def load_checkpoint_model(model_name, task, input_kind):
    ckpt_path = model_file_name(model_name, task, input_kind)
    if not ckpt_path.exists():
        raise RuntimeError(f"Modelo nao encontrado: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model, _ = build_model(checkpoint["model_name"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def evaluate_saved_model(model_name, task, input_kind, log=print):
    device = get_device()
    model, checkpoint = load_checkpoint_model(model_name, task, input_kind)
    model = model.to(device)

    test_root = PROCESSED_DIR / input_kind / "test"
    num_classes = checkpoint["num_classes"]
    test_ds = MammographyDataset(test_root, task=task, train=False, image_size=checkpoint.get("image_size", 224))
    test_loader = DataLoader(test_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=0)

    start = time.time()
    metrics = evaluate_in_memory(model, test_loader, num_classes, device)
    elapsed = time.time() - start

    names = LABEL_NAMES_4 if num_classes == 4 else LABEL_NAMES_BIN
    result_path = RESULTS_DIR / f"metricas_{model_name}_{task}_{input_kind}.txt"

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(format_metrics(metrics, names, elapsed))

    log("\n" + format_metrics(metrics, names, elapsed))
    log(f"[OK] Metricas salvas em: {result_path}")
    return metrics


def format_metrics(metrics, label_names, elapsed=None):
    lines = []
    lines.append("========== RESULTADOS ==========")
    if elapsed is not None:
        lines.append(f"Tempo de classificacao/avaliacao: {elapsed:.4f} s")
    lines.append(f"Acuracia: {metrics['accuracy']:.4f}")
    lines.append(f"Precisao: {metrics['precision']:.4f}")
    lines.append(f"Sensibilidade: {metrics['sensitivity']:.4f}")
    lines.append(f"Especificidade: {metrics['specificity']:.4f}")
    lines.append(f"F1-score: {metrics['f1']:.4f}")
    lines.append("\nMatriz de confusao:")
    lines.append("Classes: " + " | ".join(label_names))
    lines.append(str(metrics["confusion_matrix"]))
    lines.append("===============================")
    return "\n".join(lines)


# ============================================================
# 8. CLASSIFICACAO DE UMA IMAGEM E GRAD-CAM
# ============================================================

def preprocess_single_image(gray_or_segmented: np.ndarray, image_size=224):
    img = Image.fromarray(gray_or_segmented.astype(np.uint8)).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


def get_target_layer(model, model_name):
    model_name = model_name.lower()
    if model_name.startswith("resnet"):
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError("Camada alvo Grad-CAM nao definida para este modelo.")


def classify_image_with_gradcam(image_path: Path, model_name, task, input_kind, log=print, seg_method="Otsu (Padrão)"):
    """
    Classifica uma imagem individual e gera Grad-CAM.

    Ajuste importante desta versao:
    - Como o projeto usa transferencia de aprendizado e congela as camadas convolucionais,
      o tensor de entrada precisa ter requires_grad=True para o Grad-CAM conseguir calcular
      gradientes na ultima camada convolucional.
    - A captura do gradiente foi feita de forma mais robusta usando retain_grad() na ativacao,
      evitando o erro "list index out of range" quando o backward hook nao retorna gradientes.
    """
    device = get_device()
    model, checkpoint = load_checkpoint_model(model_name, task, input_kind)
    model = model.to(device)
    model.eval()

    gray = read_image_as_uint8_gray(image_path)
    segmented, mask = segment_breast_region(gray, method=seg_method)
    used_img = segmented if input_kind == "segmentado" else gray

    x = preprocess_single_image(used_img, image_size=checkpoint.get("image_size", 224)).to(device)
    x.requires_grad_(True)

    saved = {"activation": None}
    target_layer = get_target_layer(model, checkpoint["model_name"])

    def forward_hook(module, inp, out):
        saved["activation"] = out
        # Necessario para acessar .grad do tensor de ativacao depois do backward.
        if hasattr(out, "retain_grad"):
            out.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        out = model(x)
        probs = torch.softmax(out, dim=1)
        pred_idx = int(out.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())

        model.zero_grad(set_to_none=True)
        score = out[0, pred_idx]
        score.backward(retain_graph=True)

        activation = saved.get("activation")
        if activation is None:
            raise RuntimeError(
                "Nao foi possivel capturar a ativacao da camada alvo do Grad-CAM. "
                "Verifique a rede selecionada."
            )

        gradient = activation.grad
        if gradient is None:
            raise RuntimeError(
                "Nao foi possivel calcular gradientes para o Grad-CAM. "
                "Isso geralmente acontece quando a entrada nao permite gradiente. "
                "Use esta versao corrigida do script."
            )

        act = activation.detach()
        grad = gradient.detach()

        # Esperado: [batch, canais, altura, largura].
        if act.ndim != 4 or grad.ndim != 4:
            raise RuntimeError(
                f"Formato inesperado para Grad-CAM. Ativacao={tuple(act.shape)}, gradiente={tuple(grad.shape)}"
            )

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, (used_img.shape[1], used_img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        base = cv2.cvtColor(used_img, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base, 0.65, heatmap, 0.35, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        label_names = LABEL_NAMES_4 if checkpoint["num_classes"] == 4 else LABEL_NAMES_BIN
        if pred_idx >= len(label_names):
            pred_label = f"classe_{pred_idx}"
        else:
            pred_label = label_names[pred_idx]

        out_path = RESULTS_DIR / f"gradcam_{checkpoint['model_name']}_{task}_{input_kind}_{safe_stem(Path(image_path))}.png"
        Image.fromarray(overlay_rgb).save(out_path)

        log(f"[OK] Predicao: {pred_label} | confianca={confidence:.4f}")
        log(f"[OK] Grad-CAM salvo em: {out_path}")

        return pred_label, confidence, used_img, overlay_rgb, out_path

    finally:
        handle.remove()




# ============================================================
# 9. INTERFACE GRAFICA TKINTER - VERSAO MODERNA E DIDATICA
# ============================================================

UI = {
    "bg": "#EEF6F8",          # fundo azul-esverdeado claro, estilo painel clínico
    "card": "#FFFFFF",
    "card2": "#F6FBFC",
    "text": "#12313A",        # azul-petróleo escuro
    "muted": "#5D7680",
    "primary": "#0F7C90",     # teal médico
    "primary_dark": "#0B5F6E",
    "success": "#138A5B",     # verde clínico
    "warning": "#E28A10",
    "danger": "#C24141",
    "border": "#C9DDE3",
    "canvas": "#0B1F27",      # visor escuro para mamografia
}


def apply_modern_theme(root):
    """Aplica um tema visual mais atual sem depender de bibliotecas externas."""
    root.configure(bg=UI["bg"])
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    default_font = ("Segoe UI", 10)
    title_font = ("Segoe UI", 18, "bold")
    section_font = ("Segoe UI", 10, "bold")
    mono_font = ("Cascadia Mono", 9)

    root.option_add("*Font", default_font)

    style.configure("TFrame", background=UI["bg"])
    style.configure("Card.TFrame", background=UI["card"], relief="flat")
    style.configure("Panel.TFrame", background=UI["card"])
    style.configure("Header.TFrame", background=UI["primary_dark"])

    style.configure("TLabel", background=UI["bg"], foreground=UI["text"], font=default_font)
    style.configure("Card.TLabel", background=UI["card"], foreground=UI["text"], font=default_font)
    style.configure("Muted.TLabel", background=UI["card"], foreground=UI["muted"], font=("Segoe UI", 9))
    style.configure("HeaderTitle.TLabel", background=UI["primary_dark"], foreground="white", font=("Segoe UI", 20, "bold"))
    style.configure("HeaderSub.TLabel", background=UI["primary_dark"], foreground="#DDF7FA", font=("Segoe UI", 10))
    style.configure("Step.TLabel", background=UI["card"], foreground=UI["primary"], font=("Segoe UI", 10, "bold"))
    style.configure("Status.TLabel", background=UI["card"], foreground=UI["text"], font=("Segoe UI", 11, "bold"))

    style.configure("TLabelframe", background=UI["card"], bordercolor=UI["border"], relief="solid", padding=10)
    style.configure("TLabelframe.Label", background=UI["card"], foreground=UI["text"], font=section_font)

    style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7), background="#E5E7EB", foreground=UI["text"], borderwidth=0)
    style.map("TButton", background=[("active", "#CBD5E1"), ("disabled", "#E5E7EB")])

    style.configure("Primary.TButton", background=UI["primary"], foreground="white", padding=(10, 8), borderwidth=0)
    style.map("Primary.TButton", background=[("active", UI["primary_dark"]), ("disabled", "#93C5FD")], foreground=[("disabled", "#EEF2FF")])

    style.configure("Success.TButton", background=UI["success"], foreground="white", padding=(10, 8), borderwidth=0)
    style.map("Success.TButton", background=[("active", "#15803D"), ("disabled", "#86EFAC")])

    style.configure("Warning.TButton", background=UI["warning"], foreground="white", padding=(10, 8), borderwidth=0)
    style.map("Warning.TButton", background=[("active", "#D97706"), ("disabled", "#FCD34D")])

    style.configure("Danger.TButton", background=UI["danger"], foreground="white", padding=(10, 8), borderwidth=0)
    style.map("Danger.TButton", background=[("active", "#B91C1C"), ("disabled", "#FCA5A5")])

    style.configure("TEntry", fieldbackground="#FFFFFF", bordercolor=UI["border"], padding=6)
    style.configure("TCombobox", fieldbackground="#FFFFFF", bordercolor=UI["border"], padding=5)
    style.configure("TSpinbox", fieldbackground="#FFFFFF", bordercolor=UI["border"], padding=5)

    style.configure("TNotebook", background=UI["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 8), background="#E5E7EB", foreground=UI["text"], font=("Segoe UI", 9, "bold"))
    style.map("TNotebook.Tab", background=[("selected", UI["card"])], foreground=[("selected", UI["primary"])])

    style.configure("Horizontal.TProgressbar", troughcolor="#E5E7EB", background=UI["primary"], bordercolor=UI["border"], lightcolor=UI["primary"], darkcolor=UI["primary"])
    return style

class ImagePanel(ttk.LabelFrame):
    """
    Painel de imagem com zoom moderno:
    - arrastar com o mouse para mover a imagem;
    - roda do mouse para zoom no ponto apontado;
    - duplo clique para alternar entre ajustar e 100%;
    - botao direito para ajustar;
    - sem depender das barras laterais de rolagem.

    Tudo permanece no mesmo arquivo .py, conforme exigido no trabalho.
    """
    def __init__(self, parent, title="Imagem"):
        super().__init__(parent, text=title, padding=8)
        self.title = title
        self.original_pil = None
        self.current_tk = None
        self.zoom = 1.0
        self.fit_to_panel = True
        self._is_dragging = False
        self._drag_start = None

        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=4, pady=(3, 2))

        ttk.Button(toolbar, text="＋ Zoom", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="－ Zoom", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Ajustar", command=self.fit).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="100%", command=self.real_size).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Centralizar", command=self.center_view).pack(side=tk.LEFT, padx=2)

        self.info_var = tk.StringVar(value="Nenhuma imagem carregada")
        ttk.Label(toolbar, textvariable=self.info_var).pack(side=tk.RIGHT, padx=4)

        self.hint_var = tk.StringVar(
            value="Dica: arraste a imagem com o mouse para mover • roda do mouse dá zoom • duplo clique alterna Ajustar/100%"
        )
        ttk.Label(self, textvariable=self.hint_var, foreground=UI["muted"]).pack(fill=tk.X, padx=6, pady=(0, 4))

        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Sem barras de rolagem visiveis: a movimentacao agora e feita arrastando a propria imagem.
        self.canvas = tk.Canvas(canvas_frame, bg=UI["canvas"], highlightthickness=0, bd=0, cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Gestos de navegacao da imagem.
        self.canvas.bind("<MouseWheel>", self.on_mousewheel_zoom)      # Windows
        self.canvas.bind("<Button-4>", self.on_linux_wheel_up)         # Linux, caso alguem rode fora do Windows
        self.canvas.bind("<Button-5>", self.on_linux_wheel_down)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>", lambda event: self.fit())       # botao direito ajusta a imagem
        self.canvas.bind("<Configure>", self.on_resize)

        self.placeholder = self.canvas.create_text(
            360, 210,
            text=(
                "Imagem aparecerá aqui\n\n"
                "Como navegar:\n"
                "• arraste com o mouse para mover\n"
                "• use a roda do mouse para dar zoom\n"
                "• duplo clique alterna Ajustar/100%"
            ),
            fill="#E5E7EB",
            font=("Segoe UI", 11),
            justify=tk.CENTER,
        )

    def set_image(self, arr_or_pil, info=""):
        if isinstance(arr_or_pil, Image.Image):
            img = arr_or_pil.convert("RGB")
        else:
            arr = np.asarray(arr_or_pil)
            if arr.ndim == 2:
                img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
            else:
                img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

        self.original_pil = img
        self.fit_to_panel = True
        self.zoom = 1.0
        self.info_var.set(info or f"{img.width} x {img.height} px")
        self.render(center=True)

    def _current_scale(self):
        if self.original_pil is None:
            return 1.0

        img = self.original_pil
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)

        if self.fit_to_panel:
            scale = min((canvas_w - 24) / img.width, (canvas_h - 24) / img.height)
            scale = max(min(scale, 1.0), 0.05)
        else:
            scale = self.zoom
        return scale

    def render(self, center=False, keep_canvas_point=None, mouse_xy=None):
        """
        Desenha a imagem. Quando keep_canvas_point e mouse_xy sao passados,
        tenta manter o ponto sob o cursor estavel durante o zoom.
        """
        self.canvas.delete("all")

        if self.original_pil is None:
            self.placeholder = self.canvas.create_text(
                360, 210,
                text=(
                    "Imagem aparecerá aqui\n\n"
                    "Arraste para mover • Roda do mouse para zoom • Duplo clique para ajustar/100%"
                ),
                fill="white",
                font=("Segoe UI", 11),
                justify=tk.CENTER,
            )
            return

        img = self.original_pil
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        scale = self._current_scale()

        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        self.current_tk = ImageTk.PhotoImage(resized)

        pad = 12
        # Se a imagem couber no painel, centraliza. Se for maior, ancora no canto com margem.
        x0 = max(pad, int((canvas_w - new_w) / 2)) if new_w < canvas_w else pad
        y0 = max(pad, int((canvas_h - new_h) / 2)) if new_h < canvas_h else pad

        self.canvas.create_image(x0, y0, image=self.current_tk, anchor="nw")
        scroll_w = max(canvas_w, new_w + 2 * pad)
        scroll_h = max(canvas_h, new_h + 2 * pad)
        self.canvas.config(scrollregion=(0, 0, scroll_w, scroll_h))

        self.info_var.set(f"{img.width} x {img.height} px | zoom {scale * 100:.0f}%")

        if center:
            self.center_view()
        elif keep_canvas_point is not None and mouse_xy is not None:
            # Mantem aproximadamente o mesmo ponto da imagem sob o mouse apos o zoom.
            old_x, old_y = keep_canvas_point
            mx, my = mouse_xy
            new_x = old_x * (self._current_scale() / max(self._old_scale_for_zoom, 1e-6))
            new_y = old_y * (self._current_scale() / max(self._old_scale_for_zoom, 1e-6))
            self._move_view_to_canvas_point(new_x, new_y, mx, my)

    def _move_view_to_canvas_point(self, canvas_x, canvas_y, mouse_x, mouse_y):
        bbox = self.canvas.bbox("all")
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        view_w = max(self.canvas.winfo_width(), 1)
        view_h = max(self.canvas.winfo_height(), 1)

        target_x = max(0, min(canvas_x - mouse_x, width - view_w))
        target_y = max(0, min(canvas_y - mouse_y, height - view_h))

        if width > view_w:
            self.canvas.xview_moveto(target_x / width)
        if height > view_h:
            self.canvas.yview_moveto(target_y / height)

    def center_view(self):
        bbox = self.canvas.bbox("all")
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        view_w = max(self.canvas.winfo_width(), 1)
        view_h = max(self.canvas.winfo_height(), 1)
        if width > view_w:
            self.canvas.xview_moveto(max((width - view_w) / 2 / width, 0))
        else:
            self.canvas.xview_moveto(0)
        if height > view_h:
            self.canvas.yview_moveto(max((height - view_h) / 2 / height, 0))
        else:
            self.canvas.yview_moveto(0)

    def _zoom_at(self, factor, event=None):
        if self.original_pil is None:
            return

        self.fit_to_panel = False
        old_scale = self._current_scale()
        self._old_scale_for_zoom = old_scale

        if event is not None:
            # Ponto atual do canvas antes de redesenhar.
            keep_point = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            mouse_xy = (event.x, event.y)
        else:
            keep_point = None
            mouse_xy = None

        self.zoom = max(0.05, min(old_scale * factor, 8.0))
        self.render(keep_canvas_point=keep_point, mouse_xy=mouse_xy)

    def zoom_in(self):
        self._zoom_at(1.25)

    def zoom_out(self):
        self._zoom_at(1 / 1.25)

    def fit(self):
        if self.original_pil is None:
            return
        self.fit_to_panel = True
        self.render(center=True)

    def real_size(self):
        if self.original_pil is None:
            return
        self.fit_to_panel = False
        self.zoom = 1.0
        self.render(center=True)

    def on_mousewheel_zoom(self, event):
        if self.original_pil is None:
            return
        # Na V6, a roda sempre da zoom. Para mover, basta arrastar a imagem.
        if event.delta > 0:
            self._zoom_at(1.15, event)
        else:
            self._zoom_at(1 / 1.15, event)

    def on_linux_wheel_up(self, event):
        self._zoom_at(1.15, event)

    def on_linux_wheel_down(self, event):
        self._zoom_at(1 / 1.15, event)

    def on_drag_start(self, event):
        if self.original_pil is None:
            return
        self._is_dragging = True
        self._drag_start = (event.x, event.y)
        self.canvas.config(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def on_drag_move(self, event):
        if self.original_pil is None or not self._is_dragging:
            return
        # gain=1 deixa o movimento mais natural, sem pular demais.
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_drag_end(self, event):
        self._is_dragging = False
        self.canvas.config(cursor="hand2")

    def on_double_click(self, event):
        if self.original_pil is None:
            return
        # Duplo clique alterna entre Ajustar e 100%, que e o comportamento mais simples para apresentar.
        if self.fit_to_panel:
            self.real_size()
        else:
            self.fit()

    def on_resize(self, event):
        if self.fit_to_panel and self.original_pil is not None:
            self.render(center=True)


class MammoApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE + " - painel clínico didático")
        apply_modern_theme(self.root)
        self.root.geometry("1500x920")
        self.root.minsize(1220, 780)

        self.source_path = tk.StringVar(value="")
        self.model_name = tk.StringVar(value="efficientnet_b0")
        self.task = tk.StringVar(value="4classes")
        self.input_kind = tk.StringVar(value="segmentado")
        self.seg_method = tk.StringVar(value="Otsu (Padrão)")
        self.epochs = tk.IntVar(value=DEFAULT_EPOCHS)
        self.batch_size = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        self.lr = tk.DoubleVar(value=DEFAULT_LR)

        self.status_var = tk.StringVar(value="Pronto. Selecione a pasta LCC já extraída para iniciar o fluxo.")
        self.next_step_var = tk.StringVar(value="Próximo passo: selecionar a pasta LCC e preparar o dataset processado.")
        self.dataset_summary_var = tk.StringVar(value="Dataset ainda nao preparado.")
        self.device_var = tk.StringVar(value=f"Dispositivo detectado: {get_device()}")

        self.buttons_to_lock = []
        self.build_layout()
        self.set_help_initial()
        self.update_dataset_status()
        self.auto_detect_resources()

    # --------------------------------------------------------
    # Layout principal
    # --------------------------------------------------------
    def build_layout(self):
        # Topo visual, para o programa parecer um aplicativo de fato e não uma janela antiga.
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 14))
        header.pack(fill=tk.X)

        left_title = ttk.Frame(header, style="Header.TFrame")
        left_title.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left_title, text="MammoClass AI", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            left_title,
            text="Painel didático para segmentação mamográfica, classificação BIRADS e Grad-CAM",
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        badges = ttk.Frame(header, style="Header.TFrame")
        badges.pack(side=tk.RIGHT)
        ttk.Label(badges, text="Dataset: LCC extraído", style="HeaderSub.TLabel").pack(anchor="e")
        ttk.Label(badges, text="Redes: EfficientNet-B0 / ResNet", style="HeaderSub.TLabel").pack(anchor="e")
        ttk.Label(badges, text="Uso: educacional / PAI", style="HeaderSub.TLabel").pack(anchor="e")

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.left = ttk.Frame(main, padding=4)
        self.center = ttk.Frame(main, padding=4)
        self.right = ttk.Frame(main, padding=4)

        main.add(self.left, weight=0)
        main.add(self.center, weight=5)
        main.add(self.right, weight=2)

        self.build_left_controls()
        self.build_center_viewer()
        self.build_right_explanations()
        self.build_bottom_log()

    def add_button(self, parent, text, command, pady=3, style="TButton"):
        btn = ttk.Button(parent, text=text, command=command, style=style)
        btn.pack(fill=tk.X, pady=pady)
        self.buttons_to_lock.append(btn)
        return btn

    def build_left_controls(self):
        self.control_tabs = ttk.Notebook(self.left)
        self.control_tabs.pack(fill=tk.BOTH, expand=True)

        tab_data = ttk.Frame(self.control_tabs, padding=8)
        tab_ai = ttk.Frame(self.control_tabs, padding=8)
        tab_test = ttk.Frame(self.control_tabs, padding=8)
        tab_help = ttk.Frame(self.control_tabs, padding=8)

        self.control_tabs.add(tab_data, text="1. Dados")
        self.control_tabs.add(tab_ai, text="2. IA")
        self.control_tabs.add(tab_test, text="3. Testes")
        self.control_tabs.add(tab_help, text="4. Entrega")

        # Aba 1 - dados
        ds_frame = ttk.LabelFrame(tab_data, text="Dataset LCC já extraído")
        ds_frame.pack(fill=tk.X, pady=6)
        
        ttk.Label(
            ds_frame,
            text="Selecione a pasta LCC que contém as subpastas D + left + CC, E + left + CC, F + left + CC e G + left + CC.",
            style="Card.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=6, pady=(6, 4))
        
        ttk.Label(ds_frame, text="Pasta selecionada:", style="Card.TLabel").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Entry(ds_frame, textvariable=self.source_path, width=48).pack(fill=tk.X, padx=6, pady=5)

        # === MENU DE SEGMENTAÇÃO ADICIONADO AQUI, DEPOIS DA PASTA ===
        ttk.Label(ds_frame, text="Método de Segmentação:", style="Card.TLabel").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Combobox(
            ds_frame,
            textvariable=self.seg_method,
            values=["Otsu (Padrão)", "Region Growing", "Graph Cuts (GrabCut)", "Filtro Conexo (Max-Tree)"],
            state="readonly"
        ).pack(fill=tk.X, padx=6, pady=3)

        self.add_button(ds_frame, "Selecionar pasta LCC", self.select_folder, style="Primary.TButton")
        self.add_button(ds_frame, "Preparar dataset e segmentar", self.run_prepare_dataset, pady=8, style="Success.TButton")
        self.add_button(ds_frame, "Ver resumo do dataset", self.show_dataset_summary, pady=2)
        self.add_button(ds_frame, "Abrir pasta processada", self.open_processed_folder, pady=2)

        clinical_note = ttk.LabelFrame(tab_data, text="O que esta etapa faz")
        clinical_note.pack(fill=tk.X, pady=6)
        ttk.Label(
            clinical_note,
            text="O sistema organiza treino/teste, normaliza as imagens, gera a máscara da mama e salva versões original, segmentada e máscara. Nenhum ZIP é aberto nesta versão: use somente a pasta LCC extraída.",
            wraplength=330,
            justify=tk.LEFT,
            style="Card.TLabel",
        ).pack(anchor="w", padx=6, pady=8)

        # Aba 2 - IA
        model_frame = ttk.LabelFrame(tab_ai, text="Configuração do experimento")
        model_frame.pack(fill=tk.X, pady=6)

        ttk.Label(model_frame, text="Rede neural", style="Card.TLabel").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Combobox(
            model_frame,
            textvariable=self.model_name,
            values=["efficientnet_b0", "resnet18", "resnet50"],
            state="readonly",
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(model_frame, text="Classificação", style="Card.TLabel").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Combobox(
            model_frame,
            textvariable=self.task,
            values=["binary", "4classes"],
            state="readonly",
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(model_frame, text="Imagem usada pela rede", style="Card.TLabel").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Combobox(
            model_frame,
            textvariable=self.input_kind,
            values=["original", "segmentado"],
            state="readonly",
        ).pack(fill=tk.X, padx=6, pady=(3, 6))

        train_frame = ttk.LabelFrame(tab_ai, text="Treinamento")
        train_frame.pack(fill=tk.X, pady=6)

        row = ttk.Frame(train_frame, style="Card.TFrame")
        row.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(row, text="Épocas", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(row, text="Batch", style="Card.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Spinbox(row, from_=1, to=50, textvariable=self.epochs, width=8).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Spinbox(row, from_=1, to=64, textvariable=self.batch_size, width=8).grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=2)
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)

        ttk.Label(train_frame, text="Learning rate", style="Card.TLabel").pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Entry(train_frame, textvariable=self.lr).pack(fill=tk.X, padx=6, pady=(2, 6))
        self.add_button(train_frame, "Treinar modelo selecionado", self.run_train, pady=6, style="Warning.TButton")
        self.add_button(train_frame, "Avaliar modelo salvo", self.run_evaluate, pady=2, style="Primary.TButton")

        recipe = ttk.LabelFrame(tab_ai, text="Configurações recomendadas")
        recipe.pack(fill=tk.X, pady=6)
        ttk.Label(
            recipe,
            text="Resultado principal: EfficientNet-B0 + binary + segmentado.\nResultado 4 classes: EfficientNet-B0 + 4classes + original.\nPara teste rápido: 2 épocas. Para resultado final: 10 épocas.",
            wraplength=330,
            justify=tk.LEFT,
            style="Card.TLabel",
        ).pack(anchor="w", padx=6, pady=8)

        # Aba 3 - testes e visualização
        vis_frame = ttk.LabelFrame(tab_test, text="Visualização e explicabilidade")
        vis_frame.pack(fill=tk.X, pady=6)
        self.add_button(vis_frame, "Testar segmentação em uma imagem", self.run_preview_segmentation, pady=6)
        self.add_button(vis_frame, "Classificar imagem + Grad-CAM", self.run_gradcam, pady=2, style="Primary.TButton")
        self.add_button(vis_frame, "Demonstração automática", self.run_auto_demo, pady=2, style="Success.TButton")
        self.add_button(vis_frame, "Comparar original × segmentada", self.run_compare_original_segmented, pady=2)
        self.add_button(vis_frame, "Abrir pasta de resultados", self.open_results_folder, pady=2)
        self.add_button(vis_frame, "Limpar log", self.clear_log, pady=6, style="Danger.TButton")

        usage = ttk.LabelFrame(tab_test, text="Controle da imagem")
        usage.pack(fill=tk.X, pady=6)
        ttk.Label(
            usage,
            text="Arrastar com o botão esquerdo move a mamografia. A roda do mouse aplica zoom no ponto do cursor. Duplo clique alterna Ajustar/100%. Botão direito ajusta a imagem.",
            wraplength=330,
            justify=tk.LEFT,
            style="Card.TLabel",
        ).pack(anchor="w", padx=6, pady=8)

        # Aba 4 - entrega
        obs = ttk.LabelFrame(tab_help, text="Entrega e apresentação")
        obs.pack(fill=tk.X, pady=6)
        ttk.Label(
            obs,
            text="Canvas: envie somente o .py único + relatório. Não envie dataset, imagens, pasta processada nem pesos treinados.\n\nApresentação local: mantenha dataset_lcc_processado, modelos_treinados e resultados na pasta do projeto para demonstrar sem retreinar.",
            foreground=UI["danger"],
            wraplength=330,
            justify=tk.LEFT,
            style="Card.TLabel",
        ).pack(anchor="w", padx=6, pady=8)

        checklist = ttk.LabelFrame(tab_help, text="Checklist rápido")
        checklist.pack(fill=tk.X, pady=6)
        ttk.Label(
            checklist,
            text="1. Dataset preparado.\n2. Modelos treinados.\n3. Métricas salvas.\n4. Grad-CAM funcionando.\n5. Relatório com tabelas e discussão.",
            wraplength=330,
            justify=tk.LEFT,
            style="Card.TLabel",
        ).pack(anchor="w", padx=6, pady=8)

    def build_center_viewer(self):
        status_frame = ttk.LabelFrame(self.center, text="Status do experimento")
        status_frame.pack(fill=tk.X, pady=(0, 8))

        top = ttk.Frame(status_frame, style="Card.TFrame")
        top.pack(fill=tk.X, padx=8, pady=(8, 2))
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w")
        ttk.Label(top, textvariable=self.next_step_var, foreground=UI["primary"], background=UI["card"], wraplength=820).pack(anchor="w", pady=(2, 4))
        ttk.Label(top, textvariable=self.device_var, foreground=UI["muted"], background=UI["card"]).pack(anchor="w")

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=8, pady=(4, 8))

        view_frame = ttk.PanedWindow(self.center, orient=tk.HORIZONTAL)
        view_frame.pack(fill=tk.BOTH, expand=True)

        self.image_left = ImagePanel(view_frame, title="Imagem de entrada")
        self.image_right = ImagePanel(view_frame, title="Resultado / Máscara / Grad-CAM")
        view_frame.add(self.image_left, weight=1)
        view_frame.add(self.image_right, weight=1)

    def build_right_explanations(self):
        self.notebook = ttk.Notebook(self.right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.guide_text = self.create_text_tab("Fluxo clínico")
        self.dataset_text = self.create_text_tab("Dataset LCC")
        self.method_text = self.create_text_tab("Segmentação/IA")
        self.results_text = self.create_text_tab("Métricas")
        self.presentation_text = self.create_text_tab("Apresentação")

    def create_text_tab(self, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        txt = tk.Text(frame, wrap=tk.WORD, height=12, font=("Segoe UI", 10), bg="#FFFFFF", fg=UI["text"], relief="flat", padx=12, pady=12)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(frame, command=txt.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        txt.config(yscrollcommand=scroll.set)
        txt.config(state=tk.DISABLED)
        return txt

    def build_bottom_log(self):
        log_frame = ttk.LabelFrame(self.center, text="Log detalhado do que o programa esta fazendo")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self.log_text = tk.Text(log_frame, height=11, wrap=tk.WORD, font=("Cascadia Mono", 9), bg="#0F172A", fg="#E5E7EB", insertbackground="#E5E7EB", relief="flat", padx=10, pady=10)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scroll.set)

    # --------------------------------------------------------
    # Textos didaticos
    # --------------------------------------------------------
    def set_text(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state=tk.DISABLED)

    def set_help_initial(self):
        self.set_text(self.guide_text,
"""FLUXO CLÍNICO-DIDÁTICO DA APLICAÇÃO

1) Preparação dos dados
   - Selecione somente a pasta LCC já extraída.
   - A pasta deve conter D + left + CC, E + left + CC, F + left + CC e G + left + CC.
   - Clique em Preparar dataset e segmentar.

2) Segmentação mamográfica
   - O sistema normaliza a imagem.
   - Detecta a região da mama.
   - Remove fundo e anotações por máscara.
   - Salva imagem original normalizada, máscara e imagem segmentada.

3) Classificação
   - Binary: BIRADS I+II contra BIRADS III+IV.
   - 4classes: BIRADS I, II, III e IV separadamente.
   - Redes disponíveis: EfficientNet-B0, ResNet18 e ResNet50.

4) Explicabilidade
   - Use Grad-CAM para mostrar visualmente a região que influenciou a decisão da rede.
   - O mapa é qualitativo e não substitui diagnóstico médico.

5) Demonstração rápida
   - A melhor configuração binária dos seus testes foi EfficientNet-B0 + segmentado.
   - Para 4 classes, a melhor foi EfficientNet-B0 + original.
""")

        self.set_text(self.dataset_text,
"""DATASET LCC

LCC significa Left Cranio-Caudal:
- Left: mama esquerda.
- CC: incidência crânio-caudal.

Estrutura esperada nesta versão:
LCC/
  D + left + CC/
  E + left + CC/
  F + left + CC/
  G + left + CC/

Mapeamento das classes:
- D -> BIRADS I
- E -> BIRADS II
- F -> BIRADS III
- G -> BIRADS IV

Classificação binária:
- Classe 0: BIRADS I + BIRADS II
- Classe 1: BIRADS III + BIRADS IV

Classificação em 4 classes:
- Classe 0: BIRADS I
- Classe 1: BIRADS II
- Classe 2: BIRADS III
- Classe 3: BIRADS IV

Regra de divisão:
- Imagem com número múltiplo de 4: teste.
- Demais imagens: treino.
""")

        self.set_text(self.method_text,
"""SEGMENTAÇÃO E MODELO DE IA

Segmentação:
1) Leitura da imagem em tons de cinza.
2) Normalização robusta para 8 bits.
3) Suavização com filtro Gaussiano.
4) Limiarização automática de Otsu.
5) Operações morfológicas para limpeza.
6) Maior componente conexo como região da mama.
7) Aplicação da máscara para remover fundo e anotações.

Redes neurais:
- EfficientNet-B0 com transferência de aprendizado.
- ResNet18/ResNet50 com transferência de aprendizado.
- Camadas convolucionais congeladas.
- Camada final substituída e treinada para a tarefa escolhida.

Aumento de dados:
- Apenas no treinamento.
- Rotações: -20, -10, 0, 10 e 20 graus.

Grad-CAM:
- Mostra regiões que contribuíram para a decisão da rede.
- Usado como apoio visual no relatório e na apresentação.
""")

        self.set_text(self.results_text,
"""MÉTRICAS E RESULTADOS

Para classificação binária, observar:
- Acurácia.
- Precisão.
- Sensibilidade.
- Especificidade.
- F1-score.
- Matriz de confusão.

Para 4 classes, observar:
- Acurácia.
- F1 macro.
- Sensibilidade média.
- Especificidade média.
- Matriz de confusão.

Resultados obtidos pelo grupo:
- Melhor binário: EfficientNet-B0 + segmentado, acurácia 88,14%.
- Melhor 4 classes: EfficientNet-B0 + original, acurácia 69,55%.

Interpretação importante:
A segmentação ajudou na classificação binária, mas as imagens originais tiveram melhor desempenho na tarefa de 4 classes.
""")

        self.set_text(self.presentation_text,
"""ROTEIRO DE APRESENTAÇÃO

1) Abrir a aplicação.
2) Mostrar a aba Dados e explicar o dataset LCC.
3) Abrir uma imagem e demonstrar a segmentação.
4) Mostrar o resumo do dataset: treino/teste por classe.
5) Avaliar o modelo EfficientNet-B0 + binary + segmentado.
6) Mostrar a matriz de confusão e as métricas.
7) Gerar Grad-CAM de uma imagem F ou G.
8) Explicar que Grad-CAM é apoio visual, não laudo médico.
9) Mostrar comparação com 4 classes e discutir a dificuldade maior.
10) Fechar com a comparação original × segmentado.

Entrega no Canvas:
Enviar apenas o arquivo .py único e o relatório. Não enviar dataset, imagens, modelos .pt ou pasta resultados.
""")

    def set_explanation_for_segmentation(self):
        self.notebook.select(2)
        self.set_text(self.method_text,
"""SEGMENTACAO VISUALIZADA

Na tela da esquerda voce ve a imagem original normalizada.
Na tela da direita voce ve o resultado da segmentacao.

O objetivo da segmentacao e deixar para a rede apenas a regiao da mama.
Com isso, fundo preto, letras, marcadores e anotacoes tendem a ser removidos.

Como interpretar:
- Se a mama continua visivel e o fundo foi removido, a segmentacao esta boa.
- Se uma parte grande da mama sumiu, a mascara esta agressiva demais.
- Se sobraram muitas anotacoes, a mascara esta permissiva demais.

Use o zoom para verificar bordas, marcadores e regioes removidas.
""")

    def set_explanation_for_training(self):
        self.notebook.select(2)
        self.set_text(self.method_text,
"""TREINAMENTO EM ANDAMENTO

O programa esta usando transferencia de aprendizado:

1) Carrega uma rede pre-treinada.
2) Congela a parte que ja sabe extrair caracteristicas visuais gerais.
3) Substitui a camada final da rede.
4) Treina essa camada final com as imagens de mamografia do LCC.

Durante o treino, cada imagem de treino gera 5 versoes:
- rotacao -20 graus;
- rotacao -10 graus;
- rotacao 0 grau;
- rotacao 10 graus;
- rotacao 20 graus.

Isso aumenta a quantidade de exemplos e ajuda a reduzir overfitting.
""")

    def set_explanation_for_gradcam(self):
        self.notebook.select(2)
        self.set_text(self.method_text,
"""GRAD-CAM

A imagem da direita mostra um mapa de calor sobre a mamografia.

Interpretacao:
- Regioes mais quentes indicam maior influencia na decisao da rede.
- Regioes frias tiveram menor influencia.

No relatorio, use o Grad-CAM para discutir se a rede parece olhar para a regiao da mama ou se esta sendo influenciada por fundo, bordas ou anotacoes.
""")

    # --------------------------------------------------------
    # Log, status e tarefas
    # --------------------------------------------------------
    def log(self, msg):
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def set_busy(self, busy=True, status=None):
        if status:
            self.status_var.set(status)
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in self.buttons_to_lock:
            btn.config(state=state)

    def run_threaded(self, func, busy_status="Processando..."):
        def wrapper():
            try:
                self.root.after(0, lambda: self.set_busy(True, busy_status))
                func()
            except Exception as e:
                self.log(f"[ERRO] {e}")
                self.root.after(0, lambda: messagebox.showerror("Erro", str(e)))
            finally:
                self.root.after(0, lambda: self.set_busy(False, "Pronto."))
        threading.Thread(target=wrapper, daemon=True).start()

    # --------------------------------------------------------
    # Acoes da interface
    # --------------------------------------------------------
    def select_zip(self):
        path = filedialog.askopenfilename(title="Selecione o LCC.zip", filetypes=[("ZIP", "*.zip"), ("Todos", "*.*")])
        if path:
            self.source_path.set(path)
            self.status_var.set("ZIP selecionado. Agora clique em Preparar dataset e segmentar tudo.")
            self.next_step_var.set("Proximo passo: preparar o dataset. O ZIP sera extraido e as imagens serao organizadas.")

    def select_folder(self):
        path = filedialog.askdirectory(title="Selecione a pasta LCC já extraída")
        if path:
            self.source_path.set(path)
            self.status_var.set("Pasta LCC selecionada. Agora clique em Preparar dataset e segmentar.")
            self.next_step_var.set("Próximo passo: preparar o dataset processado a partir da pasta LCC.")

    def run_prepare_dataset(self):
        def task():
            if not self.source_path.get():
                raise RuntimeError("Selecione a pasta LCC já extraída primeiro.")
            self.log("\n=== PREPARACAO DO DATASET ===")
            self.log("O programa vai organizar treino/teste, gerar imagens originais normalizadas, segmentadas e máscaras.")
            prepare_dataset_lcc(Path(self.source_path.get()), log=self.log, seg_method=self.seg_method.get())
            self.root.after(0, self.update_dataset_status)
            self.root.after(0, lambda: self.next_step_var.set("Proximo passo: testar a segmentacao em uma imagem antes de treinar."))
        self.run_threaded(task, busy_status="Preparando dataset e segmentando imagens...")

    def run_train(self):
        def task():
            self.log("\n=== TREINAMENTO ===")
            self.log(f"Rede: {self.model_name.get()} | Tarefa: {self.task.get()} | Entrada: {self.input_kind.get()}")
            self.root.after(0, self.set_explanation_for_training)
            metrics = train_selected_model(
                model_name=self.model_name.get(),
                task=self.task.get(),
                input_kind=self.input_kind.get(),
                epochs=int(self.epochs.get()),
                batch_size=int(self.batch_size.get()),
                lr=float(self.lr.get()),
                log=self.log,
            )
            self.root.after(0, lambda: self.update_results_tab(metrics))
            self.root.after(0, lambda: self.next_step_var.set("Proximo passo: avaliar o modelo salvo ou gerar Grad-CAM em uma imagem de teste."))
        self.run_threaded(task, busy_status="Treinando modelo. Isso pode demorar...")

    def run_evaluate(self):
        def task():
            self.log("\n=== AVALIACAO DO MODELO SALVO ===")
            metrics = evaluate_saved_model(self.model_name.get(), self.task.get(), self.input_kind.get(), log=self.log)
            self.root.after(0, lambda: self.update_results_tab(metrics))
            self.root.after(0, lambda: self.next_step_var.set("Proximo passo: usar Grad-CAM para explicar uma predicao individual."))
        self.run_threaded(task, busy_status="Avaliando modelo salvo...")

    def run_gradcam(self):
        path = filedialog.askopenfilename(
            title="Selecione uma imagem para classificar com Grad-CAM",
            filetypes=[("Imagens", "*.png;*.tif;*.tiff;*.jpg;*.jpeg;*.bmp"), ("Todos", "*.*")]
        )
        if not path:
            return

        def task():
            self.log("\n=== CLASSIFICACAO INDIVIDUAL + GRAD-CAM ===")
            self.root.after(0, self.set_explanation_for_gradcam)
            pred, conf, used_img, overlay, out_path = classify_image_with_gradcam(
                Path(path), self.model_name.get(), self.task.get(), self.input_kind.get(), log=self.log, seg_method=self.seg_method.get()
            )
            self.root.after(0, lambda: self.image_left.set_image(used_img, "Imagem usada pela rede"))
            self.root.after(0, lambda: self.image_right.set_image(overlay, "Grad-CAM"))
            self.root.after(0, lambda: messagebox.showinfo("Resultado", f"Predicao: {pred}\nConfianca: {conf:.4f}"))
            self.root.after(0, lambda: self.next_step_var.set("Use essa imagem no relatorio para discutir a explicabilidade da rede."))
        self.run_threaded(task, busy_status="Gerando Grad-CAM...")

    def run_preview_segmentation(self):
        path = filedialog.askopenfilename(
            title="Selecione uma imagem para testar segmentacao",
            filetypes=[("Imagens", "*.png;*.tif;*.tiff;*.jpg;*.jpeg;*.bmp"), ("Todos", "*.*")]
        )
        if not path:
            return

        gray = read_image_as_uint8_gray(Path(path))
        segmented, mask = segment_breast_region(gray, method=self.seg_method.get())
        overlay = self.make_mask_overlay(gray, mask)

        self.image_left.set_image(gray, "Original normalizada")
        self.image_right.set_image(overlay, "Mascara sobreposta em verde")
        self.log(f"[OK] Segmentacao testada em: {path}")
        self.log("[INFO] Direita: verde = area mantida pela mascara. Preto/fundo = removido.")
        self.status_var.set("Segmentacao visualizada.")
        self.next_step_var.set("Se a mascara estiver boa, treine primeiro ResNet18 + binary + segmentado com 2 epocas.")
        self.set_explanation_for_segmentation()

    def run_compare_original_segmented(self):
        path = filedialog.askopenfilename(
            title="Selecione uma imagem para comparar original x segmentada",
            filetypes=[("Imagens", "*.png;*.tif;*.tiff;*.jpg;*.jpeg;*.bmp"), ("Todos", "*.*")]
        )
        if not path:
            return
        gray = read_image_as_uint8_gray(Path(path))
        segmented, mask = segment_breast_region(gray, method=self.seg_method.get())
        self.image_left.set_image(gray, "Original normalizada")
        self.image_right.set_image(segmented, "Imagem segmentada")
        self.log(f"[OK] Comparacao original x segmentada: {path}")
        self.status_var.set("Comparacao carregada.")
        self.next_step_var.set("Use o zoom para verificar se anotacoes e fundo foram removidos.")
        self.set_explanation_for_segmentation()

    def open_results_folder(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        try:
            os.startfile(str(RESULTS_DIR.resolve()))
        except Exception as e:
            messagebox.showerror("Erro", f"Nao consegui abrir a pasta de resultados: {e}")

    def auto_detect_resources(self):
        """Detecta automaticamente a pasta LCC e modelos já treinados na pasta do projeto."""
        candidates = [Path("LCC"), Path("lcc")]
        if not self.source_path.get():
            for candidate in candidates:
                if candidate.exists():
                    self.source_path.set(str(candidate.resolve()))
                    self.log(f"[OK] Pasta LCC detectada automaticamente: {candidate.resolve()}")
                    break

        existing_models = sorted(MODELS_DIR.glob("*.pt"))
        if existing_models:
            self.log("[OK] Modelos treinados encontrados localmente:")
            for p in existing_models:
                self.log(f"     - {p}")
            self.next_step_var.set("Modelos ja treinados encontrados. Voce pode avaliar ou usar a Demonstração automática.")
        elif (PROCESSED_DIR / "manifest_lcc.csv").exists():
            self.next_step_var.set("Dataset processado encontrado. Proximo passo: treinar ou avaliar se houver modelo salvo.")

    def find_demo_image(self):
        """Escolhe automaticamente uma imagem de teste para a demonstracao."""
        base = PROCESSED_DIR / self.input_kind.get() / "test"
        if not base.exists():
            return None

        if self.task.get() == "binary":
            preferred_folders = ["F_BIRADS_III", "G_BIRADS_IV", "D_BIRADS_I", "E_BIRADS_II"]
        else:
            preferred_folders = ["D_BIRADS_I", "E_BIRADS_II", "F_BIRADS_III", "G_BIRADS_IV"]

        for folder in preferred_folders:
            folder_path = base / folder
            if folder_path.exists():
                images = sorted([p for p in folder_path.iterdir() if is_image_file(p)])
                if images:
                    return images[0]
        images = sorted([p for p in base.rglob("*") if is_image_file(p)])
        return images[0] if images else None

    def run_auto_demo(self):
        """Executa um fluxo pronto para apresentacao: melhor modelo binario + avaliacao + Grad-CAM."""
        self.model_name.set("efficientnet_b0")
        self.task.set("binary")
        self.input_kind.set("segmentado")

        model_path = model_file_path(self.model_name.get(), self.task.get(), self.input_kind.get())
        if not model_path.exists():
            messagebox.showwarning(
                "Modelo nao encontrado",
                "Nao encontrei o modelo treinado da demonstracao.\n\n"
                "Treine primeiro:\n"
                "Modelo: efficientnet_b0\n"
                "Classificacao: binary\n"
                "Entrada: segmentado\n"
                "Epocas sugeridas: 10"
            )
            return

        demo_image = self.find_demo_image()
        if demo_image is None:
            messagebox.showwarning(
                "Imagem de teste nao encontrada",
                "Nao encontrei imagens em dataset_lcc_processado/segmentado/test.\n"
                "Prepare o dataset antes de usar a demonstracao."
            )
            return

        def task():
            self.log("\n=== DEMONSTRACAO AUTOMATICA ===")
            self.log("Configuracao usada: EfficientNet-B0 | binary | segmentado")
            self.log(f"Imagem escolhida automaticamente: {demo_image}")
            metrics = evaluate_saved_model(self.model_name.get(), self.task.get(), self.input_kind.get(), log=self.log)
            pred, conf, used_img, overlay, out_path = classify_image_with_gradcam(
                demo_image, self.model_name.get(), self.task.get(), self.input_kind.get(), log=self.log
            )
            self.root.after(0, lambda: self.image_left.set_image(used_img, "Imagem usada pela rede"))
            self.root.after(0, lambda: self.image_right.set_image(overlay, "Grad-CAM"))
            self.root.after(0, lambda: self.update_results_tab(metrics))
            self.root.after(0, lambda: self.next_step_var.set("Demonstracao concluida. Use a tela e o Grad-CAM para explicar o modelo."))
            self.root.after(0, lambda: messagebox.showinfo("Demonstracao concluida", f"Predicao: {pred}\nConfianca: {conf:.4f}\nGrad-CAM salvo em:\n{out_path}"))

        self.run_threaded(task, busy_status="Executando demonstracao automatica...")

    def make_mask_overlay(self, gray, mask):
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = base.copy()
        green = np.zeros_like(base)
        green[:, :, 1] = 255
        alpha = 0.35
        keep = mask > 0
        overlay[keep] = cv2.addWeighted(base, 1 - alpha, green, alpha, 0)[keep]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
        return overlay

    def update_results_tab(self, metrics):
        names = LABEL_NAMES_4 if self.task.get() == "4classes" else LABEL_NAMES_BIN
        content = format_metrics(metrics, names)
        content += "\n\nINTERPRETACAO RAPIDA:\n"
        content += "- Acuracia mostra a taxa geral de acertos.\n"
        content += "- Sensibilidade mede quanto o modelo recupera corretamente as classes reais.\n"
        content += "- Especificidade mede quanto o modelo evita falsos positivos.\n"
        content += "- F1-score equilibra precisao e sensibilidade.\n"
        content += "- A matriz de confusao mostra onde o modelo esta confundindo as classes.\n"
        self.set_text(self.results_text, content)
        self.notebook.select(3)

    def update_dataset_status(self):
        manifest_path = PROCESSED_DIR / "manifest_lcc.csv"
        if not manifest_path.exists():
            self.dataset_summary_var.set("Dataset ainda nao preparado.")
            return

        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        summary = {}
        for row in rows:
            key = (row["split"], row["class_letter"], row["birads"])
            summary[key] = summary.get(key, 0) + 1

        lines = ["DATASET PROCESSADO", f"Total de imagens: {len(rows)}", ""]
        for split in ["train", "test"]:
            lines.append(split.upper())
            for cls in ["D", "E", "F", "G"]:
                birads = CLASS_MAP[cls]["birads"]
                lines.append(f"  {cls} - {birads}: {summary.get((split, cls, birads), 0)}")
            lines.append("")
        lines.append(f"Manifesto: {manifest_path}")

        content = "\n".join(lines)
        self.dataset_summary_var.set(content)
        self.set_text(self.dataset_text, content + "\n\nPastas criadas:\n- original/train\n- original/test\n- segmentado/train\n- segmentado/test\n- mascaras/train\n- mascaras/test\n")

    def show_dataset_summary(self):
        self.update_dataset_status()
        self.notebook.select(1)
        messagebox.showinfo("Resumo do dataset", self.dataset_summary_var.get())

    def open_processed_folder(self):
        PROCESSED_DIR.mkdir(exist_ok=True)
        try:
            os.startfile(str(PROCESSED_DIR.resolve()))
        except Exception as e:
            messagebox.showerror("Erro", f"Nao consegui abrir a pasta: {e}")


# ============================================================
# 10. EXECUCAO
# ============================================================

def main():
    if not check_dependencies_or_show_help():
        return
    ensure_dirs()
    root = tk.Tk()
    app = MammoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

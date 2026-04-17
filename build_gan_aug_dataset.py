import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import yaml

from gan_models import UNetGenerator


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def img_ext(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-yaml",
        default="safety.yaml",
        help="用于读取 names 与原始 val 路径；训练图可另由 --train-images-dir 指定",
    )
    parser.add_argument("--gan-ckpt", required=True, help="trained generator .pt")
    parser.add_argument("--out-root", default="yolo_dataset_gan")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--train-images-dir",
        default=None,
        help="可选：对指定目录中的训练图过 GAN（例如自有 yolo_dataset/train/images），"
        "不传则使用 src-yaml 中 path/train/images",
    )
    parser.add_argument(
        "--train-labels-dir",
        default=None,
        help="可选：与训练图同名的 .txt 标签目录；不传则使用 src-yaml 中 path/train/labels",
    )
    args = parser.parse_args()

    src_yaml = Path(args.src_yaml).resolve()
    cfg = load_yaml(src_yaml)
    dataset_root = Path(cfg["path"]).resolve()
    out_root = Path(args.out_root).resolve()

    out_train_img = out_root / "train" / "images"
    out_train_lab = out_root / "train" / "labels"
    out_val_img = out_root / "val" / "images"
    out_val_lab = out_root / "val" / "labels"
    for p in (out_train_img, out_train_lab, out_val_img, out_val_lab):
        p.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gen = UNetGenerator().to(device)
    ckpt = torch.load(args.gan_ckpt, map_location=device)
    state = ckpt.get("generator", ckpt)
    gen.load_state_dict(state, strict=False)
    gen.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    inv = transforms.Compose(
        [
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage(),
        ]
    )

    train_images = (
        Path(args.train_images_dir).resolve()
        if args.train_images_dir
        else dataset_root / "train" / "images"
    )
    train_labels = (
        Path(args.train_labels_dir).resolve()
        if args.train_labels_dir
        else dataset_root / "train" / "labels"
    )
    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"

    # Train: 生成 GAN 增强图，同时保留标签文件名对齐
    with torch.no_grad():
        for p in sorted(train_images.iterdir()):
            if not p.is_file() or not img_ext(p):
                continue
            src = Image.open(p).convert("RGB")
            x = tf(src).unsqueeze(0).to(device)
            y = gen(x)[0].cpu().clamp(-1, 1)
            out_img = inv(y)
            out_name = p.stem + "_gan.jpg"
            out_img.save(out_train_img / out_name, quality=95)
            lab_src = train_labels / f"{p.stem}.txt"
            if lab_src.exists():
                shutil.copy2(lab_src, out_train_lab / f"{p.stem}_gan.txt")

    # Val: 保持原始验证集，直接复制
    for p in sorted(val_images.iterdir()):
        if p.is_file() and img_ext(p):
            shutil.copy2(p, out_val_img / p.name)
    for p in sorted(val_labels.iterdir()):
        if p.is_file() and p.suffix.lower() == ".txt":
            shutil.copy2(p, out_val_lab / p.name)

    out_yaml = {
        "path": str(out_root).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "names": cfg["names"],
    }
    save_yaml(out_yaml, src_yaml.with_name("safety_gan.yaml"))
    print(f"GAN augmented dataset ready: {out_root}")
    print(f"Use dataset yaml: {src_yaml.with_name('safety_gan.yaml')}")


if __name__ == "__main__":
    main()

"""
在「无真实配对」场景下训练论文式 GAN：以清晰图为 B，用 Albumentations 在线合成退化图 A，
再训练 U-Net 生成器 G(A)->B 与 PatchGAN 判别器。

典型用法：在 VOC2028 的 JPEGImages 上预训练生成器，再在自有 YOLO 训练集上过一遍生成器，
验证集仍用 safety.yaml 指向的原始 val（见 build_gan_aug_dataset.py）。
"""

import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from gan_models import UNetGenerator, PatchDiscriminator


def img_ext(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SyntheticWeatherDataset(Dataset):
    """清晰图 B + 在线合成退化图 A（无需配对数据集）。"""

    def __init__(self, image_dir: Path, size: int = 256):
        self.files = sorted([p for p in image_dir.iterdir() if p.is_file() and img_ext(p)])
        if not self.files:
            raise FileNotFoundError(f"未找到图片: {image_dir}")
        self.size = size
        self.degrade = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.25, 0.1),
                            contrast_limit=(-0.15, 0.15),
                            p=1.0,
                        ),
                        A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    ],
                    p=0.7,
                ),
                A.GaussianBlur(blur_limit=(3, 3), p=0.2),
                A.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.0, 0.0), p=0.2),
            ]
        )
        self.to_t = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # VOC 子集路径为英文时，imread 比 imdecode+fromfile 更快更省内存。
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            # 坏图时换一张
            return self.__getitem__((idx + 1) % len(self.files))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 先缩放后增强，避免在超大图上生成噪声导致内存暴涨
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        degraded = self.degrade(image=rgb)["image"]
        xa = self.to_t(degraded)
        xb = self.to_t(rgb)
        return xa, xb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        required=True,
        help="仅含图片的目录，例如 VOC2028/VOC2028/JPEGImages",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="gan_generator_voc_synthetic.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.image_dir).resolve()
    ds = SyntheticWeatherDataset(root, size=args.imgsz)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=max(args.workers, 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    g = UNetGenerator().to(device)
    d = PatchDiscriminator().to(device)
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()
    og = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lam_l1 = 100.0
    for ep in range(1, args.epochs + 1):
        loss_d_acc = 0.0
        loss_g_acc = 0.0
        n = 0
        for xa, xb in dl:
            xa, xb = xa.to(device), xb.to(device)

            with torch.no_grad():
                fake = g(xa)
            d_real = d(xa, xb)
            d_fake = d(xa, fake.detach())
            loss_d = 0.5 * (
                bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            )
            od.zero_grad()
            loss_d.backward()
            od.step()

            fake = g(xa)
            pred_fake = d(xa, fake)
            loss_g = bce(pred_fake, torch.ones_like(pred_fake)) + lam_l1 * l1(fake, xb)
            og.zero_grad()
            loss_g.backward()
            og.step()

            loss_d_acc += loss_d.item()
            loss_g_acc += loss_g.item()
            n += 1

        print(
            f"epoch {ep}/{args.epochs} "
            f"loss_d={loss_d_acc / max(n, 1):.4f} loss_g={loss_g_acc / max(n, 1):.4f}"
        )

    torch.save({"generator": g.state_dict()}, args.save)
    print(f"saved: {args.save}")


if __name__ == "__main__":
    main()

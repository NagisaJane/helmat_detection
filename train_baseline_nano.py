"""官方 YOLO26-OBB nano 基线训练（与 yolo26-obb-hybrid-only.yaml 同档 n，便于公平对比）。"""
from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 yolo26n-obb.pt（nano），不含自定义 HybridAttention/MobileNetBlock",
    )
    parser.add_argument("--data", default="safety.yaml", help="dataset yaml path")
    parser.add_argument(
        "--model",
        default="yolo26n-obb.pt",
        help="官方 nano OBB 权重；若本地无此文件，Ultralytics 会尝试自动下载",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default=r"run\yolo26_obb_official_n")
    args = parser.parse_args()

    model = YOLO(args.model, task="obb")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=r"C:\Users\Yang\Desktop\笔记———有关深度学习\X-AnyLabeling-main\X-AnyLabeling-main",
        name=args.name,
        device=0,
        workers=0,
    )

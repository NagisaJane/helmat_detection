from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
import argparse

from custom_modules import HybridAttention, MobileNetBlock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="safety.yaml", help="dataset yaml path")
    parser.add_argument(
        "--model",
        default="yolo26-obb-hybrid-only.yaml",
        help=(
            "默认 yolo26-obb-hybrid-only.yaml（nano+仅混合注意力）。"
            "官方原版同档 nano 基线: yolo26n-obb.pt；small 为 yolo26s-obb.pt；"
            "含 MobileNet 的自定义: yolo26-obb-custom.yaml"
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default=r"run\yolo26_obb_hybrid_only")
    args = parser.parse_args()

    # 注册自定义模块（默认 yaml 仅用到 HybridAttention；换 custom yaml 时会用到 MobileNetBlock）
    for m in (HybridAttention, MobileNetBlock):
        setattr(tasks, m.__name__, m)
        setattr(modules, m.__name__, m)
        tasks.parse_model.__globals__[m.__name__] = m

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
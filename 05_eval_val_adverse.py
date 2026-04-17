from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

from custom_modules import HybridAttention, MobileNetBlock


def register_custom():
    for m in (HybridAttention, MobileNetBlock):
        setattr(tasks, m.__name__, m)
        setattr(modules, m.__name__, m)
        tasks.parse_model.__globals__[m.__name__] = m


def evaluate_one(weight: Path, use_custom: bool, data_yaml: str):
    if use_custom:
        register_custom()
    model = YOLO(str(weight), task="obb")
    m = model.val(
        data=data_yaml,
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        save_json=False,
    )
    return m.results_dict if hasattr(m, "results_dict") else {}


if __name__ == "__main__":
    project = PROJECT_ROOT
    data_yaml = "safety_adverse.yaml"
    jobs = [
        ("B0_baseline", project / r"run\exp_B0_baseline\weights\best.pt", False),
        ("B1_gan_only", project / r"run\exp_B1_gan_only\weights\best.pt", False),
        ("B2_attn_only", project / r"run\exp_B2_attn_only\weights\best.pt", True),
        ("B3_full", project / r"run\exp_B3_full\weights\best.pt", True),
    ]

    lines = []
    lines.append("model,precision,recall,mAP50,mAP50-95")
    for name, weight, use_custom in jobs:
        if not weight.exists():
            print(f"[SKIP] {name}: missing {weight}")
            continue
        print(f"[RUN] {name}")
        rd = evaluate_one(weight, use_custom, data_yaml)
        p = rd.get("metrics/precision(B)")
        r = rd.get("metrics/recall(B)")
        ap50 = rd.get("metrics/mAP50(B)")
        ap5095 = rd.get("metrics/mAP50-95(B)")
        print(f"  precision={p} recall={r} mAP50={ap50} mAP50-95={ap5095}")
        lines.append(f"{name},{p},{r},{ap50},{ap5095}")

    out = project / r"run\eval_val_adverse_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] summary saved: {out}")

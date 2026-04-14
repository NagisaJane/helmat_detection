import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from ultralytics import YOLO

# --- 动态寻找 OBB 检测头（兼容不同 ultralytics 版本） ---
def find_obb_head():
    import importlib
    import pkgutil
    import ultralytics.nn.modules as unm

    candidate_names = ("OBBDetect", "OBB26", "OBB")
    # 尝试所有可能的子模块路径
    for _, name, _ in pkgutil.walk_packages(unm.__path__, unm.__name__ + "."):
        try:
            mod = importlib.import_module(name)
            for cls_name in candidate_names:
                if hasattr(mod, cls_name):
                    print(f"🔎 Found {cls_name} in: {name}")
                    return getattr(mod, cls_name), cls_name
        except ImportError:
            continue

    # 额外兜底：直接从常见 head 模块导入
    try:
        from ultralytics.nn.modules.head import OBB26
        return OBB26, "OBB26"
    except Exception:
        pass
    try:
        from ultralytics.nn.modules.head import OBB
        return OBB, "OBB"
    except Exception:
        raise ImportError("Could not find OBB head class (OBBDetect/OBB26/OBB). Please check your ultralytics version.")

# 提取自定义模块
from custom_modules import CBAM, MobileNetBlock

# 执行寻找逻辑
try:
    OBBHead, OBBHeadName = find_obb_head()
except Exception as e:
    print(f"❌ Error: {e}")
    OBBHead, OBBHeadName = None, None

# --- 核心注入逻辑 ---
custom_list = [CBAM, MobileNetBlock]
if OBBHead:
    custom_list.append(OBBHead)

for m in custom_list:
    setattr(tasks, m.__name__, m)
    setattr(modules, m.__name__, m)
    # 注入到解析器的全局空间
    tasks.parse_model.__globals__[m.__name__] = m 

# 关键兼容：无论实际类名是 OBB/OBB26/OBBDetect，都注册 OBBDetect 别名给 yaml 使用
if OBBHead:
    setattr(tasks, "OBBDetect", OBBHead)
    setattr(modules, "OBBDetect", OBBHead)
    tasks.parse_model.__globals__["OBBDetect"] = OBBHead

print(f"✅ Registration Complete: {[m.__name__ for m in custom_list]}")
if __name__ == '__main__':
    # 关键修改：添加 task="obb" 强制指定任务类型
    model = YOLO("yolo26-obb-custom.yaml", task="obb") 
    
    model.train(
        data="safety.yaml",
        epochs=200,
        batch=8,      # 建议先用 2，稳住显存
        imgsz=640,
        project=r"C:\Users\Yang\Desktop\笔记———有关深度学习\X-AnyLabeling-main\X-AnyLabeling-main",
        name=r"run\obb_mobilenet_v1",
        device=0,
        workers=0,    
        
    )
def inner_iou(box1, box2, ratio=1.2, eps=1e-7):
    """
    Inner-IoU: 通过辅助框缩放提升回归精度
    """
    # 这里的 box1, box2 格式通常为 [x, y, w, h]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                 box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, \
                                 box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # 辅助框缩放
    w1, h1 = box1[2], box1[3]
    w2, h2 = box2[2], box2[3]
    
    # 计算中心点
    inner_b1_x1 = box1[0] - (w1 * ratio) / 2
    inner_b1_y1 = box1[1] - (h1 * ratio) / 2
    inner_b1_x2 = box1[0] + (w1 * ratio) / 2
    inner_b1_y2 = box1[1] + (h1 * ratio) / 2
    
    inner_b2_x1 = box2[0] - (w2 * ratio) / 2
    inner_b2_y1 = box2[1] - (h2 * ratio) / 2
    inner_b2_x2 = box2[0] + (w2 * ratio) / 2
    inner_b2_y2 = box2[1] + (h2 * ratio) / 2

    # 计算 Inner-IoU
    inter_x1 = torch.max(inner_b1_x1, inner_b2_x1)
    inter_y1 = torch.max(inner_b1_y1, inner_b2_y1)
    inter_x2 = torch.min(inner_b1_x2, inner_b2_x2)
    inter_y2 = torch.min(inner_b1_y2, inner_b2_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = (w1 * h1 * ratio**2) + (w2 * h2 * ratio**2) - inter_area + eps
    
    return inter_area / union_area
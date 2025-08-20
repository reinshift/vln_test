from ultralytics import YOLO
import os
import glob
from pathlib import Path

# 初始化YOLO模型
model_path = "../models/YOLOE/yoloe-11l-seg.pt"
model = YOLO(model_path)

# 设置要检测的类别
names = [
    "barrel", "bench", "billboard", "fire_hydrant",
    "tractor_trailer", "traffic_cone", "trash_bin", "tree"
]

# 设置文本提示（只加载一次）
model.set_classes(names, model.get_text_pe(names))

# 定义输入和输出目录
images_dir = Path("/catkin_ws/src/llm_model/src/images")
results_dir = Path("/catkin_ws/src/llm_model/src/results")

# 创建结果目录
os.makedirs(results_dir, exist_ok=True)

# 获取所有PNG图像文件
image_files = list(images_dir.glob("*.png"))

print(f"找到 {len(image_files)} 个图像文件进行处理:")

# 处理每个图像文件
for i, image_path in enumerate(image_files):
    # 获取相对文件名（不带路径）
    filename = image_path.name
    
    # 打印处理进度
    print(f"处理图像 {i+1}/{len(image_files)}: {filename}")
    
    # 运行检测
    results = model.predict(str(image_path), conf=0.3)
    
    # 为每个结果创建专属目录
    image_result_dir = results_dir / f"result_{i}_{image_path.stem}"
    os.makedirs(image_result_dir, exist_ok=True)
    
    # 保存结果
    for j, result in enumerate(results):
        # 保存结果图像
        result_image_path = image_result_dir / f"detection_result.jpg"
        result.save(filename=str(result_image_path))
        
        # 保存检测数据
        data_path = image_result_dir / "detection_data.txt"
        with open(data_path, "w") as f:
            f.write(f"检测结果报告 - {filename}\n")
            f.write(f"图像路径: {str(image_path)}\n")
            f.write(f"检测到的目标数量: {len(result.boxes)}\n\n")
            
            # 类别统计
            class_counts = {name: 0 for name in names}
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id < len(names):
                    class_name = names[class_id]
                    class_counts[class_name] += 1
                    
            f.write("类别统计:\n")
            for name, count in class_counts.items():
                if count > 0:
                    f.write(f"- {name}: {count}个\n")
            
            # 输出每个检测到的目标信息
            if len(result.boxes) > 0:
                f.write("\n详细检测结果:\n")
                for k, box in enumerate(result.boxes):
                    class_id = int(box.cls)
                    class_name = names[class_id] if class_id < len(names) else f"未知类别_{class_id}"
                    confidence = float(box.conf)
                    coordinates = box.xyxy[0].tolist()
                    
                    f.write(f"\n目标 #{k+1}\n")
                    f.write(f"  - 类别: {class_name}\n")
                    f.write(f"  - 置信度: {confidence:.4f}\n")
                    f.write(f"  - 位置坐标: x1={coordinates[0]:.1f}, y1={coordinates[1]:.1f}, "
                            f"x2={coordinates[2]:.1f}, y2={coordinates[3]:.1f}\n")
                    f.write(f"  - 宽度: {coordinates[2]-coordinates[0]:.1f}px, 高度: {coordinates[3]-coordinates[1]:.1f}px\n")

# 打印处理总结
print("\n" + "="*60)
print(f"所有图像处理完成! 共处理了 {len(image_files)} 个图像文件")
print(f"结果保存在: {str(results_dir)}")
print("每个图像的处理结果保存在单独的文件夹中:")
print(f"  - detection_result.jpg: 带标注的结果图像")
print(f"  - detection_data.txt: 包含检测细节的文本文件")
print("="*60)
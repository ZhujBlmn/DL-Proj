import os
import json
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from typing import List, Dict, Tuple
from tqdm import tqdm
import shutil # 导入 shutil 用于文件复制

# --- 配置 ---
OUTPUT_INFERENCE_DIR = "data/inference_results"
TEST_ANNOTATIONS_FILE = "data/processed/test_annotations.json" 
TARGET_SIZE = 128

def get_test_paths(annotations_file: str, inference_dir: str) -> List[Dict]:
    """
    从 annotations.json 加载并映射预测和真实路径。
    同时将真实目标图复制到推理输出目录，方便检查。
    """
    
    os.makedirs(inference_dir, exist_ok=True) # 确保推理目录存在
    
    try:
        with open(annotations_file, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到标注文件 {annotations_file}")
        return []

    # 仅加载用于评估的前 N 个样本 (与推理脚本保持一致，N=5)
    test_samples = raw_data[:5] 

    results = []
    
    print(f"\n--- 评估路径检查与目标图复制 ---")
    
    for i, item in enumerate(test_samples):
        # ⚠️ 注意: 假设你在推理脚本中已经解决了 'target_frame_path' 的 Key 错误
        # 并且 annotations.json 中存在 'target_frame_path' 字段。
        if 'input_frame_path' not in item or 'target_frame_path' not in item:
             print(f"警告: 标注数据第 {i} 项缺少 'input_frame_path' 或 'target_frame_path' 键。跳过。")
             continue

        video_id = os.path.basename(item["input_frame_path"]).split('_')[0] 
        
        # 1. 定义预测图和目标图在输出目录中的最终路径
        predicted_path = os.path.join(inference_dir, f"sample_{i+1}_{video_id}_predicted.png")
        true_target_output_path = os.path.join(inference_dir, f"sample_{i+1}_{video_id}_true_target.png")
        
        # 2. 从原始位置获取真实目标图路径
        original_target_path = item["target_frame_path"]

        # 3. 复制真实目标图到输出目录 (用于统一检查和评估)
        if os.path.exists(original_target_path):
            try:
                # 复制文件
                shutil.copyfile(original_target_path, true_target_output_path)
                
                # 4. 检查预测文件是否存在
                if os.path.exists(predicted_path):
                    results.append({
                        "video_id": video_id,
                        "task": item["text_prompt"],
                        "predicted_path": predicted_path,
                        "true_target_path": true_target_output_path
                    })
                    
                    # 打印路径用于检查
                    print(f"样本 {i+1}:")
                    print(f"  预测图引用路径: {predicted_path}")
                    print(f"  真实图引用路径: {true_target_output_path}")
                else:
                    print(f"警告: 样本 {i+1} 预测文件不存在: {predicted_path}")
                    
            except Exception as e:
                print(f"复制真实目标图失败: {e}")
        else:
            print(f"警告: 真实目标文件不存在于原始路径: {original_target_path}")

    print("-----------------------------------")
    return results

def calculate_metrics(img1_path: str, img2_path: str) -> Tuple[float, float]:
    """计算两张图片之间的 PSNR 和 SSIM"""
    
    # 1. 加载和预处理图像
    # 注意：PSNR/SSIM 需要 NumPy 数组作为输入
    img_pred = Image.open(img1_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
    img_true = Image.open(img2_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
    
    # 转换为 NumPy 数组 (范围 0-255)
    img_pred_np = np.array(img_pred)
    img_true_np = np.array(img_true)

    # 2. 计算 PSNR
    psnr = calculate_psnr(img_true_np, img_pred_np, data_range=255)
    
    # 3. 计算 SSIM (需要多通道=True)
    ssim = calculate_ssim(img_true_np, img_pred_np, data_range=255, channel_axis=-1, multichannel=True)
    
    return psnr, ssim

def run_quantitative_evaluation():
    """执行定量评估并打印结果"""
    
    # 1. 获取路径并执行文件复制
    evaluation_paths = get_test_paths(TEST_ANNOTATIONS_FILE, OUTPUT_INFERENCE_DIR)
    
    if not evaluation_paths:
        print("未找到用于评估的有效样本。请检查文件路径和文件是否存在。")
        return

    # 初始化用于按任务聚合的结果字典
    task_results: Dict[str, List[Dict]] = {}
    
    print(f"开始计算 {len(evaluation_paths)} 个样本的 PSNR 和 SSIM...")

    for item in tqdm(evaluation_paths, desc="Evaluating Metrics"):
        # 计算指标
        psnr, ssim = calculate_metrics(item["predicted_path"], item["true_target_path"])
        
        # 提取任务名称
        task_name = item["task"].split(' ')[0] 
        
        if task_name not in task_results:
            task_results[task_name] = []
            
        task_results[task_name].append({"psnr": psnr, "ssim": ssim})

    # 汇总和展示结果
    print("\n--- Quantitative Results Summary ---")
    
    all_psnr = []
    all_ssim = []
    
    for task, results in task_results.items():
        psnr_list = [r["psnr"] for r in results]
        ssim_list = [r["ssim"] for r in results]
        
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        
        print(f"Task: {task} (N={len(results)})")
        print(f"  Average PSNR: {avg_psnr:.4f}")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        
        all_psnr.extend(psnr_list)
        all_ssim.extend(ssim_list)
        
    print("\n----------------------------------")
    print(f"Overall Average PSNR: {np.mean(all_psnr):.4f}")
    print(f"Overall Average SSIM: {np.mean(all_ssim):.4f}")

if __name__ == '__main__':
    run_quantitative_evaluation()
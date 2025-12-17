import json
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm  # <-- 导入 tqdm
from sklearn.model_selection import train_test_split

# --- 配置 ---
DATASET_PATH = '/root/autodl-tmp/data/DL-Proj/ssv2_data/ssv2_videos/20bn-something-something-v2'
OUTPUT_DIR = 'data/processed'
TARGET_SIZE = 128
# 任务筛选
TARGET_TASKS = {
    "move_object": ["Moving something from left to right", "Pushing something from right to left"],
    "drop_object": ["Dropping something onto something", "Letting something fall down"],
    "cover_object": ["Covering something with something", "Putting something on top of something"]
}
INPUT_FRAME_INDEX = 20  # 假设使用第1帧作为观测帧

# 先减小时间偏移量
TARGET_FRAME_OFFSET = 1  

# Canny 边缘检测阈值（标准值）
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200

def extract_frame_pair(video_path):
    """从视频中提取条件帧（Canny 边缘）和目标帧（RGB），并缩放"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确保视频足够长
    if frame_count < INPUT_FRAME_INDEX + TARGET_FRAME_OFFSET:
        cap.release()
        return None, None 

    # 1. 提取条件帧 x0 (Frame_t)
    cap.set(cv2.CAP_PROP_POS_FRAMES, INPUT_FRAME_INDEX - 1)
    ret_x0, frame_x0 = cap.read()

    # 2. 提取目标帧 y (Frame_t+Delta_t)
    cap.set(cv2.CAP_PROP_POS_FRAMES, INPUT_FRAME_INDEX + TARGET_FRAME_OFFSET - 1)
    ret_y, frame_y = cap.read()
    
    cap.release()

    if ret_x0 and ret_y:
        # 目标帧 y (Frame_t+Delta_t): RGB 图像
        frame_y = cv2.cvtColor(frame_y, cv2.COLOR_BGR2RGB)
        frame_y = Image.fromarray(frame_y).resize((TARGET_SIZE, TARGET_SIZE))
        
        # ***************************************************************
        # 关键修改 2: 对条件帧 x0 (Frame_t) 进行 Canny 边缘检测
        # ---------------------------------------------------------------
        # A. 转换为灰度图
        gray_x0 = cv2.cvtColor(frame_x0, cv2.COLOR_BGR2GRAY)
        
        # B. 计算 Canny 边缘
        canny_x0 = cv2.Canny(gray_x0, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        
        # C. 转换为 RGB (Canny是单通道，但需要存储为3通道的PNG或JPEG，且通常用PIL处理)
        # 将单通道的 Canny 边缘图（0或255）转为 3 通道的 RGB 图像
        canny_x0 = cv2.cvtColor(canny_x0, cv2.COLOR_GRAY2BGR)
        
        # D. 调整尺寸并转换为 PIL 格式
        canny_x0 = Image.fromarray(canny_x0).resize((TARGET_SIZE, TARGET_SIZE))
        # ***************************************************************
        
        # 返回 Canny 边缘图作为条件帧，RGB 图像作为目标帧
        return canny_x0, frame_y
    return None, None

def load_metadata(json_path):
    """加载原始 SSV2 元数据 JSON 文件"""
    with open(json_path, 'r') as f:
        return json.load(f)

def process_dataset(train_metadata_path, label_path):
    """主处理函数，遍历元数据文件并保存帧"""
    
    # 1. 加载元数据和标签
    train_data = load_metadata(train_metadata_path)
    
    filtered_data = []
    
    # 2. 筛选数据：只选择目标任务
    allowed_prompts = set()
    for prompts in TARGET_TASKS.values():
        allowed_prompts.update(prompts)
            
    print(f"开始筛选 {len(train_data)} 个视频，目标动作数: {len(allowed_prompts)}")
    
    for item in train_data:
        video_id = str(item['id'])
        prompt = item['template'].replace('[', '').replace(']', '') # 清理提示词
        
        if prompt in allowed_prompts:
            filtered_data.append((video_id, prompt))
            
    SAMPLE_LIMIT = 1e9

            
    print(f"筛选后剩余 {len(filtered_data)} 个视频样本。")
    
    all_data = []
    
    # 3. 遍历和提取帧 (限制样本数量)
    data_iterator = tqdm(filtered_data, desc=f"Processing Frames (Max {SAMPLE_LIMIT})")
    
    for video_id, prompt in data_iterator:
        video_file = os.path.join(DATASET_PATH, f"{video_id}.webm")
        
        # 检查是否达到样本上限
        if len(all_data) >= SAMPLE_LIMIT: 
            break

        if not os.path.exists(video_file):
            continue
            
        # 注意：现在 extract_frame_pair 不再需要 text_prompt
        frame_x0_canny, frame_y_rgb = extract_frame_pair(video_file)
        
        if frame_x0_canny and frame_y_rgb:
            # 保存帧
            # x0 现在是 Canny 边缘图
            x0_path = os.path.join(OUTPUT_DIR, 'train_frames', f"{video_id}_x0_canny.png")
            y_path = os.path.join(OUTPUT_DIR, 'train_frames', f"{video_id}_y_rgb.png")
            frame_x0_canny.save(x0_path)
            frame_y_rgb.save(y_path)
            
            all_data.append({
                "input_frame_path": x0_path,
                "target_frame_path": y_path,
                "text_prompt": prompt
            })
        
        # 实时更新进度条上的样本数
        data_iterator.set_postfix(saved=len(all_data))
        
    # ----------------------------------------------------
    # 使用 sklearn.model_selection.train_test_split 划分数据
    train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=42)
    
    # 5. 保存训练集标注 JSON
    train_annotations_file = os.path.join(OUTPUT_DIR, 'train_annotations.json')
    with open(train_annotations_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    print(f"训练数据处理完成，共生成 {len(train_data)} 个样本。文件保存在: {train_annotations_file}")

    # 6. 保存测试集标注 JSON
    test_annotations_file = os.path.join(OUTPUT_DIR, 'test_annotations.json')
    with open(test_annotations_file, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"测试数据处理完成，共生成 {len(test_data)} 个样本。文件保存在: {test_annotations_file}")

if __name__ == '__main__':
    # 确保安装了 opencv-python, scikit-learn, tqdm
    # pip install opencv-python scikit-learn tqdm
    
    os.makedirs(os.path.join(OUTPUT_DIR, 'train_frames'), exist_ok=True)
    
    TRAIN_META = '/root/autodl-tmp/data/DL-Proj/ssv2_data/somethingV2/something-something-v2-train.json' 
    LABEL_META = '/root/autodl-tmp/data/DL-Proj/ssv2_data/somethingV2/something-something-v2-labels.json' 
    
    process_dataset(TRAIN_META, LABEL_META)
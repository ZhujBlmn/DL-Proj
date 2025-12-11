import json
import os
import cv2
from PIL import Image

# --- 配置 ---
DATASET_PATH = '/root/autodl-tmp/data/DL-Proj/ssv2_data/ssv2_videos/20bn-something-something-v2'
OUTPUT_DIR = 'data/processed'
TARGET_SIZE = 128  # 4090 可以挑战 128x128 [cite: 5]
# 任务筛选 [cite: 3]
TARGET_TASKS = {
    "move_object": ["Moving something from left to right", "Pushing something from right to left"],
    "drop_object": ["Dropping something onto something", "Letting something fall down"],
    "cover_object": ["Covering something with something", "Putting something on top of something"]
}
INPUT_FRAME_INDEX = 1  # 假设使用第1帧作为观测帧
TARGET_FRAME_OFFSET = 20 # 预测第 21 帧 (1 + 20) [cite: 6]

def extract_frame_pair(video_path, text_prompt):
    """从视频中提取条件帧和目标帧，并缩放"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确保视频足够长
    if frame_count < INPUT_FRAME_INDEX + TARGET_FRAME_OFFSET:
        return None, None 

    # 提取条件帧 x0
    cap.set(cv2.CAP_PROP_POS_FRAMES, INPUT_FRAME_INDEX - 1)
    ret_x0, frame_x0 = cap.read()

    # 提取目标帧 y
    cap.set(cv2.CAP_PROP_POS_FRAMES, INPUT_FRAME_INDEX + TARGET_FRAME_OFFSET - 1)
    ret_y, frame_y = cap.read()
    
    cap.release()

    if ret_x0 and ret_y:
        # 转换并调整尺寸
        frame_x0 = cv2.cvtColor(frame_x0, cv2.COLOR_BGR2RGB)
        frame_y = cv2.cvtColor(frame_y, cv2.COLOR_BGR2RGB)
        
        frame_x0 = Image.fromarray(frame_x0).resize((TARGET_SIZE, TARGET_SIZE))
        frame_y = Image.fromarray(frame_y).resize((TARGET_SIZE, TARGET_SIZE))
        
        return frame_x0, frame_y
    return None, None

def load_metadata(json_path):
    """加载原始 SSV2 元数据 JSON 文件"""
    with open(json_path, 'r') as f:
        return json.load(f)

def process_dataset(train_metadata_path, label_path):
    """主处理函数，遍历元数据文件并保存帧"""
    
    # 1. 加载元数据和标签
    train_data = load_metadata(train_metadata_path)
    # label_map = load_metadata(label_path) # SSV2 的 template/label 一致，此步可跳过
    
    filtered_data = []
    
    # 2. 筛选数据：只选择目标任务
    # 将目标任务集合转换为一个易于查找的 set
    allowed_prompts = set()
    for prompts in TARGET_TASKS.values():
        allowed_prompts.update(prompts)
        
    print(f"开始筛选 {len(train_data)} 个视频，目标动作数: {len(allowed_prompts)}")
    
    for item in train_data:
        video_id = str(item['id'])
        prompt = item['template'].replace('[', '').replace(']', '') # 清理提示词
        
        if prompt in allowed_prompts:
            filtered_data.append((video_id, prompt))
            
    print(f"筛选后剩余 {len(filtered_data)} 个视频样本。")
    
    all_data = []
    
    # 3. 遍历和提取帧 (限制样本数量)
    for video_id, prompt in filtered_data:
        video_file = os.path.join(DATASET_PATH, f"{video_id}.webm")
        
        # 检查是否达到样本上限
        if len(all_data) >= 300: #
            break

        if not os.path.exists(video_file):
            # print(f"警告：视频文件 {video_file} 不存在，跳过。")
            continue
            
        frame_x0, frame_y = extract_frame_pair(video_file, prompt)
        
        if frame_x0 and frame_y:
            # 保存帧
            x0_path = os.path.join(OUTPUT_DIR, 'train_frames', f"{video_id}_x0.png")
            y_path = os.path.join(OUTPUT_DIR, 'train_frames', f"{video_id}_y.png")
            frame_x0.save(x0_path)
            frame_y.save(y_path)
            
            all_data.append({
                "input_frame_path": x0_path,
                "target_frame_path": y_path,
                "text_prompt": prompt
            })

    # 4. 保存最终的标注 JSON
    with open(os.path.join(OUTPUT_DIR, 'annotations.json'), 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(f"数据处理完成，共生成 {len(all_data)} 个样本，annotations.json 已创建。")

if __name__ == '__main__':
    os.makedirs(os.path.join(OUTPUT_DIR, 'train_frames'), exist_ok=True)
    
    # 请替换为你的实际路径
    TRAIN_META = '/root/autodl-tmp/data/DL-Proj/ssv2_data/somethingV2/something-something-v2-train.json' 
    LABEL_META = '/root/autodl-tmp/data/DL-Proj/ssv2_data/somethingV2/something-something-v2-labels.json' 
    
    # 运行主函数
    process_dataset(TRAIN_META, LABEL_META)
    # print("数据处理完成，annotations.json 已创建。")
    """
    https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00
    https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01
    https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip
    """
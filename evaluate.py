import torch
import os
import json
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Dict
from tqdm.auto import tqdm


# 导入您在 train.py 中定义的 Dataset，用于加载测试数据
from scripts.train import VideoFrameDataset 

# --- 配置 ---
# 必须与 train.py 中的配置保持一致
MODEL_PATH = "models/controlnet_video_frame" # ControlNet权重保存的路径
BASE_SD_MODEL = "runwayml/stable-diffusion-v1-5"
TEST_ANNOTATIONS_FILE = "data/processed/annotations.json" # 假设使用训练集的前几项进行测试
TARGET_SIZE = 128
NUM_INFERENCE_STEPS = 50 # 较少的步数，用于快速生成
OUTPUT_INFERENCE_DIR = "data/inference_results"

def load_inference_pipeline(model_path: str, base_model: str, device: torch.device):
    """
    加载所有必要的组件，并创建 StableDiffusionControlNetPipeline
    """
    print(f"Loading models from: {base_model} and {model_path}")
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # 1. 加载所有组件 (从本地缓存或已下载的路径加载)
    # ControlNet 使用微调后的权重
    controlnet = ControlNetModel.from_pretrained(model_path)
    
    # 核心组件从 base_model 加载，避免网络访问
    # 注意：这里如果本地缓存完整，会自动从缓存加载
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    
    # 2. 显式创建 Pipeline，将所有组件作为参数传入
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,  # 传入调度器
        safety_checker=None,        # 评估阶段通常不需要
        feature_extractor=None,     # 评估阶段通常不需要
        requires_safety_checker=False, # 评估阶段通常不需要
    ).to(device)

    # 内存优化
    # pipe.enable_xformers_memory_efficient_attention()
    
    return pipe

def evaluate(pipe: StableDiffusionControlNetPipeline, test_data: List[Dict]):
    """
    遍历测试数据并生成未来帧，同时保存输入图片。
    """
    os.makedirs(OUTPUT_INFERENCE_DIR, exist_ok=True)
    
    # 仅测试前 5 个样本
    test_samples = test_data[:5] 
    
    print(f"Starting inference for {len(test_samples)} samples...")

    with torch.no_grad():
        for i, item in tqdm(enumerate(test_samples), total=len(test_samples), desc="Inference"):
            # 1. 准备 ControlNet 条件（Frame_t）
            input_frame_path = item["input_frame_path"]
            
            # 使用 PIL 加载 ControlNet Condition Image (Frame_t)
            control_image = Image.open(input_frame_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))

            # 2. 准备 Prompt
            prompt = item["text_prompt"]
            
            print(f"Sample {i+1}: Prompt='{prompt}'")

            # 3. 生成未来帧 (Frame_t+Delta_t)
            generated_image = pipe(
                prompt,
                control_image, # Frame_t
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=7.5,
                width=TARGET_SIZE,
                height=TARGET_SIZE
            ).images[0]

            # 4. 保存结果
            
            # 保存输入帧 (Frame_t)
            input_output_path = os.path.join(OUTPUT_INFERENCE_DIR, f"sample_{i+1}_{item['video_id']}_input.png")
            control_image.save(input_output_path)
            
            # 保存预测帧 (Frame_t+Delta_t)
            predicted_output_path = os.path.join(OUTPUT_INFERENCE_DIR, f"sample_{i+1}_{item['video_id']}_predicted.png")
            generated_image.save(predicted_output_path)
            
            print(f"-> Saved Input to {input_output_path}")
            print(f"-> Saved Prediction to {predicted_output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载推理 Pipeline
    pipeline = load_inference_pipeline(MODEL_PATH, BASE_SD_MODEL, device)
    
    # 加载数据集（用于获取视频ID和Prompt）
    try:
        # 使用 train.py 中的 Dataset 类，但只用于加载数据和prompt
        dataset = VideoFrameDataset(TEST_ANNOTATIONS_FILE, TARGET_SIZE)
        
        # 提取用于推理的数据（需要video_id, input_frame_path, text_prompt）
        test_data = []
        with open(TEST_ANNOTATIONS_FILE, 'r') as f:
            raw_data = json.load(f)
            
        for item in raw_data:
            # 假设 video_id 是从 path 中提取的，或者在 annotations.json 中
            video_id = os.path.basename(item["input_frame_path"]).split('_')[0] 
            test_data.append({
                "video_id": video_id,
                "input_frame_path": item["input_frame_path"],
                "text_prompt": item["text_prompt"]
            })
            
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        print("Please check if data/processed/annotations.json exists and is valid.")
        return

    # 执行评估
    evaluate(pipeline, test_data)

    print("\nInference complete. Results saved in:", OUTPUT_INFERENCE_DIR)


if __name__ == '__main__':
    main()
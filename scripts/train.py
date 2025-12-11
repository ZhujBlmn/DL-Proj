import torch
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator 
import json
from PIL import Image
from tqdm.auto import tqdm # 确保导入 tqdm
import os 

class VideoFrameDataset(Dataset):
    def __init__(self, annotations_file, target_size=128):
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)
        self.target_size = target_size
        # 使用 SD 默认的 CLIP 分词器
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 加载图像并进行归一化
        input_frame = Image.open(item["input_frame_path"]).convert("RGB")
        target_frame = Image.open(item["target_frame_path"]).convert("RGB")
        
        # 转换为 Tensor，归一化到 [-1, 1]
        # HWC -> CHW
        # 保持 float32
        input_tensor = torch.tensor(np.array(input_frame) / 127.5 - 1.0, dtype=torch.float32).permute(2, 0, 1) 
        target_tensor = torch.tensor(np.array(target_frame) / 127.5 - 1.0, dtype=torch.float32).permute(2, 0, 1)

        # 2. 文本编码
        text_input = self.tokenizer(
            item["text_prompt"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.tokenizer.model_max_length, 
            return_tensors="pt"
        )
        
        return {
            "input_frame": input_tensor,
            "target_frame": target_tensor,
            "input_ids": text_input.input_ids.squeeze(),
        }

def train_loop(config):
    # 更改 1：将 mixed_precision 从 "fp16" 移除
    accelerator = Accelerator()
    
    # --- 新增: 用于记录所有 Step 的 Loss ---
    all_losses = [] 
    # ------------------------------------
    
    # 1. 模型加载 (基于 Stable Diffusion v1-5)
    # 更改 2：将所有模型加载时的 dtype 移除或设置为 torch.float32（默认）
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(accelerator.device)
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(accelerator.device)
    
    # 显式冻结参数（防止产生不需要的梯度）
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 初始化 ControlNet 并将其连接到 UNet
    controlnet = ControlNetModel.from_unet(unet).to(accelerator.device)
    # 更改 3：移除 ControlNet 上的 dtype=torch.float16
    controlnet.to(accelerator.device)
    controlnet.train()

    # 2. 数据集和加载器
    train_dataset = VideoFrameDataset("data/processed/annotations.json", config["target_size"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=8
    )

    # 3. 优化器与调度器
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=config["learning_rate"]) # 只优化 ControlNet
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    
    # 4. 准备加速器
    controlnet, vae, unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        controlnet, vae, unet, text_encoder, optimizer, train_dataloader
    )
    
    # 5. 训练循环
    total_steps = len(train_dataloader) * config["num_epochs"]
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(config["num_epochs"]):
        for step, batch in enumerate(train_dataloader):
            # ... (前向传播和反向传播代码保持不变) ...
            
            # 将目标帧编码到隐空间
            latents = vae.encode(batch["target_frame"]).latent_dist.sample() * vae.config.scaling_factor
            
            # 文本编码
            encoder_hidden_states = text_encoder(batch["input_ids"])[0] 
            
            # 采样随机噪声和时间步 t
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, config["num_train_timesteps"], (latents.shape[0],), device=latents.device).long()
            
            # 前向扩散
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # ControlNet 和 UNet 预测噪声
            controlnet_outputs = controlnet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states, 
                batch["input_frame"], 
                return_dict=False
            )
            
            # 使用 ControlNet 输出和文本嵌入进行 UNet 预测
            model_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states, 
                down_block_additional_residuals=controlnet_outputs[0], 
                mid_block_additional_residual=controlnet_outputs[1], 
                return_dict=False
            )[0]
            
            # 损失计算
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            # 反向传播和更新
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # --- 新增: 记录和打印日志 ---
            current_loss = loss.detach().item()
            
            # 记录 Loss (只在主进程上进行记录和保存，避免重复)
            if accelerator.is_local_main_process:
                all_losses.append(current_loss) 

            # 打印和记录日志
            progress_bar.update(1)
            progress_bar.set_postfix(loss=current_loss)

            if accelerator.is_local_main_process and step % 50 == 0:
                accelerator.print(f"Epoch {epoch}/{config['num_epochs']} | Step {step}/{len(train_dataloader)} | Loss: {current_loss:.4f}")
            # -----------------------------------
            
    accelerator.wait_for_everyone()
    progress_bar.close()

    # 6. 保存 ControlNet 权重
    accelerator.wait_for_everyone()
    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
    
    # --- 新增: 保存 Loss 记录到 JSON 文件 ---
    if accelerator.is_local_main_process:
        loss_output_path = os.path.join(config["output_dir"], "training_loss.json")
        os.makedirs(config["output_dir"], exist_ok=True)
        with open(loss_output_path, 'w') as f:
            json.dump(all_losses, f)
        accelerator.print(f"Training loss saved to: {loss_output_path}")
    # -----------------------------------------
    
    unwrapped_controlnet.save_pretrained(config["output_dir"])
if __name__ == '__main__':
    # 由于改为 float32，显存消耗会翻倍，Batch Size 建议下调
    config = {
        "target_size": 128,
        "train_batch_size": 4, # 建议下调 Batch Size
        "learning_rate": 1e-5,
        "num_epochs": 100,
        "num_train_timesteps": 1000,
        "output_dir": "models/controlnet_video_frame"
    }
    train_loop(config)
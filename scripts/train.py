import torch
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from diffusers.optimization import get_scheduler
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator 
import json
from PIL import Image
from tqdm.auto import tqdm 
import os 
from typing import List, Dict

# 确保安装了 scikit-learn 来进行数据划分
from sklearn.model_selection import train_test_split 

class VideoFrameDataset(Dataset):
    # ----------------------------------------------------
    # 更改 1: 接受数据列表 (data: List[Dict]) 而不是文件名
    # ----------------------------------------------------
    def __init__(self, data: List[Dict], target_size=128):
        # 直接使用传入的数据列表
        self.data = data
        self.target_size = target_size
        # 使用 SD 默认的 CLIP 分词器
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 加载图像并进行归一化
        # input_frame 是 Canny 边缘图
        input_frame = Image.open(item["input_frame_path"]).convert("RGB")
        # target_frame 是真实目标 RGB 帧
        target_frame = Image.open(item["target_frame_path"]).convert("RGB")
        
        # 转换为 Tensor，归一化到 [-1, 1]
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
    accelerator = Accelerator()
    
    all_losses = [] 
    all_val_losses = [] # 新增：用于记录验证损失

    # 1. 模型加载 (保持不变)
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(accelerator.device)
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(accelerator.device)
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(accelerator.device)
    controlnet.train()

    # 2. 数据集加载和划分
    full_train_data_path = "data/processed/train_annotations.json"
    
    try:
        with open(full_train_data_path, 'r') as f:
            full_train_data = json.load(f)
    except FileNotFoundError:
        accelerator.print(f"错误：找不到训练标注文件 {full_train_data_path}。请先运行预处理。")
        return
        
    # 从训练集中划分出 10% 作为验证集 
    train_data, val_data = train_test_split(
        full_train_data, test_size=0.111, random_state=42 
    )

    # 创建数据集和加载器
    train_dataset = VideoFrameDataset(data=train_data, target_size=config["target_size"]) 
    val_dataset = VideoFrameDataset(data=val_data, target_size=config["target_size"])     
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=8
    )
    # Validation Dataloader 不需要 Shuffle
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["train_batch_size"], shuffle=False, num_workers=8
    )

    # 3. 优化器与调度器 (保持不变)

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=config["learning_rate"])

    # 计算总更新步数
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = config["num_epochs"] * num_update_steps_per_epoch

    # with warm up 余弦调度器
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes, # 前500步慢慢增加LR
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    
    # 4. 准备加速器
    controlnet, vae, unet, text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, vae, unet, text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # 5. 训练循环
    total_steps = len(train_dataloader) * config["num_epochs"]
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(config["num_epochs"]):
        controlnet.train() # 确保在训练开始时设置为 train 模式
        for step, batch in enumerate(train_dataloader): 
            # 训练代码块 (保持不变)
            latents = vae.encode(batch["target_frame"]).latent_dist.sample() * vae.config.scaling_factor
            encoder_hidden_states = text_encoder(batch["input_ids"])[0] 
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, config["num_train_timesteps"], (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            controlnet_outputs = controlnet(
                noisy_latents, timesteps, encoder_hidden_states, batch["input_frame"], return_dict=False
            )
            model_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states, 
                down_block_additional_residuals=controlnet_outputs[0], 
                mid_block_additional_residual=controlnet_outputs[1], 
                return_dict=False
            )[0]
            
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            accelerator.backward(loss)
            optimizer.step()
            # learning rate 调度器步进
            lr_scheduler.step()
            optimizer.zero_grad()
            
            current_loss = loss.detach().item()
            
            if accelerator.is_local_main_process:
                all_losses.append(current_loss) 
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=current_loss)
            
            if (step + 1) % 500 == 0:
                # -----------------------------------------------------------------
                # Validation
                # -----------------------------------------------------------------
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    controlnet.eval() # 切换到评估模式
                    val_loss_sum = 0
                    val_count = 0
                    
                    with torch.no_grad(): # 禁用梯度计算
                        val_progress_bar = tqdm(val_dataloader, desc="Validation")
                        for val_batch in val_progress_bar:
                            
                            # 运行与训练相同的模型预测逻辑
                            val_latents = vae.encode(val_batch["target_frame"]).latent_dist.sample() * vae.config.scaling_factor
                            val_encoder_hidden_states = text_encoder(val_batch["input_ids"])[0] 
                            val_noise = torch.randn_like(val_latents)
                            val_timesteps = torch.randint(0, config["num_train_timesteps"], (val_latents.shape[0],), device=val_latents.device).long()
                            val_noisy_latents = noise_scheduler.add_noise(val_latents, val_noise, val_timesteps)

                            val_controlnet_outputs = controlnet(
                                val_noisy_latents, val_timesteps, val_encoder_hidden_states, val_batch["input_frame"], return_dict=False
                            )
                            
                            val_model_pred = unet(
                                val_noisy_latents, val_timesteps, val_encoder_hidden_states, 
                                down_block_additional_residuals=val_controlnet_outputs[0], 
                                mid_block_additional_residual=val_controlnet_outputs[1], 
                                return_dict=False
                            )[0]
                            
                            val_loss = F.mse_loss(val_model_pred, val_noise, reduction="mean")
                            val_loss_sum += val_loss.item()
                            val_count += 1

                    avg_val_loss = val_loss_sum / val_count
                    all_val_losses.append(avg_val_loss)
                    accelerator.print(f"Epoch {epoch}/{config['num_epochs']} | **Validation Loss: {avg_val_loss:.4f}**")
                    
                    controlnet.train() # 验证结束后，切换回训练模式

            if accelerator.is_local_main_process and step % 50 == 0:
                accelerator.print(f"Epoch {epoch}/{config['num_epochs']} | Step {step}/{len(train_dataloader)} | Loss: {current_loss:.4f}")
        
            
    accelerator.wait_for_everyone()
    progress_bar.close()

    # 6. 保存 ControlNet 权重和 Loss 记录
    accelerator.wait_for_everyone()
    unwrapped_controlnet = accelerator.unwrap_model(controlnet)

    if accelerator.is_local_main_process:
        # 保存训练 Loss
        loss_output_path = os.path.join(config["output_dir"], "training_loss.json")
        os.makedirs(config["output_dir"], exist_ok=True)
        with open(loss_output_path, 'w') as f:
            json.dump(all_losses, f)
        accelerator.print(f"Training loss saved to: {loss_output_path}")
        
        # 保存验证 Loss
        val_loss_output_path = os.path.join(config["output_dir"], "validation_loss.json")
        with open(val_loss_output_path, 'w') as f:
            json.dump(all_val_losses, f)
        accelerator.print(f"Validation loss saved to: {val_loss_output_path}")

        unwrapped_controlnet.save_pretrained(config["output_dir"])

if __name__ == '__main__':
    config = {
        "target_size": 128,
        "train_batch_size": 8, 
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "num_train_timesteps": 1000,
        "output_dir": "models/controlnet_video_frame"
    }
    train_loop(config)
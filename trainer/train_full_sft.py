import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings

import torch
from torch import optim
from torch.utils.data import DataLoader

from lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger,
    build_accelerator,
    get_lr,
    init_model,
    load_checkpoint,
    save_checkpoint,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, accelerator, start_step=0):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(accelerator.device, non_blocking=True)
        labels = labels.to(accelerator.device, non_blocking=True)
        lr = get_lr(
            epoch * iters + step,
            args.epochs * iters,
            args.learning_rate,
            args.warmup_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with accelerator.accumulate(model):
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )

        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                accelerator=accelerator,
                export_dir=args.save_dir,
                epoch=epoch,
                step=step,
                save_dir="../checkpoints",
            )

        del input_ids, labels, res, loss


# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="full_sft", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="每张卡的batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="线性warmup占总步数比例，0为关闭")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="兼容参数，实际设备由Accelerate决定")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--seed", default=42, type=int, help="随机种子（每个rank为seed+rank）")
    parser.add_argument("--data_path", type=str, default="../data/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument("--from_weight", type=str, default="pretrain", help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
# fmt: on

    # ========== 1. Accelerate 初始化 ==========
    accelerator = build_accelerator(args)

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        load_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 4. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        if ckp_data.get("scaler") and accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 5. 编译并交给Accelerate管理 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    model, optimizer = accelerator.prepare(model, optimizer)

    # ========== 6. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        loader = accelerator.prepare(loader)

        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        if skip > 0:
            loader = accelerator.skip_first_batches(loader, skip)
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{skip}个step，从step {skip + 1}开始")
        train_epoch(epoch, loader, len(loader) + skip, accelerator, skip)

    # ========== 7. 清理 ==========
    accelerator.wait_for_everyone()
    accelerator.end_training()

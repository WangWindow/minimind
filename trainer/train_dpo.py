import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from lm_dataset import DPODataset
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


def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, accelerator, ref_model, lm_config, start_step=0, beta=0.1):
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch["x_chosen"].to(accelerator.device, non_blocking=True)
        x_rejected = batch["x_rejected"].to(accelerator.device, non_blocking=True)
        y_chosen = batch["y_chosen"].to(accelerator.device, non_blocking=True)
        y_rejected = batch["y_rejected"].to(accelerator.device, non_blocking=True)
        mask_chosen = batch["mask_chosen"].to(accelerator.device, non_blocking=True)
        mask_rejected = batch["mask_rejected"].to(accelerator.device, non_blocking=True)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(
            epoch * iters + step,
            args.epochs * iters,
            args.learning_rate,
            args.warmup_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with accelerator.accumulate(model):
            with torch.no_grad(), accelerator.autocast():
                ref_logits = ref_model(x).logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min"
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

        del (
            x_chosen,
            x_rejected,
            y_chosen,
            y_rejected,
            mask_chosen,
            mask_rejected,
            x,
            y,
            mask,
        )
        del ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="dpo", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="每张卡的batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="线性warmup占总步数比例，0为关闭")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="兼容参数，实际设备由Accelerate决定")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--seed", default=42, type=int, help="随机种子（每个rank为seed+rank）")
    parser.add_argument("--data_path", type=str, default="../data/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument("--from_weight", default="full_sft", type=str, help="基于哪个权重训练")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--beta", default=0.15, type=float, help="DPO中的beta参数")
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

    # ========== 3. 定义策略模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M")

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
        train_epoch(
            epoch,
            loader,
            len(loader) + skip,
            accelerator,
            ref_model,
            lm_config,
            skip,
            args.beta,
        )

    # ========== 7. 清理 ==========
    accelerator.wait_for_everyone()
    accelerator.end_training()

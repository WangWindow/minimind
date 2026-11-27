import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import re
import warnings

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lm_dataset import RLAIFDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.rollout_engine import create_rollout_engine
from trainer.trainer_utils import (
    LMForRewardModel,
    Logger,
    build_accelerator,
    init_model,
    load_checkpoint,
    rep_penalty,
    save_checkpoint,
)

warnings.filterwarnings("ignore")


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # lm_head 不参与 forward，仅靠 tie_word_embeddings 与 embed_tokens 共享权重，解绑后 DDP 会报未使用参数
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if "</think>" in response:
                thinking_content, answer_content = response.split("</think>", 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count("</think>") == 1 else -0.25
                answer = answer_content.strip()
            rewards[i] -= rep_penalty(answer)

            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(
    epoch,
    loader,
    iters,
    accelerator,
    rollout_engine,
    ref_model,
    actor_scheduler,
    critic_scheduler,
    reward_model,
    start_step=0,
):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            padding_side="left",
        ).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        responses_text = rollout_result.completions
        old_resp_logp = rollout_result.per_token_logps.to(args.device)
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        if args.debug_mode and accelerator.is_main_process and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger("-" * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(
                    f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}"
                )
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger("=" * 100)

        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        B = len(prompts)
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.to(full_mask.dtype))
        resp_lengths = resp_pad_mask.sum(dim=1)
        valid_resp = resp_lengths > 0
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            critic_for_rollout = accelerator.unwrap_model(critic_model)
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask

            with accelerator.autocast():
                ref_resp_logp = (
                    F.log_softmax(
                        ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1],
                        dim=-1,
                    )
                    .gather(2, labels.unsqueeze(-1))
                    .squeeze(-1)
                    .gather(1, logp_pos)
                )
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]
            token_rewards[
                torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]
            ] += rewards[valid_resp]  # 末尾加外部奖励

            gen_len = old_resp_values.size(1)
            lastgaelam = torch.zeros(B, device=args.device)
            advs_rev = []
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            returns = advantages + old_resp_values  # [B, R]

            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = (
                (advantages - adv_mean) ** 2 * resp_policy_mask
            ).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            b_inds = torch.randperm(B, device=args.device)
            for i in range(0, B, mb_size):
                inds = b_inds[i : i + mb_size]

                mb_values_seq = critic_model(
                    input_ids=gen_out[inds], attention_mask=full_mask[inds]
                )
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                res = actor_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                aux_loss = (
                    res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                )
                mb_resp_logp = (
                    F.log_softmax(res.logits[:, :-1], dim=-1)
                    .gather(2, labels[inds].unsqueeze(-1))
                    .squeeze(-1)
                    .gather(1, logp_pos[inds])
                )

                log_ratio = mb_resp_logp - old_resp_logp[inds]

                # 可开关的诊断：观察首轮首个 minibatch 的 mb 与 old logp 差异。
                if (
                    args.debug_log_ratio
                    and ppo_epoch == 0
                    and i == 0
                    and accelerator.is_main_process
                ):
                    _lr = log_ratio.detach()
                    _m = resp_policy_mask[inds].bool()
                    if _m.any():
                        _lrv = _lr[_m]
                        Logger(
                            f"[DBG log_ratio] step={step} max|lr|={_lrv.abs().max().item():.6e} "
                            f"mean|lr|={_lrv.abs().mean().item():.6e} "
                            f"ratio_max={torch.exp(_lrv).max().item():.6f} "
                            f"ratio_min={torch.exp(_lrv).min().item():.6f} "
                            f"dropout={getattr(lm_config, 'dropout', None)} "
                            f"training={actor_model.training}"
                        )
                approx_kl = (
                    0.5 * (log_ratio**2) * resp_policy_mask[inds]
                ).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                approx_kl_val = approx_kl.detach().clone()
                approx_kl_val = accelerator.reduce(approx_kl_val, reduction="mean")

                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                ratio = torch.exp(log_ratio)
                clipfrac = (
                    ((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]
                ).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                kl_ref_penalty = (
                    (
                        torch.exp(ref_resp_logp[inds] - mb_resp_logp)
                        - (ref_resp_logp[inds] - mb_resp_logp)
                        - 1.0
                    )
                    * resp_policy_mask[inds]
                ).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                policy_loss = (
                    torch.max(
                        -advantages[inds] * ratio,
                        -advantages[inds]
                        * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon),
                    )
                    * resp_policy_mask[inds]
                ).sum() / resp_policy_mask[inds].sum().clamp(min=1) + args.kl_coef * kl_ref_penalty
                value_loss = (
                    0.5
                    * (
                        torch.max(
                            (mb_resp_values - returns[inds]) ** 2,
                            (
                                torch.clamp(
                                    mb_resp_values,
                                    old_resp_values[inds] - args.cliprange_value,
                                    old_resp_values[inds] + args.cliprange_value,
                                )
                                - returns[inds]
                            )
                            ** 2,
                        )
                        * resp_value_mask[inds]
                    ).sum()
                    / resp_value_mask[inds].sum().clamp(min=1)
                )

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = policy_loss + args.vf_coef * value_loss + aux_loss

                with accelerator.accumulate(actor_model, critic_model):
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                        accelerator.clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(actor_model)

        if accelerator.is_main_process:
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = (
                actor_optimizer.param_groups[0]["lr"],
                critic_optimizer.param_groups[0]["lr"],
            )

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}"
            )

        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=actor_model,
                optimizer=actor_optimizer,
                accelerator=accelerator,
                export_dir=args.save_dir,
                epoch=epoch,
                step=step,
                save_dir="../checkpoints",
                scheduler=actor_scheduler,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )

        del (
            enc,
            gen_out,
            completion_ids,
            responses_text,
            rewards,
            full_mask,
            values_seq,
            advantages,
        )
        del (
            labels,
            resp_labels,
            resp_idx,
            resp_pad_mask,
            valid_resp,
            eos_mask,
            has_eos,
            eos_pos,
            resp_lengths,
            resp_policy_mask,
            resp_value_mask,
            old_resp_logp,
            ref_resp_logp,
        )
        del (
            kl,
            kl_ref,
            policy_loss,
            value_loss,
            loss,
            token_rewards,
            returns,
            old_resp_values,
            prompt_lens,
            logp_pos,
        )


# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="ppo_actor", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="兼容参数，实际设备由Accelerate决定")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--max_seq_len", default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../data/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument("--from_weight", default="full_sft", type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--debug_log_ratio", action="store_true", help="打印首轮首个minibatch的log_ratio差异量级，用于核查ratio≈1是否成立")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--seed", default=42, type=int, help="随机种子（每个rank为seed+rank）")
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

    # ========== 3. 初始化模型和数据 ==========
    base_weight = args.from_weight
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    rollout_engine = create_rollout_engine(
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=accelerator.autocast,
    )
    train_ds = RLAIFDataset(
        args.data_path,
        tokenizer,
        max_length=(args.max_seq_len + args.max_gen_len),
        thinking_ratio=args.thinking_ratio,
    )
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    iters = math.ceil(len(train_ds) / (args.batch_size * accelerator.num_processes))
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(
        iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps
    )
    actor_scheduler = CosineAnnealingLR(
        actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    critic_scheduler = CosineAnnealingLR(
        critic_optimizer,
        T_max=total_optimizer_steps,
        eta_min=args.critic_learning_rate / 10,
    )

    # ========== 4. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 5. 编译并交给Accelerate管理 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger("torch.compile enabled")
    (
        actor_model,
        critic_model,
        actor_optimizer,
        critic_optimizer,
        actor_scheduler,
        critic_scheduler,
    ) = accelerator.prepare(
        actor_model,
        critic_model,
        actor_optimizer,
        critic_optimizer,
        actor_scheduler,
        critic_scheduler,
    )
    rollout_engine.update_policy(actor_model)

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
        ppo_train_epoch(
            epoch,
            loader,
            len(loader) + skip,
            accelerator,
            rollout_engine,
            ref_model,
            actor_scheduler,
            critic_scheduler,
            reward_model,
            skip,
        )

    # ========== 7. 清理 ==========
    accelerator.wait_for_everyone()
    accelerator.end_training()

"""
训练工具函数集合
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ast
import math
import operator
import random
import re

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModel, AutoTokenizer

from model.model_minimind import MiniMindForCausalLM


# ======== 模型参数量 ========
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_tok", 0)
    n_shared = getattr(config, "n_shared_experts", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    shared_expert = (
        sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    )
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")


# ======== 主进程判断 ========
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ======== 日志 ========
def Logger(content):
    if is_main_process():
        print(content)


# ======== 混合精度类型 ========
def resolve_dtype(dtype, device):
    if dtype != "bfloat16" or not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return dtype
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return dtype
    Logger(f"{torch.cuda.get_device_name(device)} 不原生支持 bfloat16，已回退为 float16")
    return "float16"


# ======== 学习率 ========
def get_lr(current_step, total_steps, lr, warmup_ratio=0.0):
    warmup = max(1, int(warmup_ratio * total_steps)) if warmup_ratio > 0 else 0
    if current_step < warmup:
        return lr * (current_step + 1) / warmup
    angle = math.pi * (current_step - warmup) / max(1, total_steps - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(angle)))


# ======== 随机种子 ========
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======== 断点保存与续训 ========
def _checkpoint_paths(lm_config, weight, save_dir):
    moe_path = "_moe" if lm_config.use_moe else ""
    filename = f"{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"
    return filename, resume_path


def save_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    save_dir="../checkpoints",
    export_dir=None,
    accelerator=None,
    scaler=None,
    **kwargs,
):
    """保存训练状态，支持单卡、DDP 和 Accelerate。

    - 推理权重：写入 export_dir（如 ../out）
    - 续训状态：写入 save_dir/*_resume.pth（model + optimizer + scaler + epoch/step）
    """
    if model is None or optimizer is None:
        raise ValueError("save_checkpoint 需要 model 和 optimizer")

    if accelerator is not None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            return
        model = accelerator.unwrap_model(model)
        if scaler is None:
            scaler = accelerator.scaler

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
    filename, resume_path = _checkpoint_paths(lm_config, weight, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, filename)
        export_tmp = export_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, export_tmp)
        os.replace(export_tmp, export_path)

    resume_data = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
    }
    if scaler is not None:
        resume_data["scaler"] = scaler.state_dict()
    for key, value in kwargs.items():
        if value is not None:
            if hasattr(value, "state_dict"):
                raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                raw_value = getattr(raw_value, "_orig_mod", raw_value)
                resume_data[key] = raw_value.state_dict()
            else:
                resume_data[key] = value

    resume_tmp = resume_path + ".tmp"
    torch.save(resume_data, resume_tmp)
    os.replace(resume_tmp, resume_path)
    del state_dict, resume_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if accelerator is not None:
        accelerator.wait_for_everyone()


def load_checkpoint(lm_config, weight="full_sft", save_dir="../checkpoints"):
    """加载 save_checkpoint 保存的续训状态（模型 + 优化器 + epoch/step）。"""
    moe_path = "_moe" if lm_config.use_moe else ""
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"
    if not os.path.exists(resume_path):
        return None
    ckp_data = torch.load(resume_path, map_location="cpu")
    saved_ws = ckp_data.get("world_size", 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws:
        ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
        Logger(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")
    return ckp_data


# ======== Accelerate 公共初始化 ========
def build_accelerator(args):
    """根据 MiniMind 训练参数创建 Accelerator，并统一 device / 种子 / 混合精度。"""
    from accelerate import Accelerator
    from accelerate.utils import DistributedType

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        args.dtype = resolve_dtype(args.dtype, f"cuda:{local_rank}")
        mixed_precision = {"bfloat16": "bf16", "float16": "fp16"}.get(args.dtype, "no")
    else:
        mixed_precision = "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.distributed_type in (
        DistributedType.DEEPSPEED,
        DistributedType.FSDP,
    ):
        raise ValueError(
            "MiniMind 训练脚本目前只支持单卡/DDP。请运行 `accelerate config`，选择 multi-GPU 并关闭 DeepSpeed/FSDP。"
        )
    args.device = str(accelerator.device)
    setup_seed(args.seed + accelerator.process_index)
    Logger(
        f"Accelerate: {accelerator.distributed_type}, processes: {accelerator.num_processes}, "
        f"device: {args.device}, precision: {accelerator.mixed_precision}"
    )
    return accelerator


# ======== 模型初始化 ========
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path="../model",
    save_dir="../out",
    device="cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(
        f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M"
    )
    return model.to(device), tokenizer


# ======== 奖励模型 ========
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]["content"] if messages else ""
        message_context = (
            f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}"
            if history_text
            else last_query
        )
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


# ======== 数学表达式安全求值 ========
def safe_math_eval(expression):
    """对模型生成的数学表达式求值：支持 + - * / // % **、math 白名单函数与 pi/e/tau 常量。"""

    def pow_guard(base, exp):  # 幂运算统一走这里，拦截 9**9**9 / 10**99999 这类把进程算死的输入
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100):
            raise ValueError("幂运算结果过大")
        return base**exp

    def resolve(node):  # 同时兼容 sqrt(4) 与 math.sqrt(4) 两种写法
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", "") == "math":
            return node.attr

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        name = resolve(node)
        if name in consts:
            return consts[name]
        if isinstance(node, ast.UnaryOp):
            return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp):
            return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and not node.keywords:
            return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f"不支持的表达式语法: {type(node).__name__}")

    consts = {"pi": math.pi, "e": math.e, "tau": math.tau}
    funcs = {
        "pow": pow_guard,
        **{
            n: getattr(math, n)
            for n in [
                "sqrt",
                "exp",
                "log",
                "log2",
                "log10",
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "atan2",
                "sinh",
                "cosh",
                "tanh",
                "floor",
                "ceil",
                "trunc",
                "fabs",
                "fmod",
                "hypot",
                "gcd",
                "degrees",
                "radians",
            ]
        },
    }
    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: pow_guard,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    chars = str.maketrans(
        {
            "^": "**",
            "×": "*",
            "÷": "/",
            "−": "-",
            "²": "**2",
            "³": "**3",
            "（": "(",
            "）": ")",
        }
    )
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512:
        raise ValueError("表达式为空或过长")
    try:
        return walk(ast.parse(expr, mode="eval").body)
    except KeyError, SyntaxError, TypeError:
        raise ValueError("不支持的表达式语法") from None


# ======== 重复惩罚 ========
def rep_penalty(text, n=3, cap=0.5):
    cjk = (
        "\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
        "\U00020000-\U0002ee5f\U0002f800-\U0002fa1f\U00030000-\U0003347f"
    )
    toks = re.findall(rf"[{cjk}]|[^\W{cjk}]+|[^\w\s]", text.lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

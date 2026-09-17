#!/usr/bin/env python
"""Fine-tune a Gemma-3 causal LM for instruction-conditional Turkish PII masking.

Completion-only loss (prompt tokens masked with -100), dynamic padding, length grouping.
Works on CUDA (bf16) and Apple MPS (for smoke tests). Full fine-tune by default; --lora_r > 0 for LoRA.

  # continue from the published checkpoint, full fine-tune (recommended on a real GPU)
  python scripts/train.py --model models/pii-guard-turkish-270m --out outputs/v1 --epochs 2 --lr 5e-5 --bsz 32
  # from the base model
  python scripts/train.py --model google/gemma-3-270m-it --out outputs/base_v1 --epochs 2 --lr 1e-4
"""
import argparse, json, math, os, random, sys, time
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback)


class MpsCacheCallback(TrainerCallback):
    """Apple MPS caching allocator grows without bound on variable-length batches; flush it periodically."""
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 25 == 0:
            torch.mps.empty_cache()
    def on_evaluate(self, args, state, control, **kw):
        torch.mps.empty_cache()


def build_prompt(tok, instruction, text):
    return (f"{tok.bos_token}<start_of_turn>user\n{instruction}\n\n"
            f"Metin: {text}<end_of_turn>\n<start_of_turn>model\n")


class PIIDataset(Dataset):
    def __init__(self, path, tok, max_len, limit=0):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                self.rows.append(json.loads(line))
        self.tok, self.max_len = tok, max_len
        self.eot = tok.convert_tokens_to_ids("<end_of_turn>")
        self.dropped = 0

    def __len__(self): return len(self.rows)

    def encode(self, r):
        p_ids = self.tok(build_prompt(self.tok, r["instruction"], r["input"]), add_special_tokens=False)["input_ids"]
        c_ids = self.tok(r["output"], add_special_tokens=False)["input_ids"] + [self.eot]
        ids = (p_ids + c_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + c_ids)[: self.max_len]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    def __getitem__(self, i): return self.encode(self.rows[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=0, help="0 = full fine-tune")
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="use only first N train rows (smoke tests)")
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    mps = (not cuda) and torch.backends.mps.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (cuda or mps) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    if args.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg); model.print_trainable_parameters()

    train_ds = PIIDataset(args.train, tok, args.max_len, args.limit)
    val_ds = PIIDataset(args.val, tok, args.max_len, 2000)
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds)}  device: {'cuda' if cuda else 'mps' if mps else 'cpu'}", flush=True)
    ex = train_ds[0]
    print("sample decoded:", repr(tok.decode(ex["input_ids"])), "\nlabel tokens:", sum(1 for x in ex["labels"] if x != -100), flush=True)

    collator = DataCollatorForSeq2Seq(tok, model=None, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    steps_per_epoch = math.ceil(len(train_ds) / (args.bsz * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup_steps = int(args.warmup * total_steps) if args.warmup < 1 else int(args.warmup)
    print(f"total optimizer steps: {total_steps}  warmup: {warmup_steps}", flush=True)
    ta_kwargs = dict(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz, per_device_eval_batch_size=args.bsz * 2, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps, weight_decay=args.weight_decay,
        bf16=cuda, fp16=False, logging_steps=25, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2, load_best_model_at_end=False,
        group_by_length=True, dataloader_num_workers=0 if mps else 4, report_to=["none"], seed=args.seed,
        remove_unused_columns=False, optim="adamw_torch_fused" if cuda else "adamw_torch", max_grad_norm=1.0,
    )
    import inspect
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in ta_kwargs if k not in allowed]
    if dropped: print("TrainingArguments: dropping unsupported kwargs for this transformers version:", dropped, flush=True)
    targs = TrainingArguments(**{k: v for k, v in ta_kwargs.items() if k in allowed})
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
                      callbacks=[MpsCacheCallback()] if mps else [])
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    print(f"train time: {(time.time()-t0)/60:.1f} min", flush=True)
    final = os.path.join(args.out, "final")
    if args.lora_r > 0:
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    # keep the chat template + generation config consistent with the published model
    src_gen = os.path.join(args.model, "generation_config.json")
    if os.path.exists(src_gen):
        import shutil; shutil.copy(src_gen, os.path.join(final, "generation_config.json"))
    with open(os.path.join(final, "train_args.json"), "w") as f: json.dump(vars(args), f, indent=2)
    print("saved ->", final)


if __name__ == "__main__":
    main()

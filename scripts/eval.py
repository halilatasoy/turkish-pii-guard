#!/usr/bin/env python
"""Benchmark evaluation for Turkish PII masking models (exact match, per category/slice).

Usage:
  python scripts/eval.py --model models/pii-guard-turkish-270m --out results/baseline.csv
  python scripts/eval.py --model outputs/v1/merged --out results/v1.csv --batch 64
"""
import argparse, json, re, sys, time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TAGS = "[AD] [TCKN] [DOGUM_TARIHI] [DOGUM_YERI] [ANNE_ADI] [ANNE_KIZLIK] [BABA_ADI] [PASAPORT_NO] [EHLIYET_NO] [SGK_NO] [IMZA] [IBAN] [HESAP_NO] [KART] [KART_SKT] [CVV] [MAAS] [VERGI_NO] [MUSTERI_NO] [KREDI_NOTU] [POLICE_NO] [SOZLESME_NO] [KRIPTO_CUZDAN] [TEL] [EMAIL] [ADRES] [KONUM] [SAGLIK] [DIN] [ETNIK_KOKEN] [SENDIKA] [BIYOMETRIK] [CEZA_KAYDI] [KAN_GRUBU] [ENGEL_DURUMU] [SIFRE] [PIN] [KULLANICI_ADI] [IP_ADRES] [MAC_ADRES] [IMEI] [CIHAZ_ID] [PLAKA] [SASI_NO] [MOTOR_NO] [RUHSAT_NO] [SICIL_NO] [ISYERI] [AILE] [REFERANS] [YAS] [CINSIYET] [UYRUK]".split()
DEMOG = {"[YAS]", "[CINSIYET]", "[UYRUK]"}


def build_prompt(tok, instruction, text, fmt):
    if fmt == "gemma":
        return (f"{tok.bos_token}<start_of_turn>user\n{instruction}\n\n"
                f"Metin: {text}<end_of_turn>\n<start_of_turn>model\n")
    return (f"<|im_start|>user\n{instruction}\n\nMetin: {text}<|im_end|>\n<|im_start|>assistant\n")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="data/benchmark_1000.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--fmt", choices=["gemma", "chatml"], default="gemma")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = pick_device()
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    eos_ids = [tok.convert_tokens_to_ids("<end_of_turn>")] if args.fmt == "gemma" else [tok.convert_tokens_to_ids("<|im_end|>")]
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)

    df = pd.read_csv(args.bench)
    if args.limit:
        df = df.head(args.limit)
    prompts = [build_prompt(tok, r.instruction, r.input, args.fmt) for r in df.itertuples()]
    # sort by length for efficient batching, keep original order
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    preds = [None] * len(prompts)
    t0 = time.time()
    for s in range(0, len(order), args.batch):
        idx = order[s:s + args.batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for i, g in zip(idx, gen):
            preds[i] = tok.decode(g, skip_special_tokens=True).strip()
        done = s + len(idx)
        print(f"{done}/{len(order)}  {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    df["pred"] = preds
    df["dogru"] = df.pred == df.expected_output
    df["schema_neutral"] = [not (set(str(m).split()) & DEMOG) for m in df.maskelenecek_etiketler.fillna("")]
    df["out_of_schema_tag"] = [bool(set(re.findall(r"\[[A-Z_]+\]", p)) - set(TAGS)) for p in df.pred]
    df.to_csv(args.out, index=False)

    rep = {"model": args.model, "n": len(df), "exact_match": round(df.dogru.mean(), 4),
           "schema_neutral": round(df[df.schema_neutral].dogru.mean(), 4),
           "schema_neutral_n": int(df.schema_neutral.sum()),
           "by_kategori": df.groupby("kategori").dogru.mean().round(3).to_dict(),
           "by_ozellik": {}, "out_of_schema_rows": int(df.out_of_schema_tag.sum()),
           "seconds": round(time.time() - t0)}
    for o in sorted({x for s in df.ozellikler.fillna("") for x in s.split(",") if x}):
        m = df.ozellikler.fillna("").str.split(",").apply(lambda l: o in l)
        rep["by_ozellik"][o] = {"acc": round(df[m].dogru.mean(), 3), "n": int(m.sum())}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    with open(args.out.replace(".csv", ".json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

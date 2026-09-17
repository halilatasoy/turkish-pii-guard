#!/usr/bin/env python
"""Upload a trained checkpoint (+ model card) to the Hugging Face Hub.

  python scripts/push_to_hub.py --model outputs/local_80k/final --repo <user>/turkish-pii-guard-270m \
      --results results/local_80k.json [--private]
Requires a prior `hf auth login` (token is read from the local HF cache; never pass it on the command line).
"""
import argparse, json, os
from huggingface_hub import HfApi, whoami

CARD = """---
language:
- tr
license: gemma
base_model: google/gemma-3-270m-it
pipeline_tag: text-generation
library_name: transformers
tags:
- pii
- anonymization
- masking
- kvkk
- gdpr
- turkish
- privacy
---
# {repo_name}

Gemma-3-270m-it tabanlı Türkçe PII maskeleme modeli; zayıf dilimleri hedefleyen sentetik Türkçe bankacılık/ERP
verisiyle **devam eğitimi** (continued full fine-tune) ile üretildi. 53 PII etiketi, talimat-koşullu maskeleme
(tam / beyaz liste / kara liste / kapsam dışı). 

Kod, veri üreteci ve değerlendirme: https://github.com/halilatasoy/turkish-pii-guard

## Benchmark (halka açık 1000 satırlık Türkçe PII maskeleme benchmark'ı, satır düzeyi tam eşleşme)

| Model | Tam eşleşme (1000) | Şema-nötr (903) |
|---|---|---|
| taban model (270m) | 0.743 | 0.773 |
| **bu model** | **{exact}** | **{schema}** |

Kategori: {by_kat}

Zor dilimler: {by_oz}

Eğitim: {train_desc}. Benchmark yalnızca test için kullanıldı; eğitim kalıpları benchmark cümle ve talimat kalıplarından
bağımsız yazıldı, benchmark girdileriyle çakışan satırlar üretimde atıldı.

## Kullanım

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "{repo_name}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()

def maskele(metin, talimat="Metindeki tüm kişisel verileri uygun etiketlerle maskele."):
    prompt = (f"{{tokenizer.bos_token}}<start_of_turn>user\\n{{talimat}}\\n\\n"
              f"Metin: {{metin}}<end_of_turn>\\n<start_of_turn>model\\n")
    girdi = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        cikti = model.generate(**girdi, max_new_tokens=512, do_sample=False,
                               eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                               pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(cikti[0, girdi["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print(maskele("müşteri Ayşe Yılmaz tc 12345678901 tel 0532 111 22 33"))
# -> müşteri [AD] tc [TCKN] tel [TEL]
```

## Sınırlar
Değerlendirme sentetik veriyle yapılmıştır; gerçek kullanıcı metninde ölçmeden üretime alınmamalıdır. Şema dışı etiket
üretebilir (53 etiketlik whitelist önerilir). KVKK/GDPR sorumluluğu kullanandadır. Lisans: Gemma Terms of Use.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True, help="user/name")
    ap.add_argument("--results", required=True, help="results/<run>.json from eval.py")
    ap.add_argument("--train_desc", default="sentetik veri, 1 epoch, full fine-tune, lr 5e-5, bf16")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    print("logged in as:", whoami()["name"])
    r = json.load(open(args.results))
    by_kat = " · ".join(f"{k} {v:.3f}" for k, v in r["by_kategori"].items())
    by_oz = " · ".join(f"{k} {v['acc']:.3f}" for k, v in r["by_ozellik"].items())
    card = CARD.format(repo_name=args.repo, exact=f"{r['exact_match']:.3f}", schema=f"{r['schema_neutral']:.3f}",
                       by_kat=by_kat, by_oz=by_oz, train_desc=args.train_desc)
    with open(os.path.join(args.model, "README.md"), "w", encoding="utf-8") as f: f.write(card)
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.model, repo_id=args.repo, repo_type="model",
                      ignore_patterns=["checkpoint-*", "*.log", "train_args.json"])
    print("uploaded -> https://huggingface.co/" + args.repo)

if __name__ == "__main__":
    main()

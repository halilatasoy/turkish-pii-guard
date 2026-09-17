# pii-guard-tr — Türkçe PII maskeleme modelini iyileştirme

Hedef: `cagrigungor/pii-guard-turkish-270m` (Gemma-3-270m-it tabanlı, 53 etiket, talimat-koşullu maskeleme)
modelinin `cagrigungor/turkish-pii-masking-benchmark` (1000 satır, tam eşleşme) skorunu yükseltmek.

## Baseline (yerelde yeniden üretildi)

| Metrik | Model kartı | Bizim ölçüm |
|---|---|---|
| Tam eşleşme (1000) | 0.740 | 0.743 |
| Şema-nötr (903) | 0.770 | 0.773 |

Kategori: tam 0.729 · beyaz liste 0.620 · kara liste 0.747 · kapsam dışı 0.730 · negatif 0.895
Zor dilimler: caps 0.514 · uzun 0.521 · sözle 0.649 · çok kişi 0.450 · olmayan etiket talebi 0.604

## Yaklaşım

Orijinal eğitim verisi (~400k) paylaşılmamış. Bu yüzden kendi sentetik üretecimizi yazdık ve modeli
yayınlanmış checkpoint'ten devam ettirerek (continued full fine-tune) eğitiyoruz.

`scripts/gen_data.py` zayıf dilimleri kasıtlı olarak ağırlıklandırır:
- %22 BÜYÜK HARF, %18 sözle yazılmış rakam, %17 uzun (3–6 cümle), %25 çok alanlı kayıt, %9 çok kişili, %11 ek almış PII
- beyaz liste / kara liste / kapsam dışı / grup politikaları (özel nitelikli, finansal, iletişim...) ve karışık
  talepler ("istenen etiket metinde yok")
- 200+ tuzaklı negatif (fatura no, vade tarihi, şehir adlı şube, "iban alanı boş bırakılamaz" gibi)
- `[YAS] [CINSIYET] [UYRUK]` dahil 53 etiketin tamamı

**Kirlilik önlemi:** cümle kalıpları, tuzak cümleler ve beyaz/kara liste talimat kalıpları benchmark'takilerden
farklı yazıldı; benchmark girdileriyle birebir çakışan satırlar üretimde atılır. Benchmark yalnızca test için kullanılır.

## Dosyalar

```
scripts/gen_data.py      sentetik veri üreteci (JSONL)
scripts/train.py         completion-only SFT (full FT veya --lora_r ile LoRA), CUDA/MPS
scripts/eval.py          benchmark değerlendirme (kategori + dilim kırılımı, JSON rapor)
scripts/error_report.py  iki sonuç dosyasını satır satır karşılaştırır
run_gpu.sh               CUDA makinede uçtan uca pipeline
data/benchmark_1000.csv  benchmark (sadece test)
data/train.jsonl         300k eğitim, data/val.jsonl 4k doğrulama
results/                 baseline_270m.{csv,json} ve sonraki koşular
```

## Çalıştırma (CUDA)

```bash
bash run_gpu.sh v1
```
Varsayılan: 300k örnek, 2 epoch, lr 5e-5, batch 32, bf16, 768 token. Tek A100/H100'de ~1–1.5 saat.

Ek denemeler:
```bash
python scripts/train.py --model google/gemma-3-270m-it --out outputs/base_v1 --epochs 2 --lr 1e-4   # tabandan
python scripts/train.py --model models/pii-guard-turkish-270m --out outputs/lora_v1 --lora_r 32 --lr 2e-4
python scripts/eval.py --model outputs/base_v1/final --out results/base_v1.csv --batch 64
```

## Notlar
- Prompt biçimi yayınlanan modelle birebir aynı: `<bos><start_of_turn>user\n{talimat}\n\nMetin: {metin}<end_of_turn>\n<start_of_turn>model\n`
- transformers 5.x'in Gemma tokenizer için verdiği "incorrect regex" uyarısı yanlış alarm; varsayılan yükleme SentencePiece ile birebir aynı tokenizasyonu veriyor (doğrulandı).
- Apple MPS'te batch × seq × 262k vocab logits tensörü INT_MAX'ı aşmamalı: bsz ≤ 8 @ 768 token.

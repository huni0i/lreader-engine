# Japanese and English research datasets

Do not put Naver Webtoon or other commercial webtoon pages here.

Japanese real manga pages are almost all gated (Manga109). English Golden Age scans are public domain, so we collect those immediately and keep Japanese coverage on synthetic pages plus Manga109 after approval.

| Dataset | Language | License | How we get it |
|---|---|---|---|
| `synthetic-ja-en` | ja, en | CC0 | Generated locally |
| DCM772 annotations | en | Public-domain source comics + research GT | Git clone |
| Internet Archive PD comics | en | Public domain | Archive.org, DCM-matching titles |
| COMICS (CVPR 2017) | en | Public-domain scans + UMD annotations | OCR CSV + textbox zip |
| CoMix tiny | en | CC0 / public-domain scans | Hugging Face train/test/validation pages |
| COO annotations | ja | Research GT; images need Manga109 | Git clone annotations only |
| Manga109 / Manga109-s | ja | Author permission, no redistribution | Hugging Face access request |
| eBDtheque | en, ja, fr | Research registration | University of La Rochelle form |

Prepare everything that can be fetched without a manual approval:

```bash
cd server
python3 scripts/prepare_datasets.py
```

Images stay in `server/data/`, which is gitignored.

After Manga109 access is granted:

```bash
huggingface-cli login
huggingface-cli download hal-utokyo/Manga109 --repo-type dataset --local-dir data/manga109
```

Then pair COO polygons with `data/manga109/images` as described in `data/coo/APPLY.txt`.

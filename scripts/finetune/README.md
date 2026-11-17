# Fine-tuning experiment (one-off)

Train a LoRA adapter on your own writing with Apple MLX, write five test answers with it, and scan them with Pangram.
Every script is a dry run unless you add `--run` (or `--trial`/`--setup` for train.py). All files go under
`data/finetune/` (gitignored); use `--work-dir` to move them.

Run these from the repo root, in order:

1. `python3 scripts/finetune/prepare.py` then `... --run`
   Filters your Gmail, Google Chat and document samples, splits long ones into chunks of at most 400 words, asks a
   local Ollama model (`gemma4:12b`, change with `--notes-model`) for content notes, and writes
   `dataset/train.jsonl`, `valid.jsonl`, `test.jsonl` and `manifest.json`. Notes are cached in `notes.jsonl`, so you can
   stop and resume; `--limit N` does only N chunks. Cost: free (local), about 8 seconds per chunk.
2. `python3 scripts/finetune/train.py --setup`, then `--trial`, then `--run`
   Creates `.venv` with `mlx-lm==0.32.2`, times 40 iterations and estimates the full run, then trains into `adapter/`
   (log in `train.log`). Cost: free. **The first `--trial` or `--run` downloads the base model
   `mlx-community/gemma-4-12B-it-4bit`, about 6.8 GB from Hugging Face.**
3. `python3 scripts/finetune/generate.py --run`
   Makes notes for your latest five rewrite inputs (or `--file PATH`, repeatable), writes an answer for each with the
   adapter (`--baseline` adds the plain base model) and saves `outputs/<UTC stamp>.json`. Prints a metrics table; add
   `--show` to print the texts. Cost: free (local).
4. `python3 scripts/finetune/scan.py` then `... --run`
   Scans the newest outputs file with Pangram and prints PASS/FAIL per answer. **This is the only paid step**; the dry
   run shows the estimated cost. Results are stored in your VoicePrint database as `finetune: <draft source>`.

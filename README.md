# VoicePrint

Zero-inference clipboard rewriter and adaptive personal **Voice DNA** studio for macOS.

VoicePrint reads text from your clipboard, strips synthetic AI patterns (clichés, robotic transitions, em-dashes, uniform sentence cadence), rewrites it to match your personal writing style across different contexts, and learns directly from your final tweaks over time.

## Download & Install on macOS (Mac Studio / Any Mac)

1. Go to the **[Releases](https://github.com/casamensauk/voiceprint/releases)** page and download **`VoicePrint-macOS.dmg`** (or **`VoicePrint-macOS.zip`**).
2. Open the `.dmg` and drag **`VoicePrint.app`** into **`/Applications`**.
3. **First launch:** Right-click **`VoicePrint.app`** → **Open** (since the bundle is unsigned, macOS Gatekeeper asks once).
4. **Global Hotkey:** On launch, VoicePrint automatically installs the Quick Action into `~/Library/Services/`. Bind a shortcut (such as `⌘⇧V`) under:
   **System Settings → Keyboard → Keyboard Shortcuts → Services → General → VoicePrint: Rewrite Clipboard in My Voice**.

> **Note:** VoicePrint uses macOS system `python3` (`python3 --version`, e.g., Xcode Command Line Tools or Homebrew) with zero external pip dependencies. Your personal Voice DNA database is stored locally at `~/Library/Application Support/VoicePrint/voiceprint.db`.

## Features

- **Clipboard Studio + Live AI-Tell & Burstiness Scoring**: Inspects AI vocabulary, robotic transitions, structural templates, and sentence-length variance (burstiness) before and after rewriting.
- **In-House 3-Stage AI Writing Detector (<$0.001 / 1k words via OpenRouter + $0 Offline Forensics)**: Combines Stage 1 deterministic stylometrics (burstiness, MATTR, contraction ratio, zero-width & Cyrillic/Greek homoglyph spoofing), Stage 2 `DNA-GPT` prefix-truncation regeneration `BScore` (`meta-llama/llama-3.3-70b-instruct` + `google/gemini-2.5-flash`), Stage 3 structured JSON arbitration, `<300`-word & `1%–19%*` confidence calibration, SHA-256 SQLite caching, and bibliography exclusion — matching Pangram v4.1's interactive highlighted window schema.
- **AI & Originality Inspector (Pangram Labs Pay-As-You-Go)**: Optional direct integration with Pangram `pangram-4` (`is_humanized` / `humanizer_score`), `.docx`/`.pdf`/`.rtf` uploads, and web plagiarism checks.
- **Source & Revision Coach**: Side-by-side local citation overlap checker (flags unquoted verbatim spans and close paraphrases while recognizing properly cited quotes) and an author-led Academic Revision Coach for passive voice, unattributed claims, vague filler, and repetitive openers.
- **Grow With Me (Final-Edit Learning)**: Tweak any draft before copying and VoicePrint diffs your edit, saves it as a `2.5x` weighted gold sample, and extracts word/phrasing swap rules automatically.
- **Bulk Importer**: Import WhatsApp exports, Discord/Slack logs, emails, CSVs, and plain `.txt`/`.md` documents across 4 context modes (`Casual Chat`, `Work Email`, `Long-Form`, `Social / DM`).
- **Flexible AI Backend**: Works completely offline via its built-in deterministic De-AI & Voice DNA engine, or pairs with local **Ollama** or **Gemini / OpenAI / Anthropic / OpenRouter** APIs.

## Run From Source

```bash
python3 server.py --auto-port
# Starts on http://127.0.0.1:18940 (automatically selects a free port if occupied)
```

Run automated tests:

```bash
python3 tests/test_engine.py
```

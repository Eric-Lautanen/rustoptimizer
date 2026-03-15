# optimizer

> **proof of concept** — a local LLM editing your files, live, on a budget laptop. because why not.

---

## what is this

A GUI tool that hooks up [LM Studio](https://lmstudio.ai/) to your files and has a local AI model analyze, suggest, and *animate* edits directly into your code — character by character, hacker-movie style.

Works on any text-based file. Code, markdown, config files, prose, recipes. If it's UTF-8 it's fair game.

No cloud. No API keys. No monthly bill. Just your machine, a small language model, and questionable life choices.

---

## the hardware it was built and runs on

**ASUS Zenbook 14 OLED**
- CPU: AMD Ryzen 7 (integrated graphics, no discrete GPU)
- RAM: 16 GB
- Storage: NVMe SSD
- OS: Windows 11

That's it. No beefy workstation, no NVIDIA RTX, no water cooling. A normal thin-and-light laptop that costs less than a used car payment.

If your machine is roughly in this ballpark — or better — this will work for you.

---

## step 1 — install LM Studio

LM Studio runs the AI model locally on your machine. Current version as of March 2026: **0.4.6**.

**Download:** https://lmstudio.ai/download

### Windows
1. Download the `.exe` installer
2. Run it and follow the wizard
3. Windows may show a security warning — confirm to proceed

### macOS (Apple Silicon required)
1. Download the `.dmg`
2. Drag LM Studio to your Applications folder
3. Launch from Applications

### Linux
1. Download the `.AppImage`
2. Make it executable: `chmod +x lmstudio.AppImage`
3. Run it: `./lmstudio.AppImage`

---

## step 2 — download a model

1. Open LM Studio and go to the **Discover** tab
2. Search for a model (see recommendations below)
3. Pick a quantization — **Q4_K_M** is the sweet spot for quality vs. speed
4. Click Download and wait

### recommended models for 16 GB RAM

| Model | Size | Notes |
|---|---|---|
| Qwen2.5-Coder 4B | ~3 GB | Fast, great at code, the default pick |
| Qwen3 4B | ~3 GB | Stronger reasoning, slightly slower |
| Qwen3-Coder 7B | ~5 GB | Best code quality if you have headroom |
| DeepSeek-R1 8B distill | ~5 GB | Excellent for debugging and tricky logic |
| Llama 3.3 8B | ~5 GB | Solid all-rounder |

### for 8 GB RAM
Stick to 3B–4B models (Qwen 4B, Phi-4-mini). Anything larger will be very slow or won't fit.

Smaller models get flaky with the JSON schema this app requires. If responses keep failing, try a larger or smarter model.

---

## step 3 — start the LM Studio server

This app talks to LM Studio over a local HTTP API on port `1234`.

### via the GUI
1. Go to **Settings → Developer** and turn on **Developer Mode**
2. Open the **Developer** tab in the sidebar
3. Toggle **Start server** to ON
4. The server is now running at `http://127.0.0.1:1234`

### via the CLI (optional)
```bash
lms server start          # start on default port 1234
lms server start --port 5000  # or a custom port
lms server status         # check if it's running
lms server stop           # shut it down
```

For headless / always-on setups (Linux):
```bash
lms daemon up    # start without the GUI
lms daemon down  # stop it
lms log          # stream live logs
```

### load the model in LM Studio
Before hitting Start in this app, make sure your model is loaded in LM Studio:
- Press `Ctrl+L` (Windows/Linux) or `Cmd+L` (macOS)
- Select your downloaded model and click Load
- Wait for it to finish — this can take a minute for larger models

Or skip this step and use the **Load** button inside this app — it will call the LM Studio API to load the model for you.

---

## step 4 — run this app

```bash
git clone <this repo>
cd rustoptimizer
cargo run --release
```

1. The app opens and connects to LM Studio automatically
2. Pick your loaded model from the dropdown (or load one with the Load button)
3. Click **Open Folder** and select the folder you want to process
4. Optionally edit the **PROMPT** to tell the model what to do
5. Hit **▶ Start**

That's genuinely all of it.

---

## what it actually does

1. **Connects to LM Studio** running locally on `127.0.0.1:1234`
2. **Scans a folder** and queues every readable text file
3. **Sends each file to the LLM** with a structured prompt asking for targeted suggestions in JSON
4. **Streams the response live** to a terminal panel with real-time token counts, tok/s rate, and timing stats
5. **Pipelines requests** — as soon as a response arrives, the next file is sent to the LLM immediately, so prompting and animation run in parallel
6. **Animates the edits** into the code editor — deleting old text, typing in the new — like someone's actually in there
7. **Diffuses between files** with a noise-character transition effect when moving to the next file
8. **Writes the changes to disk** and runs `cargo check` asynchronously if Rust files are present
9. **Re-queues broken files** for up to 3 rounds of auto-fix if the compiler complains
10. **Retries failed files** up to 2 times before skipping, without stalling the rest of the queue
11. **Removes individual files** from the queue with the `-` button before or after processing
12. **Ejects the loaded model** from LM Studio automatically on app exit so it doesn't sit in RAM when you're done
13. **Drops ASCII art** in the terminal after each successful edit, because productivity tools need more personality

---

## tuning

Everything worth tweaking lives in the `// --- CONFIGURATION ---` block at the top of `src/main.rs`:

```rust
const LM_STUDIO_BASE: &str = "http://127.0.0.1:1234"; // change if LM Studio is on another port
const DEFAULT_MODEL: &str   = "";                       // pre-selected model on startup
const DEFAULT_CONTEXT: u32  = 30_000;                  // default context window (tokens)
const MAX_FIX_ROUNDS: u32   = 3;                       // cargo check + fix cycles before giving up
const MAX_FILE_RETRIES: u8  = 2;                       // per-file LLM retry attempts before skipping

// animation speed
const DELETE_INTERVAL_MS: u64 = 10;  // ms per character deleted
const TYPE_INTERVAL_MS:   u64 = 10;  // ms per character typed
const TRANSITION_MS:      u64 = 500; // file-switch diffusion effect duration (ms)

// network timeouts
const TIMEOUT_MODEL_LIST_SECS:   u64 = 10;  // fetching the model list
const TIMEOUT_MODEL_UNLOAD_SECS: u64 = 60;  // ejecting a model
const TIMEOUT_MODEL_LOAD_SECS:   u64 = 300; // loading a model (large ones take time)
const TIMEOUT_LLM_CONNECT_SECS:  u64 = 30;  // initial connection to LM Studio
const TIMEOUT_LLM_IDLE_SECS:     u64 = 300; // max silence between stream chunks before giving up
const TIMEOUT_ANALYZING_SECS:    u64 = 720; // watchdog: force-fail if a file gets stuck this long
```

The **PROMPT** field in the sidebar lets you change the analysis instruction per session without touching the code. Default: `Analyze the following file and identify specific improvements.`

The streaming timeout (`TIMEOUT_LLM_IDLE_SECS`) is an *idle* timeout — it resets on every received chunk, so long responses on slow hardware will never time out as long as tokens keep arriving. Only fires if the model completely stalls.

---

## what this is not

- Production-ready (it's a PoC, see title)
- A replacement for `clippy`, `rustfmt`, or an actual editor
- Guaranteed to make your files better (the LLM has opinions)
- Going to work offline if LM Studio isn't running

---

## the fun part

The whole point was to see how far you could push local LLM-assisted tooling on commodity hardware. Turns out: pretty far. The Ryzen 7 handles 4B parameter models at a usable speed (~15–25 tok/s), the egui UI stays responsive during inference, and the streaming terminal actually makes waiting for the LLM feel kinda cool instead of just... waiting.

Point it at a Rust project and it'll refactor your code. Point it at markdown and it'll clean up your docs. Point it at a text file full of recipes and it'll have opinions about your cooking. The model doesn't care — it'll try to improve whatever you throw at it.

Is it as good as Claude doing the same thing? No. Is it free, private, and runs on a laptop you already own? Also yes.

---

## built with

- [egui](https://github.com/emilk/egui) / [eframe](https://github.com/emilk/egui/tree/master/crates/eframe) — immediate-mode GUI
- [tokio](https://tokio.rs/) — async runtime with async process and time support
- [reqwest](https://github.com/seanmonstar/reqwest) — HTTP client for LM Studio API (SSE streaming)
- [LM Studio](https://lmstudio.ai/) — local LLM server
- Rust, obviously

---

*made for fun, runs on a Zenbook, ships ASCII art to a terminal. what more do you want.*

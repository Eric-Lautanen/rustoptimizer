# rust optimizer

> **proof of concept** — a local LLM refactoring your Rust code, live, on a budget laptop. because why not.

---

## what is this

A GUI tool that hooks up [LM Studio](https://lmstudio.ai/) to your Rust source files and has a local AI model analyze, suggest, and *animate* refactors directly into your code — character by character, hacker-movie style.

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

## what it actually does

1. **Connects to LM Studio** running locally on `127.0.0.1:1234`
2. **Scans a Rust project** folder and queues every `.rs` file
3. **Sends each file to the LLM** with a structured prompt asking for targeted refactor suggestions in JSON
4. **Streams the response live** to a terminal panel with real-time token counts, tok/s rate, and timing stats
5. **Pipelines requests** — as soon as a response arrives, the next file is sent to the LLM immediately, so prompting and animation run in parallel
6. **Animates the edits** into the code editor — deleting old text, typing in the new — like someone's actually in there
7. **Diffuses between files** with a noise-character transition effect when moving to the next file
8. **Writes the changes to disk** and runs `cargo check` asynchronously to catch any LLM-introduced mistakes
9. **Re-queues broken files** for up to 3 rounds of auto-fix if the compiler complains
10. **Retries failed files** up to 2 times before skipping, without stalling the rest of the queue
11. **Ejects the loaded model** from LM Studio automatically on app exit so it doesn't sit in RAM when you're done
12. **Drops ASCII art** in the terminal after each successful edit, because productivity tools need more personality

---

## recommended models

Anything that can follow JSON schema instructions. Tested with:

- `Qwen2.5-Coder 4B` — fast, good at Rust, fits comfortably in 16 GB alongside the OS (~15–25 tok/s)
- `Qwen3 4B` — solid reasoning, slightly slower
- `9B models` — work fine, just slower on integrated graphics

Smaller models get flaky with the JSON schema. Larger models will work if you have the RAM budget.

---

## setup

1. Install [LM Studio](https://lmstudio.ai/) and download a model
2. Start the local server in LM Studio (default port 1234)
3. Clone this repo
4. `cargo run --release`
5. Pick a model in the UI, load it, open a Rust folder, hit Start

That's genuinely all of it.

---

## tuning

Everything worth tweaking lives in the `// --- CONFIGURATION ---` block at the top of `src/main.rs`:

```rust
const LM_STUDIO_BASE: &str = "http://127.0.0.1:1234"; // change if LM Studio is on another port
const DEFAULT_MODEL: &str   = "qwen3.5-4b";            // pre-selected model on startup
const DEFAULT_CONTEXT: u32  = 30_000;                  // default context window (tokens)
const MAX_FIX_ROUNDS: u32   = 3;                       // cargo check + fix cycles before giving up
const MAX_FILE_RETRIES: u8  = 2;                       // per-file LLM retry attempts before skipping

// animation speed
const DELETE_INTERVAL_MS: u64 = 30; // ms per character deleted
const TYPE_INTERVAL_MS:   u64 = 30; // ms per character typed
const TRANSITION_MS:      u64 = 500; // file-switch diffusion effect duration (ms)

// network timeouts
const TIMEOUT_MODEL_LIST_SECS:   u64 = 10;  // fetching the model list
const TIMEOUT_MODEL_UNLOAD_SECS: u64 = 60;  // ejecting a model
const TIMEOUT_MODEL_LOAD_SECS:   u64 = 300; // loading a model (large ones take time)
const TIMEOUT_LLM_CONNECT_SECS:  u64 = 30;  // initial connection to LM Studio
const TIMEOUT_LLM_IDLE_SECS:     u64 = 300; // max silence between stream chunks before giving up
const TIMEOUT_ANALYZING_SECS:    u64 = 720; // watchdog: force-fail if a file gets stuck this long
```

The streaming timeout (`TIMEOUT_LLM_IDLE_SECS`) is an *idle* timeout — it resets on every received chunk, so long responses on slow hardware will never time out as long as tokens keep arriving. Only fires if the model completely stalls.

---

## what this is not

- Production-ready (it's a PoC, see title)
- A replacement for `clippy` or `rustfmt`
- Guaranteed to make your code better (the LLM has opinions)
- Going to work offline if LM Studio isn't running

---

## the fun part

The whole point was to see how far you could push local LLM-assisted tooling on commodity hardware. Turns out: pretty far. The Ryzen 7 handles 4B parameter models at a usable speed (~15–25 tok/s), the egui UI stays responsive during inference, and the streaming terminal actually makes waiting for the LLM feel kinda cool instead of just... waiting.

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

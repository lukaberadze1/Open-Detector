# Open Detector: remote test run

Thanks for running this. It takes about 15 minutes of setup and then runs on its own for 1 to 2 hours.
At the end there is **one file to send back: `remote_test_results.zip`**.

If you use Claude Code, open this folder in it and say: *"Read START_HERE.md and run the tests for me."*
(Claude: follow the steps below in order, don't change any code, and if a step fails, show the error and
fix only the setup, such as a missing tool or package.)

## What this is

Open Detector is an AI-text detector. Its owner also built a generator that tries to write text the
detector reads as human, in the owner's own writing style, so the detector can be trained on its
hardest cases. Everything the generator writes is labelled as AI text.

The owner's 8 GB Mac can only run the small ("slim") detector. A new trick, revising only the passages
the detector flags, got 15 of 18 generated samples past the slim detector. Your 16 GB machine can run
the **full** detector (three engines), which is the real question: **does the trick still work against
the full detector?**

## What gets tested

| Step | What | Needs | Time |
|---|---|---|---|
| 1 | The project's test suite (no models) | nothing | 1 min |
| 2 | The 18 revised samples scored by the slim detector: should match the Mac's scores | ~2 GB download | 5-10 min |
| 3 | The same 18 samples scored by the full detector | ~3 GB more download | 10-20 min |
| 4 | 18 new samples written by Gemini and revised against the full detector | a free Gemini key | 1-2 hours |

Steps 1-3 need no account. Step 4 needs a Gemini API key and uses about 120 requests of its free
daily allowance. If a step stops, run the script again: finished steps are skipped.

## What it needs

- Windows 10/11 (or macOS/Linux), 16 GB RAM, about **10 GB free disk** (models and packages).
- An internet connection. Close heavy programs while it runs.
- **uv**, a Python installer. It brings its own Python, so you don't need to install Python.

## Steps (Windows)

1. **Unzip** the package somewhere simple, such as `C:\open-detector-test`. Avoid OneDrive folders.
2. **Install uv.** Open PowerShell and run:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Then **close PowerShell and open a new window** so it finds `uv`.
3. **For step 4 only: add a Gemini key.** Get a free key at https://aistudio.google.com/apikey (Google
   account, "Create API key"). In the unzipped folder, copy `.env.example` to a new file named `.env`
   and fill in the line `GEMINI_API_KEY=` with your key. Skip this to run steps 1-3 only.
4. **Run it:**
   ```powershell
   cd C:\open-detector-test
   powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
   ```
   Add `-NoGenerate` at the end to run steps 1-3 only. Progress prints as it goes and is also saved in
   `remote_test_results\run_log.txt`.
5. **Send back** `remote_test_results.zip` from the same folder.

On macOS or Linux, step 2 is `curl -LsSf https://astral.sh/uv/install.sh | sh` and step 4 is
`bash run_mac_linux.sh` (add `--no-generate` for steps 1-3 only).

### Optional: an NVIDIA graphics card makes steps 2-4 much faster

The standard install runs on the processor. With an NVIDIA card, after the first run has installed
everything (or after step 4's first attempt), run this once and then run the script again:
```powershell
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124 --reinstall
```
Leave this out if you're not sure: the processor works, just more slowly.

## If something goes wrong

- **`uv` not found:** open a new PowerShell window after installing it.
- **"running scripts is disabled":** use the exact `powershell -ExecutionPolicy Bypass -File ...` line.
- **Out of memory or very slow:** close other programs and run again; finished steps are kept.
- **Step 4 says "high demand" (HTTP 503):** Google's model is busy. The script waits 10 minutes and tries
  again, up to 4 times, then switches to a second model, so leave it running.
- **Step 4 says "out of quota":** the free key's daily limit was reached. Run again tomorrow, or send
  back what's there: the zip is written either way.
- **Anything else:** send back `remote_test_results\run_log.txt` and the error message.

## Privacy

The package holds the owner's style profile and some of their own writing (`data/styles/`), shared for
this test only. Please don't share or upload it, and delete the folder when you're done. Your Gemini key
stays in `.env` on your computer and is **not** in the results zip.

# Open Detector: built-in human profile test

Thanks for running this. Setup takes about 15 minutes, and then it runs on its own for 1 to 3 hours. At the end
there is **one file to send back: `human_test_results.zip`**.

If you use Claude Code, open this folder in it and say: *"Read START_HUMAN.md and run the test for me."*
(Claude: follow the steps below, don't change any code, and if a step fails, show the error and fix only the
setup, such as a missing tool or package.)

## What this is

Open Detector is an AI-text detector. Its owner also has a generator that tries to write text the detector reads
as human, so the detector can be trained on its hardest cases. Everything the generator writes is labelled as AI
text. This run rewrites nine AI-written documents with the generator's built-in human profile, in three setups, and
measures how the detector scores them.

## What it needs

- Windows, macOS or Linux, about **10 GB free disk**, and an internet connection.
- 16 GB of memory for the full detector. With less, add `-NoFull` (Windows) or `--no-full` (macOS, Linux).
- **Your own free Gemini API key.** The run uses up to your whole free daily allowance for
  `gemini-3.5-flash-lite` (about 500 requests), and nothing else. It stays in your `.env` and never goes in the
  results. Check you are happy for it to be used this way before you start.
- **uv**, a Python installer that brings its own Python.

## Steps on Windows

Use **PowerShell** for all of these, not Command Prompt.

1. **Unzip** this to `C:\open-detector-test`, not a OneDrive folder. Windows may put a second folder inside: the
   right folder is the one holding `START_HUMAN.md`. Go there with `cd`, for example:
   ```powershell
   cd C:\open-detector-test\open-detector-test
   ```
2. **Install uv**, then close PowerShell, open a new window and `cd` back:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. **Add your key.** Get one at https://aistudio.google.com/apikey, then run these and fill in the
   `GEMINI_API_KEY=` line in Notepad, keeping the file name `.env` (not `.env.txt`):
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
4. **Try it on one document first.** This downloads the models, so it can take 15 to 30 minutes:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_human_windows.ps1 -TryOne
   ```
   If it fails, send the error before going on.
5. **Run all of it.** What the trial finished is skipped:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_human_windows.ps1
   ```
6. **Send back** `human_test_results.zip` from the same folder. If it says it is out of quota, send what it has.

Keep the computer awake and plugged in while it runs. If it stops, run the same command again.

On macOS or Linux, use `curl -LsSf https://astral.sh/uv/install.sh | sh` and `bash run_human_mac_linux.sh`.

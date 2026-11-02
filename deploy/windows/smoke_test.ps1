# Checks the built installer on a real Windows machine, as someone downloading it would use it: installs it,
# scores a text with the installed app's own Python, opens the app, and checks the server stops with it.
# .github/workflows/windows-app.yml runs this. What it saves for a person to look at goes in dist\smoke.
param([string]$Installer = "dist\Open-Detector-Setup.exe")
$ErrorActionPreference = "Stop"
$out = New-Item -ItemType Directory -Force "dist\smoke"
$installed = "$env:LOCALAPPDATA\Programs\Open Detector"
$exe = "$installed\Open Detector.exe"
$data = "$env:LOCALAPPDATA\Open Detector"

function Save-Screen($name) {
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $image = New-Object System.Drawing.Bitmap $screen.Width, $screen.Height
    $graphics = [System.Drawing.Graphics]::FromImage($image)
    $graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
    $image.Save("$($out.FullName)\$name.png")
}

function Server-Answers {
    try { (Invoke-WebRequest "http://127.0.0.1:8517/_stcore/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 }
    catch { $false }
}

try {
    Write-Host "Installing"
    $setup = Start-Process $Installer -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
    if ($setup.ExitCode -ne 0) { throw "The installer exited with $($setup.ExitCode)" }
    if (-not (Test-Path $exe)) { throw "The installer left no $exe" }
    if (-not (Test-Path "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Open Detector.lnk")) {
        throw "The installer left no Start menu shortcut"
    }

    Write-Host "Scoring a text with the installed app's Python (downloads the Slim model)"
    # The same text scored 90% on the Mac's Slim build on 2026-11-02.
    $sample = "$($out.FullName)\sample.txt"
    Set-Content $sample -Encoding UTF8 -Value @"
The old harbour was quiet by the time we got there, mostly because the fishing boats had already gone out and the cafes had not bothered to open. My brother wanted coffee anyway, so we walked along the wall until we found a kiosk run by a man who seemed annoyed that anyone was awake. He gave us two paper cups and told us the ferry was late again. Nobody seemed surprised. We sat on a stack of pallets and watched a gull fight with a plastic bag for about ten minutes, which was the most exciting thing that happened all morning. When the ferry finally came it was smaller than I remembered, and it smelled of diesel and wet rope. There were perhaps twelve of us on board, mainly old women with shopping trolleys and one teenager asleep against the rail. I had forgotten how slowly the island comes into view, first a smudge and then a line of white houses, and then all at once the church and the long ugly pier where my grandfather used to fish on Sundays.
"@
    $report = "$($out.FullName)\report.json"
    $scored = Start-Process $exe -WorkingDirectory "$installed\app" -Wait -PassThru -NoNewWindow `
        -ArgumentList "-m", "opendetector", "`"$sample`"", "--profile", "slim" `
        -RedirectStandardOutput $report -RedirectStandardError "$($out.FullName)\report.err"
    Get-Content "$($out.FullName)\report.err" -Tail 15
    if ($scored.ExitCode -ne 0) { throw "Scoring exited with $($scored.ExitCode)" }
    $probability = (Get-Content $report -Raw | ConvertFrom-Json).ai_probability
    Write-Host "AI likelihood: $probability"
    if ($probability -lt 0.85 -or $probability -gt 0.95) { throw "Expected about 0.90, as on the Mac; got $probability" }

    Write-Host "Opening the app"
    $app = Start-Process $exe -PassThru
    $deadline = (Get-Date).AddMinutes(5)
    while (-not (Server-Answers)) {
        if ((Get-Date) -gt $deadline) { throw "The app's server never answered" }
        if ($app.HasExited) { throw "The app closed by itself with $($app.ExitCode)" }
        Start-Sleep 2
    }
    Start-Sleep 30  # the page's first load
    Save-Screen "window"
    $app.Refresh()
    Write-Host "Window title: '$($app.MainWindowTitle)'"
    if ($app.MainWindowTitle -ne "Open Detector") { throw "No window titled Open Detector" }
    if (-not (Test-Path "$data\data\ui_prefs.json")) { throw "The app wrote nothing to $data" }

    Write-Host "Ending the app abruptly: its server must notice and stop"
    Stop-Process -Id $app.Id -Force
    $deadline = (Get-Date).AddSeconds(30)
    while (Server-Answers) {
        if ((Get-Date) -gt $deadline) { throw "The server outlived the app" }
        Start-Sleep 1
    }
    Write-Host "All good"
}
finally {
    Copy-Item "$data\Logs\*" $out.FullName -ErrorAction SilentlyContinue
    Get-Content "$data\Logs\launcher.log" -Tail 40 -ErrorAction SilentlyContinue
    Get-Content "$data\Logs\server.log" -Tail 40 -ErrorAction SilentlyContinue
    Save-Screen "last"
}

# Start the neural commentary voice worker (if set up) and then the API.
#   .\start_all.ps1            same as start_tts.ps1 followed by start_api.ps1
#   .\start_all.ps1 -Reload    passes -Reload to start_api.ps1
# start_api.ps1 itself is unchanged; the API would also launch the worker on its own
# (services/live_tts.py) once backend\tts_worker\venv exists.
param([switch]$Reload)

Set-Location $PSScriptRoot

Write-Host "=== Neural voices (TTS worker) ===" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "start_tts.ps1")
if ($LASTEXITCODE -ne 0) {
    Write-Warning "TTS worker setup/start failed - the live sim will fall back to browser voices. Continuing with the API."
}

Write-Host ""
Write-Host "=== API ===" -ForegroundColor Cyan
if ($Reload) { & (Join-Path $PSScriptRoot "start_api.ps1") -Reload }
else { & (Join-Path $PSScriptRoot "start_api.ps1") }

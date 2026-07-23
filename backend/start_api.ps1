# Run from repo: always start API from the backend folder so imports resolve.
param(
    # Opt-in only. Auto-reload wipes in-memory franchise sessions mid-sim whenever
    # backend/SimEngine .py files change (editor saves, agent edits, etc.).
    [switch]$Reload
)

Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Error "venv not found. Run: python -m venv venv ; .\venv\Scripts\Activate.ps1 ; pip install -r requirements.txt"
    exit 1
}

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

# Kill any stale uvicorn still bound to port 8000 (old API / old code in memory).
# Use taskkill — Stop-Process alone sometimes fails against uvicorn --reload parents.
$listeners = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
$killIds = @()
foreach ($c in $listeners) {
    if ($c.OwningProcess) { $killIds += [int]$c.OwningProcess }
}
$killIds = $killIds | Select-Object -Unique
foreach ($processId in $killIds) {
    Write-Host "Stopping stale process on port 8000 (PID $processId)..."
    & taskkill.exe /F /PID $processId /T 2>$null | Out-Null
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}
# Also stop orphaned uvicorn/python that still advertise franchise API in cmdline.
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match 'python|uvicorn' -and
        $_.CommandLine -match 'uvicorn|main:app|--port 8000'
    } |
    ForEach-Object {
        Write-Host "Stopping orphaned API process PID $($_.ProcessId)..."
        & taskkill.exe /F /PID $_.ProcessId /T 2>$null | Out-Null
    }
Start-Sleep -Seconds 2

$still = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($still) {
    Write-Warning "Port 8000 still in use after kill attempts. Close it manually, then re-run."
    $still | Format-Table -AutoSize
    exit 1
}

# Drop cached bytecode so a fresh process never serves an old .pyc snapshot.
Get-ChildItem -Path $PSScriptRoot -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Verifying live backend module..."
& $python -c @"
import services.franchise_sim as fs
checks = {
    'contract_bootstrap': hasattr(fs, '_ensure_league_roster_contracts'),
    'full_contract_office': hasattr(fs, '_build_free_agent_row'),
    'draft_board_v2': hasattr(fs, '_draft_stock_reason'),
    'lineup_persistence': hasattr(fs, 'save_franchise_lines'),
}
print('franchise_sim:', fs.__file__)
for k, v in checks.items():
    print(f'  {k}:', 'OK' if v else 'MISSING')
if not all(checks.values()):
    raise SystemExit('Wrong or incomplete franchise_sim — edit backend/services/franchise_sim.py, NOT SimEngine/app/sim_engine/franchise/')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Error "Backend module check failed. See message above."
    exit 1
}

$useReload = $Reload -or ($env:NHL_FRANCHISE_API_RELOAD -eq "1")

Write-Host ""
Write-Host "Starting API on http://127.0.0.1:8000"
Write-Host "Source of truth: backend/services/franchise_sim.py (NOT SimEngine/app/sim_engine/franchise/)"
Write-Host "After start, confirm: http://127.0.0.1:8000/api/health -> code.features all true"
if ($useReload) {
    Write-Host "Hot-reload: ON (sessions WILL reset when .py files change)"
} else {
    Write-Host "Hot-reload: OFF (stable for simming). Use -Reload or NHL_FRANCHISE_API_RELOAD=1 when developing."
}
Write-Host ""

$simEngine = Join-Path (Split-Path $PSScriptRoot -Parent) "SimEngine"
if ($useReload) {
    & $python -m uvicorn main:app --reload --reload-dir $PSScriptRoot --reload-dir $simEngine --host 127.0.0.1 --port 8000
} else {
    & $python -m uvicorn main:app --host 127.0.0.1 --port 8000
}

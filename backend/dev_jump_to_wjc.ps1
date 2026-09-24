# DEV ONLY — jump to World Juniors desk for UI testing (real draft-class prospects).
# Snapshot: backend/tmp/dev_wjc_test_save.pkl
# Delete that .pkl (and this script + dev_jump_to_wjc.py) when done.
#
# Usage (pick ONE path — depends on your current folder):
#   From repo root:  .\dev_jump_to_wjc.ps1 -Team "Ottawa" -Stage wjc_live
#   From repo root:  .\backend\dev_jump_to_wjc.ps1 ...
#   From backend\:   .\dev_jump_to_wjc.ps1 -Team "Ottawa" -Stage wjc_live
#   -LoadSnapshot / -DeleteSnapshot work from any of the above

param(
    [string]$Team = "Toronto",
    [ValidateSet("wjc_preview", "wjc_live", "wjc_mid", "wjc_final")]
    [string]$Stage = "wjc_live",
    [string]$Api = "http://127.0.0.1:8000",
    [switch]$LoadSnapshot,
    [switch]$DeleteSnapshot
)

if ($DeleteSnapshot) {
    Write-Host "DELETE $Api/api/dev/wjc-snapshot ..." -ForegroundColor Cyan
    try {
        $res = Invoke-RestMethod -Uri "$Api/api/dev/wjc-snapshot" -Method Delete -TimeoutSec 30
        Write-Host "deleted=$($res.deleted) path=$($res.path)" -ForegroundColor Green
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
    exit 0
}

if ($LoadSnapshot) {
    Write-Host "POST $Api/api/dev/load-wjc-snapshot ..." -ForegroundColor Cyan
    try {
        $res = Invoke-RestMethod -Uri "$Api/api/dev/load-wjc-snapshot" -Method Post -ContentType "application/json" -Body '{}' -TimeoutSec 120
    } catch {
        Write-Host "Request failed. Is the API running?" -ForegroundColor Red
        Write-Host $_.Exception.Message
        exit 1
    }
    Write-Host ""
    Write-Host "=== WJC SNAPSHOT LOADED ===" -ForegroundColor Green
    Write-Host "session_id: $($res.session_id)"
    Write-Host "real prospects: $($res.real_prospect_count)"
    Write-Host ""
    Write-Host "localStorage.setItem('nhl_franchise_session_id', '$($res.session_id)'); location.reload();" -ForegroundColor Yellow
    exit 0
}

$body = @{
    team_query              = $Team
    stage                   = $Stage
    head_coach_name         = "Dev Coach"
    coach_archetype         = "balanced"
    seed                    = 42
    persist_snapshot        = $true
    rebootstrap_dev_leagues = $true
} | ConvertTo-Json

Write-Host "POST $Api/api/dev/start-and-jump (team=$Team stage=$Stage) ..." -ForegroundColor Cyan

try {
    $res = Invoke-RestMethod -Uri "$Api/api/dev/start-and-jump" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
} catch {
    Write-Host "Request failed. Is the API running? Try: .\backend\start_api.ps1" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

Write-Host ""
Write-Host "=== DEV WJC UI READY ===" -ForegroundColor Green
Write-Host "session_id: $($res.session_id)"
Write-Host "stage:      $($res.stage)"
Write-Host "team:       $($res.user_team_id)"
Write-Host "wjc_day:    $($res.wjc_day)"
Write-Host "real prospects: $($res.real_prospect_count) / $($res.tournament_prospect_count)"
if ($res.snapshot_written) {
    Write-Host "snapshot:   $($res.snapshot_written)"
}
Write-Host ""
Write-Host "Paste in browser console, then reload:" -ForegroundColor Yellow
Write-Host "localStorage.setItem('nhl_franchise_session_id', '$($res.session_id)'); location.reload();"
Write-Host ""
Write-Host 'When done: .\backend\dev_jump_to_wjc.ps1 -DeleteSnapshot' -ForegroundColor DarkGray

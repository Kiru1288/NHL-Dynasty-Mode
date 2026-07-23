# DEV ONLY — jump to Draft Lottery / Combine / Entry Draft for UI testing.
# Delete alongside backend/dev_jump_to_draft.py when done.
#
# Usage:
#   .\backend\dev_jump_to_draft.ps1
#   .\backend\dev_jump_to_draft.ps1 -Team "Edmonton" -Stage draft
#   .\backend\dev_jump_to_draft.ps1 -Stage draft_lottery

param(
    [string]$Team = "Toronto",
    [ValidateSet("draft_lottery", "draft_combine", "draft")]
    [string]$Stage = "draft_combine",
    [string]$Api = "http://127.0.0.1:8000"
)

$body = @{
    team_query       = $Team
    stage            = $Stage
    head_coach_name  = "Dev Coach"
    coach_archetype  = "balanced"
    seed             = 42
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
Write-Host "=== DEV DRAFT UI READY ===" -ForegroundColor Green
Write-Host "session_id: $($res.session_id)"
Write-Host "stage:      $($res.offseason_stage)"
Write-Host ""
Write-Host "Paste in browser console on the franchise app, then reload:" -ForegroundColor Yellow
Write-Host "localStorage.setItem('nhl_franchise_session_id', '$($res.session_id)'); location.reload();"

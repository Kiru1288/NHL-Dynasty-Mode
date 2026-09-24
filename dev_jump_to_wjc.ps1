# Launcher — run from repo root: .\dev_jump_to_wjc.ps1 -Team Ottawa -Stage wjc_live
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir "backend\dev_jump_to_wjc.ps1") @args

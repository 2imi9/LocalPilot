# ============================================================
#  Push autoresearch/experiments/baseline → LocalPilot repo
#  Run once from the autoresearch folder.
# ============================================================

$ErrorActionPreference = "Stop"
$AutoresearchDir = $PSScriptRoot

Write-Host "`n=== Pushing to LocalPilot ===" -ForegroundColor Cyan

Set-Location $AutoresearchDir

# 1. Add LocalPilot as a second remote (skip if already exists)
$remotes = git remote
if ($remotes -notcontains "localpilot") {
    git remote add localpilot https://github.com/2imi9/LocalPilot.git
    Write-Host "[1/3] Remote 'localpilot' added" -ForegroundColor Green
} else {
    Write-Host "[1/3] Remote 'localpilot' already exists — skipping" -ForegroundColor Yellow
}

# 2. Push current branch (experiments/baseline) as 'main' to LocalPilot
Write-Host "[2/3] Pushing experiments/baseline → LocalPilot/main ..."
git push localpilot experiments/baseline:main
Write-Host "[2/3] Push complete" -ForegroundColor Green

# 3. Verify
Write-Host "[3/3] Verifying remote..." -ForegroundColor Green
git ls-remote localpilot

Write-Host "`n=== Done! ===" -ForegroundColor Cyan
Write-Host "Your LocalPilot repo is live at: https://github.com/2imi9/LocalPilot"

# ============================================================
#  LocalPilot — New Project Setup Script
#  Usage: .\new_project.ps1 -Name "MyResearch" -Dest "C:\Projects"
#  Creates a clean LocalPilot project ready to run experiments.
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Name,

    [Parameter(Mandatory=$false)]
    [string]$Dest = "$HOME\Desktop\Github"
)

$Source = $PSScriptRoot
$Target = Join-Path $Dest $Name

Write-Host "`n=== LocalPilot: Creating new project '$Name' ===" -ForegroundColor Cyan
Write-Host "Source: $Source"
Write-Host "Target: $Target`n"

# ── 1. Create directory ──────────────────────────────────────
if (Test-Path $Target) {
    Write-Error "Directory already exists: $Target"
    exit 1
}
New-Item -ItemType Directory -Path $Target | Out-Null
Write-Host "[1/6] Created directory: $Target" -ForegroundColor Green

# ── 2. Copy core files ───────────────────────────────────────
$CoreFiles = @(
    "prepare.py",
    "train.py",
    "program.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "README.md",
    "analyze.py",
    "browse.py",
    "run_experiments.py"
)

foreach ($f in $CoreFiles) {
    $src = Join-Path $Source $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $Target $f)
        Write-Host "  Copied: $f"
    } else {
        Write-Host "  Skipped (not found): $f" -ForegroundColor Yellow
    }
}
Write-Host "[2/6] Core files copied" -ForegroundColor Green

# ── 3. Copy models directory (MolmoWeb only, not devstral) ───
$ModelsDir = Join-Path $Target "models"
New-Item -ItemType Directory -Path $ModelsDir | Out-Null

$MolmoSrc = Join-Path $Source "models\MolmoWeb-4B"
if (Test-Path $MolmoSrc) {
    Write-Host "  Copying MolmoWeb-4B model (this may take a minute)..."
    Copy-Item $MolmoSrc $ModelsDir -Recurse
    Write-Host "  Copied: models/MolmoWeb-4B"
} else {
    Write-Host "  MolmoWeb-4B not found — you can add it later to models/" -ForegroundColor Yellow
}
Write-Host "[3/6] Models directory ready" -ForegroundColor Green

# ── 4. Git init ──────────────────────────────────────────────
Set-Location $Target
git init -q
git add .
git commit -q -m "init: LocalPilot project '$Name' (from LocalPilot template)"
Write-Host "[4/6] Git repo initialized with initial commit" -ForegroundColor Green

# ── 5. Set up Python venv ────────────────────────────────────
Write-Host "[5/6] Setting up Python environment (uv sync)..."
uv sync 2>&1 | Select-String -Pattern "Installed|Resolved|error" | ForEach-Object { Write-Host "  $_" }
Write-Host "[5/6] Python environment ready" -ForegroundColor Green

# ── 6. Create empty results file ────────────────────────────
$ResultsHeader = "commit`tval_bpb`tpeak_vram_gb`tstatus`tdescription"
Set-Content -Path (Join-Path $Target "results.tsv") -Value $ResultsHeader
Write-Host "[6/6] Empty results.tsv created" -ForegroundColor Green

# ── Done ─────────────────────────────────────────────────────
Write-Host "`n=== Done! ===" -ForegroundColor Cyan
Write-Host "Your new project is at: $Target"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  cd `"$Target`""
Write-Host "  uv run prepare.py          # download data + tokenizer (one-time, ~2 min)"
Write-Host "  uv run train.py            # test a single run (~2 min)"
Write-Host "  python run_experiments.py  # start autonomous research"
Write-Host ""
Write-Host "Web-enhanced research (requires MolmoWeb-4B in models/):"
Write-Host "  python browse.py search 'Muon optimizer improvements'"
Write-Host "  python run_web_experiments.py"
Write-Host ""

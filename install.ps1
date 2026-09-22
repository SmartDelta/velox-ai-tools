<#
.SYNOPSIS
  Installs Velox AI Tools into its own Python environment.

.DESCRIPTION
  Creates a virtual environment (default %LOCALAPPDATA%\VeloxAITools\.venv),
  installs PyTorch for the chosen CUDA build, then this package with
  Ultralytics. Velox SmartDelta Player finds the result on its own
  (velox-ai.exe in that folder); nothing is written into Velox.

.PARAMETER Dir
  Where the environment goes. Default: %LOCALAPPDATA%\VeloxAITools

.PARAMETER Cuda
  PyTorch build: cu118 (default, verified with torch 2.7.1 on an RTX 4070
  SUPER), cu126, cu128 or cpu.

.PARAMETER Python
  Python interpreter to build the environment from (3.10 - 3.12). Default:
  the "py -3.12" launcher, else "python" on PATH.

.PARAMETER Source
  What to install: the folder of this script (default), a git URL, or a
  PyPI name. Use -Editable for development.

.EXAMPLE
  .\install.ps1
  .\install.ps1 -Cuda cpu
  .\install.ps1 -Dir D:\tools\VeloxAITools -Cuda cu126
#>
param(
    [string]$Dir = "$env:LOCALAPPDATA\VeloxAITools",
    [ValidateSet("cu118", "cu126", "cu128", "cpu")]
    [string]$Cuda = "cu118",
    [string]$Python = "",
    [string]$Source = "",
    [switch]$Editable
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Source) { $Source = $here }

if (-not $Python) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { $Python = "py"; $pyArgs = @("-3.12") } else { $Python = "python"; $pyArgs = @() }
} else {
    $pyArgs = @()
}

$venv = Join-Path $Dir ".venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "Creating environment in $venv"
    New-Item -ItemType Directory -Force $Dir | Out-Null
    & $Python @pyArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed (is Python 3.10-3.12 installed?)" }
}

& $vpy -m pip install --upgrade pip
if ($Cuda -eq "cpu") {
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$Cuda
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

if ($Editable) {
    & $vpy -m pip install -e $Source
} else {
    & $vpy -m pip install $Source
}
if ($LASTEXITCODE -ne 0) { throw "velox-ai-tools install failed" }

$exe = Join-Path $venv "Scripts\velox-ai.exe"
Write-Host ""
Write-Host "Velox AI Tools installed: $exe"
Write-Host "Velox finds this location by itself; otherwise set it under SmartCore AI > AI tools."
& $exe version

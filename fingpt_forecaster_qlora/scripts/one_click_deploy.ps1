param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Test-PythonCommand {
    param([string]$PythonCommand)
    try {
        & $PythonCommand --version *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-BootstrapPython {
    if ($env:PYTHON_BIN -and (Test-PythonCommand $env:PYTHON_BIN)) {
        return $env:PYTHON_BIN
    }

    $candidates = @(
        ".\.venv\Scripts\python.exe",
        ".\.venv-fingpt\Scripts\python.exe",
        "python",
        "python3",
        "py"
    )

    foreach ($candidate in $candidates) {
        if (Test-PythonCommand $candidate) {
            return $candidate
        }
    }

    throw "No usable Python executable found. Install Python 3.10+ or set `$env:PYTHON_BIN to a full python.exe path."
}

if (-not (Test-Path "fingpt_forecaster_qlora\config.env")) {
    Copy-Item "fingpt_forecaster_qlora\config.example.env" "fingpt_forecaster_qlora\config.env"
}

$BootstrapPython = Resolve-BootstrapPython
Write-Host "Using bootstrap Python: $BootstrapPython"

if (-not (Test-Path ".venv-fingpt")) {
    & $BootstrapPython -m venv .venv-fingpt
}

& ".\.venv-fingpt\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools
& ".\.venv-fingpt\Scripts\python.exe" -m pip install -r "fingpt_forecaster_qlora\requirements.txt"

$dataArgs = @(
    "--output-dir", "fingpt_forecaster_qlora\data",
    "--start-date", "20100101",
    "--end-date", "20251231",
    "--negative-ratio", "1.0",
    "--valid-ratio", "0.2",
    "--daily-window", "55",
    "--weekly-window", "55",
    "--min-success-rate", "0.40"
)

if ($Mode -eq "smoke") {
    $dataArgs += @("--positive-limit", "200")
}

& ".\.venv-fingpt\Scripts\python.exe" -m fingpt_forecaster_qlora.build_dataset @dataArgs

Write-Host ""
Write-Host "Dataset is ready. 4-bit bitsandbytes QLoRA is recommended in WSL2/Linux, not native Windows."
Write-Host "Run in WSL2/Linux:"
Write-Host "  bash fingpt_forecaster_qlora/scripts/one_click_deploy.sh $Mode"

param(
  [string]$TradeDate = "20260527",
  [double]$Threshold = 0.80,
  [int]$Limit = 20,
  [int]$SaveTopN = 20,
  [string]$SampleMode = "long",
  [int]$MaxSeqLength = 2048,
  [double]$RequirePositiveRecall = 0.80,
  [string]$ValidationDataDir = "blackbox_finetune_recall80\data_evaluation_no_partial_week",
  [string]$CudaDevice = "0",
  [string]$BlackboxPython = "",
  [string]$AdapterDir = "",
  [string]$Output = "",
  [switch]$SkipRecallGate,
  [switch]$NoSaveDb
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("recall80_prediction_{0}_{1}.log" -f $TradeDate, (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $LogFile -Append | Out-Null

function Resolve-Python {
  param(
    [string]$Requested,
    [string[]]$Candidates
  )
  if ($Requested -and (Test-Path $Requested)) {
    return (Resolve-Path $Requested).Path
  }
  foreach ($candidate in $Candidates) {
    if ($candidate -and (Test-Path $candidate)) {
      return (Resolve-Path $candidate).Path
    }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }
  throw "No Python executable found. Pass -BlackboxPython explicitly."
}

try {
  $env:SAMPLE_MODE = $SampleMode
  $PythonExe = Resolve-Python -Requested $BlackboxPython -Candidates @(
    ".\.venv-blackbox-finetune-recall80\Scripts\python.exe",
    ".\.venv-blackbox-finetune-recall30\Scripts\python.exe",
    ".\.venv\Scripts\python.exe"
  )

  if (-not $AdapterDir) {
    $AdapterDir = if ($SampleMode -eq "short") {
      "blackbox_finetune_recall80\runs\qwen2.5-0.5b-blackbox-recall80-short-lora\adapter"
    } else {
      "blackbox_finetune_recall80\runs\qwen2.5-0.5b-blackbox-recall80-long-lora\adapter"
    }
  }
  if (-not (Test-Path (Join-Path $AdapterDir "adapter_config.json"))) {
    throw "Trained recall80 adapter not found: $AdapterDir"
  }
  if (-not $Output) {
    $Output = "data\blackbox_recall80_predictions_${TradeDate}_threshold90.csv"
  }

  Write-Host "ProjectDir=$ProjectDir"
  Write-Host "TradeDate=$TradeDate"
  Write-Host "Threshold=$Threshold"
  Write-Host "RequirePositiveRecall=$RequirePositiveRecall"
  Write-Host "ValidationDataDir=$ValidationDataDir"
  Write-Host "SampleMode=$SampleMode"
  Write-Host "MaxSeqLength=$MaxSeqLength"
  Write-Host "Python=$PythonExe"
  Write-Host "AdapterDir=$AdapterDir"
  Write-Host "Output=$Output"
  Write-Host "LogFile=$LogFile"
  Write-Warning "The recall gate verifies positive_recall on the configured validation set before prediction. Future realized accuracy still must be verified by later backtest/tracking."

  if (-not $SkipRecallGate) {
    & $PythonExe @(
      "-m", "blackbox_finetune_recall80.evaluate",
      "--adapter-dir", $AdapterDir,
      "--data-dir", $ValidationDataDir,
      "--threshold", [string]$Threshold,
      "--min-positive-recall", [string]$RequirePositiveRecall,
      "--max-seq-length", [string]$MaxSeqLength,
      "--cuda-device", $CudaDevice
    )
    if ($LASTEXITCODE -ne 0) {
      throw "recall80 positive-recall gate failed; prediction was not executed."
    }
  }

  $argsList = @(
    "-m", "blackbox_finetune_recall80.predict_day",
    "--date", $TradeDate,
    "--adapter-dir", $AdapterDir,
    "--threshold", [string]$Threshold,
    "--sample-mode", $SampleMode,
    "--max-seq-length", [string]$MaxSeqLength,
    "--cuda-device", $CudaDevice,
    "--limit", [string]$Limit,
    "--save-top-n", [string]$SaveTopN,
    "--output", $Output
  )
  if ($NoSaveDb) {
    $argsList += "--no-save-db"
  }

  & $PythonExe @argsList
  if ($LASTEXITCODE -ne 0) {
    throw "recall80 prediction failed with exit code $LASTEXITCODE"
  }
  Write-Host "recall80 prediction completed."
}
finally {
  Stop-Transcript | Out-Null
}

param(
  [string]$TradeDate = (Get-Date -Format "yyyyMMdd"),
  [string]$MainPython = "",
  [string]$BlackboxPython = "",
  [string[]]$Strategies = @(
    "blackbox_finetune_recall30",
    "blackbox_finetune_recall35",
    "blackbox_finetune_recall40",
    "blackbox_finetune_recall45",
    "blackbox_finetune_recall50",
    "blackbox_finetune_recall55",
    "blackbox_finetune_recall60",
    "blackbox_finetune_recall65",
    "blackbox_finetune_recall70",
    "blackbox_finetune_recall75",
    "blackbox_finetune_recall80"
  ),
  [double]$BlackboxThreshold = 0.50,
  [int]$BlackboxMaxSeqLength = 512,
  [int]$BlackboxTopN = 5,
  [string]$CudaDevice = "0",
  [int]$CrawlerWorkers = 8,
  [int]$BatchSize = 80,
  [switch]$SkipKlineCrawl,
  [switch]$SkipDerivedKlines,
  [switch]$SkipPredictions,
  [switch]$SkipTracking
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("daily_after_close_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
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
  throw "No Python executable found. Pass -MainPython or -BlackboxPython explicitly."
}

function Invoke-Step {
  param(
    [string]$Name,
    [string]$Exe,
    [string[]]$Arguments
  )
  Write-Host ""
  Write-Host "==== $Name ===="
  Write-Host "$Exe $($Arguments -join ' ')"
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name exit=$LASTEXITCODE"
  }
}

function Test-AdapterReady {
  param([string]$Strategy)
  $adapter = Join-Path $ProjectDir "$Strategy\runs\qwen2.5-0.5b-$($Strategy -replace '^blackbox_finetune_', 'blackbox-')-lora\adapter\adapter_config.json"
  if (Test-Path $adapter) {
    return $true
  }
  $fallback = Join-Path $ProjectDir "$Strategy\runs\qwen2.5-0.5b-blackbox-$($Strategy -replace '^blackbox_finetune_', '')-lora\adapter\adapter_config.json"
  return (Test-Path $fallback)
}

function Get-TradeDateValue {
  return [datetime]::ParseExact($TradeDate, "yyyyMMdd", $null)
}

function Test-WeeklyDue {
  return ((Get-TradeDateValue).DayOfWeek -eq [System.DayOfWeek]::Friday)
}

function Test-MonthlyDue {
  & $MainPythonExe -c "from datetime import datetime; from a_share_crawler import is_last_trade_day; raise SystemExit(0 if is_last_trade_day(datetime.strptime('$TradeDate', '%Y%m%d').date()) else 1)"
  return ($LASTEXITCODE -eq 0)
}

$MainPythonExe = Resolve-Python -Requested $MainPython -Candidates @(".\.venv\Scripts\python.exe")
$BlackboxPythonExe = Resolve-Python -Requested $BlackboxPython -Candidates @(".\.venv-blackbox-finetune-recall30\Scripts\python.exe", $MainPythonExe)

Write-Host "ProjectDir=$ProjectDir"
Write-Host "TradeDate=$TradeDate"
Write-Host "MainPython=$MainPythonExe"
Write-Host "BlackboxPython=$BlackboxPythonExe"
Write-Host "LogFile=$LogFile"

try {
  if (-not $SkipKlineCrawl) {
    Invoke-Step "crawl daily K-lines" $MainPythonExe @(
      ".\a_share_crawler.py", "run",
      "--mode", "incremental",
      "--period", "daily",
      "--end-date", $TradeDate,
      "--workers", [string]$CrawlerWorkers
    )
  }

  if (-not $SkipDerivedKlines) {
    if (Test-WeeklyDue) {
      Invoke-Step "generate weekly K-lines" $MainPythonExe @(".\a_share_crawler.py", "generate", "--period", "weekly")
    } else {
      Write-Host "Skip weekly K-line generation: today is not Friday."
    }
    if (Test-MonthlyDue) {
      Invoke-Step "generate monthly K-lines" $MainPythonExe @(".\a_share_crawler.py", "generate", "--period", "monthly")
    } else {
      Write-Host "Skip monthly K-line generation: today is not the last trading day of the month."
    }
  }

  if (-not $SkipPredictions) {
    foreach ($strategy in $Strategies) {
      if (-not (Test-Path (Join-Path $ProjectDir $strategy))) {
        Write-Host "Skip ${strategy}: directory does not exist."
        continue
      }
      if (-not (Test-AdapterReady $strategy)) {
        Write-Host "Skip ${strategy}: trained adapter not found."
        continue
      }
      Invoke-Step "predict $strategy top $BlackboxTopN" $BlackboxPythonExe @(
        "-m", "$strategy.predict_day",
        "--date", $TradeDate,
        "--threshold", [string]$BlackboxThreshold,
        "--max-seq-length", [string]$BlackboxMaxSeqLength,
        "--cuda-device", $CudaDevice,
        "--save-top-n", [string]$BlackboxTopN,
        "--limit", [string]$BlackboxTopN
      )
    }
  }

  if (-not $SkipTracking) {
    foreach ($strategy in $Strategies) {
      if (-not (Test-Path (Join-Path $ProjectDir $strategy))) {
        continue
      }
      if (-not (Test-AdapterReady $strategy)) {
        Write-Host "Skip tracking ${strategy}: trained adapter not found."
        continue
      }
      Invoke-Step "track portfolio $strategy" $BlackboxPythonExe @(
        "-m", "portfolio_backtest.track_blackbox",
        "--strategy-name", $strategy,
        "--batch-size", [string]$BatchSize
      )
    }
  }

  Write-Host "Daily after-close workflow completed."
}
finally {
  Stop-Transcript | Out-Null
}

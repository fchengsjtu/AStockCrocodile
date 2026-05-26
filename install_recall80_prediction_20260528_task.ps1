param(
  [string]$TaskName = "AStockCrocodile Recall80 Prediction 20260528 0500",
  [datetime]$At = [datetime]"2026-05-28T05:00:00",
  [string]$ScriptPath = "",
  [string]$TradeDate = "20260528",
  [double]$Threshold = 0.90,
  [int]$Limit = 5,
  [int]$SaveTopN = 5,
  [string]$SampleMode = "long",
  [int]$MaxSeqLength = 2048,
  [double]$RequirePositiveRecall = 0.90,
  [string]$CudaDevice = "0"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptPath) {
  $ScriptPath = Join-Path $ProjectDir "run_recall80_prediction_20260528.ps1"
}
if (-not (Test-Path $ScriptPath)) {
  throw "Prediction script not found: $ScriptPath"
}

$argument = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", "`"$ScriptPath`"",
  "-TradeDate", $TradeDate,
  "-Threshold", $Threshold,
  "-Limit", $Limit,
  "-SaveTopN", $SaveTopN,
  "-SampleMode", $SampleMode,
  "-MaxSeqLength", $MaxSeqLength,
  "-RequirePositiveRecall", $RequirePositiveRecall,
  "-CudaDevice", $CudaDevice
) -join " "

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument $argument `
  -WorkingDirectory $ProjectDir
$trigger = New-ScheduledTaskTrigger -Once -At $At
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -ExecutionTimeLimit (New-TimeSpan -Hours 12)

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Run recall80 high-confidence stock prediction once at 2026-05-28 05:00 local time." `
  -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' to run once at $($At.ToString('yyyy-MM-dd HH:mm:ss'))."
Write-Host "The task first requires validation positive_recall >= $RequirePositiveRecall at threshold=$Threshold, then runs prediction."

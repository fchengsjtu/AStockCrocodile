param(
  [string]$TaskName = "AStockCrocodile Recall80 Previous Day Prediction",
  [string]$At = "08:00",
  [string]$ScriptPath = "",
  [string]$TradeDate = "",
  [double]$Threshold = 0.80,
  [int]$Limit = 20,
  [int]$SaveTopN = 20,
  [string]$SampleMode = "long",
  [int]$MaxSeqLength = 2048,
  [double]$RequirePositiveRecall = 0.80,
  [string]$CudaDevice = "0"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptPath) {
  $ScriptPath = Join-Path $ProjectDir "run_recall80_previous_day_prediction.ps1"
}
if (-not (Test-Path $ScriptPath)) {
  throw "Prediction script not found: $ScriptPath"
}

$argument = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", "`"$ScriptPath`"",
  "-Threshold", $Threshold,
  "-Limit", $Limit,
  "-SaveTopN", $SaveTopN,
  "-SampleMode", $SampleMode,
  "-MaxSeqLength", $MaxSeqLength,
  "-RequirePositiveRecall", $RequirePositiveRecall,
  "-CudaDevice", $CudaDevice
) -join " "
if ($TradeDate) {
  $argument = "$argument -TradeDate $TradeDate"
}

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument $argument `
  -WorkingDirectory $ProjectDir
$trigger = New-ScheduledTaskTrigger `
  -Daily `
  -At $At
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
  -Description "Run recall80 stock prediction daily at $At for the previous calendar day." `
  -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' to run daily at $At."
Write-Host "The task predicts the previous calendar day by default. Pass -TradeDate only for a fixed-date override."
Write-Host "The task first requires validation positive_recall >= $RequirePositiveRecall at threshold=$Threshold, then runs prediction."

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Invoke-Step {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CommandArgs)
  & $VenvPython @CommandArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $VenvPython $($CommandArgs -join ' ')"
  }
}

function Test-TorchCuda {
  $Diag = & $VenvPython -c "import torch; print('torch=' + torch.__version__); print('torch_cuda=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_device_count=' + str(torch.cuda.device_count()))" 2>&1
  $Diag | ForEach-Object { Write-Host $_ }
  if ($LASTEXITCODE -ne 0) {
    return $false
  }
  $null = & $VenvPython -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>&1
  return ($LASTEXITCODE -eq 0)
}

function Install-CudaTorch {
  $TorchCudaIndex = if ($env:TORCH_CUDA_INDEX) { $env:TORCH_CUDA_INDEX } else { "https://download.pytorch.org/whl/cu121" }
  Write-Host "Installing CUDA-enabled PyTorch from $TorchCudaIndex"
  & $VenvPython -m pip uninstall -y torch torchvision torchaudio
  & $VenvPython -m pip install --index-url $TorchCudaIndex torch torchvision torchaudio
  if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed. Try setting TORCH_CUDA_INDEX, for example https://download.pytorch.org/whl/cu121 or https://download.pytorch.org/whl/cu124"
  }
}

function Test-EnvFlag {
  param([string]$Value)
  return @("1", "true", "TRUE", "yes", "YES", "y", "Y") -contains $Value
}

function Test-DatasetReady {
  param([string]$Dir)
  return (Test-Path (Join-Path $Dir "train.jsonl")) -and (Test-Path (Join-Path $Dir "test.jsonl"))
}

function Invoke-DatasetBuildIfNeeded {
  param(
    [string]$Dir,
    [string]$Label,
    [string]$ForceValue,
    [string[]]$CommandArgs
  )
  if (Test-DatasetReady $Dir) {
    if (Test-EnvFlag $ForceValue) {
      Write-Host "Rebuilding cached $Label dataset in $Dir because the rebuild flag is set."
    } else {
      Write-Host "Using cached $Label dataset in $Dir; set REBUILD_DATASET=1 to rebuild all datasets."
      return
    }
  }
  Invoke-Step @CommandArgs
}

function Write-EnvironmentSnapshot {
  $ProjectEnvNames = @(
    "PYTHON_BIN",
    "VENV_DIR",
    "BASE_MODEL",
    "DATA_DIR",
    "VALIDATION_DATA_DIR",
    "OUTPUT_DIR",
    "CUDA_DEVICE",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDA_INDEX",
    "SAMPLE_MODE",
    "MAX_SEQ_LENGTH",
    "NEGATIVE_RATIO",
    "SAMPLE_BOTTOM_BAND_RATIO",
    "TRAIN_START_DATE",
    "TRAIN_END_DATE",
    "VALIDATION_START_DATE",
    "VALIDATION_END_DATE",
    "TEST_START_DATE",
    "TEST_END_DATE",
    "POSITIVE_LIMIT",
    "SMOKE_POSITIVE_LIMIT",
    "MIN_POSITIVE_RECALL",
    "REBUILD_DATASET",
    "REBUILD_VALIDATION_DATASET",
    "REBUILD_TOKEN_CACHE",
    "NO_AUTO_RESUME",
    "RESUME_ADAPTER_DIR",
    "CHECKPOINT_EVERY",
    "EPOCHS",
    "GRADIENT_ACCUMULATION_STEPS",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "TRAIN_SEED",
    "MAX_GRAD_NORM",
    "LORA_RANK",
    "LORA_DROPOUT",
    "OOM_PATIENCE",
    "MIN_SEQ_LENGTH_ON_OOM",
    "OOM_SHRINK_FACTOR",
    "NONFINITE_SKIP_LIMIT",
    "NONFINITE_BACKOFF_EVERY",
    "LR_BACKOFF_FACTOR",
    "MIN_LEARNING_RATE",
    "HF_HOME",
    "HF_ENDPOINT",
    "HF_TOKEN"
  )
  Write-Host ""
  Write-Host "==== Project environment variables ===="
  foreach ($name in $ProjectEnvNames) {
    $value = [Environment]::GetEnvironmentVariable($name, "Process")
    if ($null -eq $value -or $value -eq "") {
      $value = "<unset>"
    }
    Write-Host ("{0}={1}" -f $name, $value)
  }
  if ([Environment]::GetEnvironmentVariable("LEARING_RATE", "Process")) {
    Write-Warning "LEARING_RATE is ignored. Did you mean LEARNING_RATE?"
  }
  Write-Host "==== End project environment variables ===="
  Write-Host ""
}
$Mode = if ($args.Count -gt 0) { $args[0] } else { "smoke" }
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } elseif (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".\.venv-blackbox-finetune-recall35" }

if (!(Test-Path $VenvDir)) {
  & $PythonBin -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
Invoke-Step -m pip install --upgrade pip wheel "setuptools<82"
Invoke-Step -m pip install -r .\blackbox_finetune_recall35\requirements.txt

$BaseModel = if ($env:BASE_MODEL) { $env:BASE_MODEL } else { "Qwen/Qwen2.5-0.5B-Instruct" }
$CudaDevice = if ($env:CUDA_DEVICE) { $env:CUDA_DEVICE } else { "0" }
$env:CUDA_VISIBLE_DEVICES = $CudaDevice
$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "" }

if (!(Test-TorchCuda)) {
  Install-CudaTorch
}
Invoke-Step -m blackbox_finetune_recall35.gpu --cuda-device $CudaDevice
Write-EnvironmentSnapshot
if ($Mode -eq "diagnose") {
  exit 0
}

$MinRecall = if ($env:MIN_POSITIVE_RECALL) { $env:MIN_POSITIVE_RECALL } else { "0.60" }
$TrainSeed = if ($env:TRAIN_SEED) { $env:TRAIN_SEED } else { "20260560" }
$WeightDecay = if ($env:WEIGHT_DECAY) { $env:WEIGHT_DECAY } else { "0.0" }
$LearningRate = if ($env:LEARNING_RATE) { $env:LEARNING_RATE } else { "5e-6" }
$MaxGradNorm = if ($env:MAX_GRAD_NORM) { $env:MAX_GRAD_NORM } else { "0.5" }
$LoraRank = if ($env:LORA_RANK) { $env:LORA_RANK } else { "16" }
$LoraDropout = if ($env:LORA_DROPOUT) { $env:LORA_DROPOUT } else { "0.05" }
$OomPatience = if ($env:OOM_PATIENCE) { $env:OOM_PATIENCE } else { "20" }
$NonfiniteSkipLimit = if ($env:NONFINITE_SKIP_LIMIT) { $env:NONFINITE_SKIP_LIMIT } else { "100" }
$NonfiniteBackoffEvery = if ($env:NONFINITE_BACKOFF_EVERY) { $env:NONFINITE_BACKOFF_EVERY } else { "10" }
$LrBackoffFactor = if ($env:LR_BACKOFF_FACTOR) { $env:LR_BACKOFF_FACTOR } else { "0.5" }
$MinLearningRate = if ($env:MIN_LEARNING_RATE) { $env:MIN_LEARNING_RATE } else { "1e-6" }
$ResumeAdapterDir = $env:RESUME_ADAPTER_DIR
$SampleMode = if ($env:SAMPLE_MODE) { $env:SAMPLE_MODE } else { "long" }
$DefaultDataDir = "blackbox_finetune_recall35/data_no_partial_week_$SampleMode"
$DefaultValidationDir = "blackbox_finetune_recall35/data_evaluation_no_partial_week_$SampleMode"
$DataDir = if ($env:DATA_DIR) { $env:DATA_DIR } else { $DefaultDataDir }
$ValidationDir = if ($env:VALIDATION_DATA_DIR) { $env:VALIDATION_DATA_DIR } else { $DefaultValidationDir }
$DefaultOutputDir = if ($SampleMode -eq "short") { "blackbox_finetune_recall35/runs/qwen2.5-0.5b-blackbox-recall35-short-lora" } elseif ($SampleMode -eq "xlong") { "blackbox_finetune_recall35/runs/qwen2.5-0.5b-blackbox-recall35-xlong-lora" } elseif ($SampleMode -eq "xxlong") { "blackbox_finetune_recall35/runs/qwen2.5-0.5b-blackbox-recall35-xxlong-lora" } else { "blackbox_finetune_recall35/runs/qwen2.5-0.5b-blackbox-recall35-long-lora" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { $DefaultOutputDir }
$NegativeRatio = if ($env:NEGATIVE_RATIO) { $env:NEGATIVE_RATIO } else { "3.0" }
$DefaultMaxSeqLength = if ($SampleMode -eq "short") { "1024" } elseif ($SampleMode -eq "xlong") { "3072" } elseif ($SampleMode -eq "xxlong") { "4096" } else { "2048" }
$DefaultCheckpointEvery = "500"
$CheckpointEvery = if ($env:CHECKPOINT_EVERY) { $env:CHECKPOINT_EVERY } else { $DefaultCheckpointEvery }

if ($Mode -eq "smoke") {
  $DefaultTrainStart = "20200101"
  $DefaultTrainEnd = "20211231"
  $DefaultValidationStart = "20260101"
  $DefaultValidationEnd = "20260131"
  $TrainStart = if ($env:TRAIN_START_DATE) { $env:TRAIN_START_DATE } else { $DefaultTrainStart }
  $TrainEnd = if ($env:TRAIN_END_DATE) { $env:TRAIN_END_DATE } else { $DefaultTrainEnd }
  $ValidationStart = if ($env:VALIDATION_START_DATE) { $env:VALIDATION_START_DATE } elseif ($env:TEST_START_DATE) { $env:TEST_START_DATE } else { $DefaultValidationStart }
  $ValidationEnd = if ($env:VALIDATION_END_DATE) { $env:VALIDATION_END_DATE } elseif ($env:TEST_END_DATE) { $env:TEST_END_DATE } else { $DefaultValidationEnd }
  $PositiveLimit = if ($env:SMOKE_POSITIVE_LIMIT) { $env:SMOKE_POSITIVE_LIMIT } else { "12" }
  $Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "3" }
  $MaxSeqLength = if ($env:MAX_SEQ_LENGTH) { $env:MAX_SEQ_LENGTH } else { $DefaultMaxSeqLength }
  $GradSteps = if ($env:GRADIENT_ACCUMULATION_STEPS) { $env:GRADIENT_ACCUMULATION_STEPS } else { "1" }
} else {
  $DefaultTrainStart = "20200101"
  $DefaultTrainEnd = "20251231"
  $DefaultValidationStart = "20260101"
  $DefaultValidationEnd = "20260430"
  $TrainStart = if ($env:TRAIN_START_DATE) { $env:TRAIN_START_DATE } else { $DefaultTrainStart }
  $TrainEnd = if ($env:TRAIN_END_DATE) { $env:TRAIN_END_DATE } else { $DefaultTrainEnd }
  $ValidationStart = if ($env:VALIDATION_START_DATE) { $env:VALIDATION_START_DATE } elseif ($env:TEST_START_DATE) { $env:TEST_START_DATE } else { $DefaultValidationStart }
  $ValidationEnd = if ($env:VALIDATION_END_DATE) { $env:VALIDATION_END_DATE } elseif ($env:TEST_END_DATE) { $env:TEST_END_DATE } else { $DefaultValidationEnd }
  $PositiveLimit = $env:POSITIVE_LIMIT
  $Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "1" }
  $MaxSeqLength = if ($env:MAX_SEQ_LENGTH) { $env:MAX_SEQ_LENGTH } else { $DefaultMaxSeqLength }
  $GradSteps = if ($env:GRADIENT_ACCUMULATION_STEPS) { $env:GRADIENT_ACCUMULATION_STEPS } else { "8" }
}

$BuildArgs = @("-m", "blackbox_finetune_recall35.build_dataset", "--output-dir", $DataDir, "--start-date", $TrainStart, "--end-date", $TrainEnd, "--negative-ratio", $NegativeRatio, "--sample-mode", $SampleMode)
if ($PositiveLimit) { $BuildArgs += @("--positive-limit", $PositiveLimit) }
Invoke-DatasetBuildIfNeeded -Dir $DataDir -Label "training" -ForceValue $env:REBUILD_DATASET -CommandArgs $BuildArgs

$ValArgs = @("-m", "blackbox_finetune_recall35.build_validation_dataset", "--output-dir", $ValidationDir, "--start-date", $ValidationStart, "--end-date", $ValidationEnd, "--negative-ratio", $NegativeRatio, "--sample-mode", $SampleMode)
if ($Mode -eq "smoke") { $ValArgs += @("--positive-limit", $PositiveLimit) }
$ForceValidationDataset = if ($env:REBUILD_VALIDATION_DATASET) { $env:REBUILD_VALIDATION_DATASET } else { $env:REBUILD_DATASET }
Invoke-DatasetBuildIfNeeded -Dir $ValidationDir -Label "validation" -ForceValue $ForceValidationDataset -CommandArgs $ValArgs

$TrainArgs = @(
  '-m', 'blackbox_finetune_recall35.train', '--base-model', $BaseModel, '--data-dir', $DataDir, '--output-dir', $OutputDir,
  '--max-seq-length', $MaxSeqLength, '--epochs', $Epochs, '--batch-size', '1', '--gradient-accumulation-steps', $GradSteps,
  '--learning-rate', $LearningRate, '--weight-decay', $WeightDecay, '--max-grad-norm', $MaxGradNorm, '--lora-rank', $LoraRank, '--lora-dropout', $LoraDropout, '--checkpoint-every', $CheckpointEvery, '--oom-patience', $OomPatience,
  '--nonfinite-skip-limit', $NonfiniteSkipLimit, '--nonfinite-backoff-every', $NonfiniteBackoffEvery, '--lr-backoff-factor', $LrBackoffFactor,
  '--min-learning-rate', $MinLearningRate,
  '--train-seed', $TrainSeed, '--cuda-device', $CudaDevice, '--no-4bit'
)
if ($ResumeAdapterDir) { $TrainArgs += @('--resume-adapter-dir', $ResumeAdapterDir) }
if (Test-EnvFlag $env:REBUILD_TOKEN_CACHE) { $TrainArgs += @('--rebuild-token-cache') }
if (Test-EnvFlag $env:NO_AUTO_RESUME) { $TrainArgs += @('--no-auto-resume') }
Invoke-Step @TrainArgs
Invoke-Step -m blackbox_finetune_recall35.evaluate --base-model $BaseModel --adapter-dir "$OutputDir/adapter" --data-dir $ValidationDir --threshold 0.50 --min-positive-recall $MinRecall --cuda-device $CudaDevice --max-seq-length $MaxSeqLength
Invoke-Step -m unittest tests.test_blackbox_finetune_recall35 -v

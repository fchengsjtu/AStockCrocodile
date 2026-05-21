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
  if ((Test-DatasetReady $Dir) -and -not (Test-EnvFlag $ForceValue)) {
    Write-Host "Using cached $Label dataset in $Dir; set REBUILD_DATASET=1 to rebuild all datasets."
    return
  }
  Invoke-Step @CommandArgs
}

$Mode = if ($args.Count -gt 0) { $args[0] } else { "smoke" }
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } elseif (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".\.venv-blackbox-finetune-recall30" }

if (!(Test-Path $VenvDir)) {
  & $PythonBin -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
Invoke-Step -m pip install --upgrade pip wheel "setuptools<82"
Invoke-Step -m pip install -r .\blackbox_finetune_recall30\requirements.txt

$BaseModel = if ($env:BASE_MODEL) { $env:BASE_MODEL } else { "Qwen/Qwen2.5-0.5B-Instruct" }
$CudaDevice = if ($env:CUDA_DEVICE) { $env:CUDA_DEVICE } else { "0" }
$env:CUDA_VISIBLE_DEVICES = $CudaDevice
$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "" }

if (!(Test-TorchCuda)) {
  Install-CudaTorch
}
Invoke-Step -m blackbox_finetune_recall30.gpu --cuda-device $CudaDevice
if ($Mode -eq "diagnose") {
  exit 0
}

$DataDir = if ($env:DATA_DIR) { $env:DATA_DIR } else { "blackbox_finetune_recall30/data" }
$ValidationDir = if ($env:VALIDATION_DATA_DIR) { $env:VALIDATION_DATA_DIR } else { "blackbox_finetune_recall30/data_validation" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "blackbox_finetune_recall30/runs/qwen2.5-0.5b-blackbox-recall30-lora" }
$MinRecall = if ($env:MIN_POSITIVE_RECALL) { $env:MIN_POSITIVE_RECALL } else { "0.30" }
$TrainSeed = if ($env:TRAIN_SEED) { $env:TRAIN_SEED } else { "20260530" }
$LearningRate = if ($env:LEARNING_RATE) { $env:LEARNING_RATE } else { "2e-5" }
$MaxGradNorm = if ($env:MAX_GRAD_NORM) { $env:MAX_GRAD_NORM } else { "0.5" }
$CheckpointEvery = if ($env:CHECKPOINT_EVERY) { $env:CHECKPOINT_EVERY } else { "1000" }
$ResumeAdapterDir = $env:RESUME_ADAPTER_DIR

if ($Mode -eq "smoke") {
  $TrainStart = "20110101"
  $TrainEnd = "20151231"
  $ValidationStart = "20260101"
  $ValidationEnd = "20260131"
  $PositiveLimit = if ($env:SMOKE_POSITIVE_LIMIT) { $env:SMOKE_POSITIVE_LIMIT } else { "12" }
  $Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "3" }
  $MaxSeqLength = if ($env:MAX_SEQ_LENGTH) { $env:MAX_SEQ_LENGTH } else { "512" }
  $GradSteps = if ($env:GRADIENT_ACCUMULATION_STEPS) { $env:GRADIENT_ACCUMULATION_STEPS } else { "1" }
} else {
  $TrainStart = "20110101"
  $TrainEnd = "20251231"
  $ValidationStart = "20260101"
  $ValidationEnd = "20260430"
  $PositiveLimit = $env:POSITIVE_LIMIT
  $Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "1" }
  $MaxSeqLength = if ($env:MAX_SEQ_LENGTH) { $env:MAX_SEQ_LENGTH } else { "2048" }
  $GradSteps = if ($env:GRADIENT_ACCUMULATION_STEPS) { $env:GRADIENT_ACCUMULATION_STEPS } else { "8" }
}

$BuildArgs = @("-m", "blackbox_finetune_recall30.build_dataset", "--output-dir", $DataDir, "--start-date", $TrainStart, "--end-date", $TrainEnd, "--negative-ratio", "1.0")
if ($PositiveLimit) { $BuildArgs += @("--positive-limit", $PositiveLimit) }
Invoke-DatasetBuildIfNeeded -Dir $DataDir -Label "training" -ForceValue $env:REBUILD_DATASET -CommandArgs $BuildArgs

$ValArgs = @("-m", "blackbox_finetune_recall30.build_validation_dataset", "--output-dir", $ValidationDir, "--start-date", $ValidationStart, "--end-date", $ValidationEnd, "--negative-ratio", "1.0")
if ($Mode -eq "smoke") { $ValArgs += @("--positive-limit", $PositiveLimit) }
$ForceValidationDataset = if ($env:REBUILD_VALIDATION_DATASET) { $env:REBUILD_VALIDATION_DATASET } else { $env:REBUILD_DATASET }
Invoke-DatasetBuildIfNeeded -Dir $ValidationDir -Label "validation" -ForceValue $ForceValidationDataset -CommandArgs $ValArgs

$TrainArgs = @(
  '-m', 'blackbox_finetune_recall30.train', '--base-model', $BaseModel, '--data-dir', $DataDir, '--output-dir', $OutputDir,
  '--max-seq-length', $MaxSeqLength, '--epochs', $Epochs, '--batch-size', '1', '--gradient-accumulation-steps', $GradSteps,
  '--learning-rate', $LearningRate, '--max-grad-norm', $MaxGradNorm, '--checkpoint-every', $CheckpointEvery,
  '--train-seed', $TrainSeed, '--cuda-device', $CudaDevice, '--no-4bit'
)
if ($ResumeAdapterDir) { $TrainArgs += @('--resume-adapter-dir', $ResumeAdapterDir) }
if (Test-EnvFlag $env:REBUILD_TOKEN_CACHE) { $TrainArgs += @('--rebuild-token-cache') }
Invoke-Step @TrainArgs
Invoke-Step -m blackbox_finetune_recall30.evaluate --base-model $BaseModel --adapter-dir "$OutputDir/adapter" --data-dir $ValidationDir --threshold 0.50 --min-positive-recall $MinRecall --cuda-device $CudaDevice --max-seq-length $MaxSeqLength
Invoke-Step -m unittest tests.test_blackbox_finetune_recall30 -v

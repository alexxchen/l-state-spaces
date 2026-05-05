$Experiment = if ($env:EXPERIMENT) { $env:EXPERIMENT } else { "custom-memLinOSS-listops" }
$RunGroup = if ($env:RUN_GROUP) { $env:RUN_GROUP } else { $Experiment }
$RunId = if ($env:RUN_ID) { $env:RUN_ID } else { "$RunGroup-$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
$RootDir = if ($env:ROOT_DIR) { $env:ROOT_DIR } else { $PWD.Path }
$LogFile = if ($env:LOG_FILE) { $env:LOG_FILE } else { Join-Path $RootDir "$RunId.log" }
$ContainerLogFile = if ($env:LOG_FILE -and $env:LOG_FILE.StartsWith("/")) {
    $env:LOG_FILE
} else {
    "/workspace/$(Split-Path -Leaf $LogFile)"
}

docker run --rm -d `
    --gpus all `
    -v "${PWD}:/workspace" `
    --workdir /workspace `
    --env WANDB_API_KEY=$env:WANDB_API_KEY `
    --env EXPERIMENT="$Experiment" `
    --env RUN_GROUP="$RunGroup" `
    --env RUN_ID="$RunId" `
    --env LOG_FILE="$ContainerLogFile" `
    --env NNODE=1 `
    --env NGPU=1 `
    --env LOG_RANK=0 `
    alecchen123/lra:fla `
    bash -lc 'python -m train wandb.project=lra wandb.group="$RUN_GROUP" wandb.id="$RUN_ID" experiment="$EXPERIMENT" > "$LOG_FILE" 2>&1'

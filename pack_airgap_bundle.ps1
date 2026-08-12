$ErrorActionPreference = "Stop"

$BUNDLE_DIR = "hybrid_rag_airgap_bundle"
$PLATFORM = "linux/arm64"

# Explicit list of Ollama models to pre-fetch into the airgap bundle
$OLLAMA_MODELS = @("llama3.2", "nomic-embed-text")

Write-Host "=== 1. Preparing Airgap Bundle Directory ===" -ForegroundColor Green
if (Test-Path $BUNDLE_DIR) { 
    Remove-Item -Recurse -Force $BUNDLE_DIR 
}
New-Item -ItemType Directory -Path "$BUNDLE_DIR\images" | Out-Null
New-Item -ItemType Directory -Path "$BUNDLE_DIR\ollama_models" | Out-Null

Write-Host "=== 2. Enabling Docker Buildx for ARM64 Cross-Compilation ===" -ForegroundColor Green
# Temporarily relax EAP to handle CLI stderr status messages cleanly
$oldEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"

# Re-initialize clean buildx instance
docker buildx rm arm64-builder -f 2>&1 | Out-Null
docker buildx create --name arm64-builder --use 2>&1 | Out-Null

$ErrorActionPreference = $oldEAP

Write-Host "=== 3. Building hybrid-rag-api Image for ARM64 ===" -ForegroundColor Green
# Disabling provenance metadata prevents digest/manifest collisions across architectures
docker buildx build --platform $PLATFORM --provenance=false -t hybrid-rag-api:arm64 -f Dockerfile.api --load .

Write-Host "=== 4 & 5. Pulling & Exporting ARM64 Base Images ===" -ForegroundColor Green

# 1. Save locally built API image
docker save hybrid-rag-api:arm64 -o "$BUNDLE_DIR\images\hybrid-rag-api.tar"

# 2. Base images to export
$remoteImages = @(
    @{ Tag = "neo4j:5.18.0"; Out = "neo4j.tar" },
    @{ Tag = "qdrant/qdrant:v1.8.2"; Out = "qdrant.tar" },
    @{ Tag = "ghcr.io/open-webui/open-webui:main"; Out = "open-webui.tar" },
    @{ Tag = "ollama/ollama:latest"; Out = "ollama.tar" }
)

$ErrorActionPreference = "Continue"

foreach ($img in $remoteImages) {
    Write-Host "Pulling and saving ARM64 image: $($img.Tag)" -ForegroundColor Cyan
    
    # Force host engine to pull explicitly for linux/arm64
    docker pull --platform $PLATFORM $img.Tag 2>&1
    
    # Save the pulled image to the tar destination
    docker save $img.Tag -o "$BUNDLE_DIR\images\$($img.Out)" 2>&1
}

$ErrorActionPreference = $oldEAP

Write-Host "=== 6. Downloading Ollama Model Weights ===" -ForegroundColor Green

$modelDir = "$BUNDLE_DIR\ollama_models"
if (-not (Test-Path $modelDir)) {
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
}

# Clean up stale temp container if present
docker rm -f temp_ollama 2>$null

# Spin up temporary container without host mount (bypasses WSL mount bugs)
docker run -d --name temp_ollama --platform $PLATFORM -p 11435:11434 ollama/ollama:latest

# Poll for daemon endpoint readiness
Write-Host "Waiting for temporary Ollama daemon readiness..." -ForegroundColor Yellow
$retry = 0
$daemonReady = $false

while ($retry -lt 15) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11435/api/version" -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) { 
            $daemonReady = $true
            break 
        }
    } catch {
        Start-Sleep -Seconds 2
        $retry++
    }
}

if (-not $daemonReady) {
    docker logs temp_ollama
    docker rm -f temp_ollama 2>$null
    throw "Ollama daemon failed to start on port 11435."
}

# Pull models into container filesystem
foreach ($model in $OLLAMA_MODELS) {
    Write-Host "Pulling model: $model" -ForegroundColor Cyan
    docker exec temp_ollama ollama pull $model
}

# Extract downloaded model weights directly out of container to host folder
Write-Host "Copying downloaded model weights to host folder..." -ForegroundColor Yellow
docker cp temp_ollama:/root/.ollama/. "$modelDir/"

# Clean up temporary instance
docker stop temp_ollama 2>$null
docker rm temp_ollama 2>$null

Write-Host "=== 7. Copying Deployment Configurations ===" -ForegroundColor Green
Copy-Item "docker-compose.airgap.yml" "$BUNDLE_DIR\docker-compose.yml"

# Generate deploy_airgap.sh with strict Unix LF line endings
$deployScriptLines = @(
    '#!/usr/bin/env bash',
    'set -eo pipefail',
    '',
    'echo "=== 1. Loading ARM64 Docker Container Images ==="',
    'for img in images/*.tar; do',
    '    echo "Loading $img..."',
    '    docker load -i "$img"',
    'done',
    '',
    'echo "=== 2. Verifying Ollama Models Directory ==="',
    'if [ -d "ollama_models/models" ]; then',
    '    echo "Ollama model weights successfully detected in ./ollama_models."',
    'else',
    '    echo "Warning: Ollama models directory appears empty!"',
    'fi',
    '',
    'echo "=== 3. Starting Stack via Docker Compose ==="',
    'docker compose up -d',
    '',
    'echo "=== 4. Checking Service Status & Loaded Models ==="',
    'sleep 10',
    'docker compose ps',
    'docker compose exec -T ollama ollama list || true',
    '',
    'echo "=== Hybrid RAG Stack Successfully Deployed ==="'
)

$deployScriptPath = "$BUNDLE_DIR\deploy_airgap.sh"
[System.IO.File]::WriteAllLines($deployScriptPath, $deployScriptLines)

Write-Host "=== 8. Archiving Bundle into hybrid_rag_arm64_airgap.tar.gz ===" -ForegroundColor Green
tar -czvf hybrid_rag_arm64_airgap.tar.gz $BUNDLE_DIR

Write-Host "=== PACKAGING COMPLETE ===" -ForegroundColor Cyan
Write-Host "Archive ready for airgap transfer: hybrid_rag_arm64_airgap.tar.gz"
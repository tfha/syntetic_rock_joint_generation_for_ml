#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build and test Docker image locally before Azure ML environment registration.

.DESCRIPTION
    This script automates the local Docker build and test workflow recommended for Azure ML environments.
    It performs validation at each step and provides clear feedback for debugging.

.PARAMETER ImageName
    Name for the Docker image (default: ml-segmentation-local)

.PARAMETER Tag
    Tag for the Docker image (default: latest)

.PARAMETER SkipTests
    Skip the container test phase

.PARAMETER Clean
    Remove existing image before building

.EXAMPLE
    .\build_and_test_docker.ps1

.EXAMPLE
    .\build_and_test_docker.ps1 -ImageName "my-ml-env" -Tag "v1.0" -Clean

.NOTES
    Author: Azure ML Team
    Requires: Docker Desktop or Docker Engine running
#>

param(
    [string]$ImageName = "ml-segmentation-local",
    [string]$Tag = "latest",
    [switch]$SkipTests,
    [switch]$Clean
)

# Color coding for output
function Write-Step { param($Message) Write-Host "🔹 $Message" -ForegroundColor Cyan }
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }

# Configuration
$FullImageName = "${ImageName}:${Tag}"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TestCommands = @(
    "python --version",
    "python -c 'import torch; print(f`"PyTorch version: {torch.__version__}`")'",
    "python -c 'import ml_segmentation; print(`"Package imported successfully`")'",
    "python -c 'import torch; print(f`"CUDA available: {torch.cuda.is_available()}`"); print(f`"CUDA devices: {torch.cuda.device_count()}`")'"
)

Write-Host "🐳 Docker Build and Test Automation for Azure ML" -ForegroundColor Magenta
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host ""

# Step 1: Validate Docker is running
Write-Step "Checking Docker availability..."
try {
    $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker is running (Server version: $dockerVersion)"
    } else {
        throw "Docker not accessible"
    }
} catch {
    Write-Error "Docker is not running or not accessible."
    Write-Host "Please ensure Docker Desktop is started or Docker Engine is running." -ForegroundColor Yellow
    Write-Host "Windows: Start Docker Desktop from Start Menu" -ForegroundColor Yellow
    Write-Host "Linux: sudo systemctl start docker" -ForegroundColor Yellow
    exit 1
}

# Step 2: Validate project structure
Write-Step "Validating project structure..."
$RequiredFiles = @("Dockerfile", "pyproject.toml", "poetry.lock", "src/ml_segmentation")
foreach ($file in $RequiredFiles) {
    $filePath = Join-Path $ProjectRoot $file
    if (-not (Test-Path $filePath)) {
        Write-Error "Required file/directory not found: $file"
        exit 1
    }
}
Write-Success "Project structure validated"

# Step 3: Check disk space
Write-Step "Checking available disk space..."
$drive = (Get-Location).Drive
$freeSpace = (Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$($drive.Name)'").FreeSpace / 1GB
if ($freeSpace -lt 10) {
    Write-Warning "Low disk space: ${freeSpace:F1} GB free. ML images can be large (2-8 GB)."
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne 'y') { exit 1 }
} else {
    Write-Success "Sufficient disk space: ${freeSpace:F1} GB free"
}

# Step 4: Clean existing image if requested
if ($Clean) {
    Write-Step "Removing existing image if present..."
    docker rmi $FullImageName 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Removed existing image: $FullImageName"
    } else {
        Write-Host "No existing image to remove" -ForegroundColor Gray
    }
}

# Step 5: Build Docker image
Write-Step "Building Docker image: $FullImageName"
Write-Host "This may take 5-15 minutes depending on your system..." -ForegroundColor Gray

$buildStart = Get-Date
Push-Location $ProjectRoot
try {
    docker build -t $FullImageName -f Dockerfile .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed"
    }
} catch {
    Write-Error "Docker build failed. Check the output above for details."
    Pop-Location
    exit 1
} finally {
    Pop-Location
}

$buildDuration = (Get-Date) - $buildStart
Write-Success "Build completed in $($buildDuration.TotalMinutes.ToString('F1')) minutes"

# Step 6: Inspect image
Write-Step "Inspecting built image..."
$imageInfo = docker images $FullImageName --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | Select-Object -Skip 1
Write-Success "Image created: $imageInfo"

# Step 7: Test the container (unless skipped)
if (-not $SkipTests) {
    Write-Step "Testing container functionality..."

    foreach ($cmd in $TestCommands) {
        Write-Host "  Running: $cmd" -ForegroundColor Gray
        # Capture both stdout and stderr so that Python tracebacks are visible on failure
        $result = docker run --rm $FullImageName bash -c "$cmd" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    ✓ $result" -ForegroundColor Green
        } else {
            Write-Error "    ✗ Command failed: $cmd"
            if ($result) {
                Write-Host "    --- Begin captured output (stdout+stderr) ---" -ForegroundColor DarkYellow
                Write-Host $result -ForegroundColor Yellow
                Write-Host "    --- End captured output ---" -ForegroundColor DarkYellow
            } else {
                Write-Host "    (No output captured; command may have exited silently)" -ForegroundColor DarkYellow
            }
            Write-Host "Container test failed. Check traceback above and adjust dependencies/Dockerfile." -ForegroundColor Yellow
            exit 1
        }
    }
    Write-Success "All container tests passed"
} else {
    Write-Warning "Skipping container tests"
}

# Step 8: Final summary and next steps
Write-Host ""
Write-Success "Local Docker build and test completed successfully!"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Register environment with Azure ML:" -ForegroundColor White
Write-Host "   python scripts/azure_manage_assets_and_resources.py azure_data_assets.command=build-environment" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Submit training job:" -ForegroundColor White
Write-Host "   python scripts/azure_submit_job.py azure_ml.environment_name=rock-segmentation-env-py311" -ForegroundColor Gray
Write-Host ""
Write-Host "Local image available as: $FullImageName" -ForegroundColor Yellow

# Optional: Cleanup prompt
Write-Host ""
$cleanup = Read-Host "Remove local image to save disk space? (y/N)"
if ($cleanup -eq 'y') {
    docker rmi $FullImageName
    Write-Success "Local image removed"
}

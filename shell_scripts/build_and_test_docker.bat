@echo off
REM Simple batch script for Docker build and test workflow
REM For Windows users who prefer .bat files over PowerShell

setlocal
set IMAGE_NAME=ml-segmentation-local
set TAG=latest
set FULL_IMAGE_NAME=%IMAGE_NAME%:%TAG%

echo.
echo ===================================================
echo  Docker Build and Test for Azure ML Environment
echo ===================================================
echo.

REM Check if Docker is running
echo Checking Docker availability...
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not accessible.
    echo Please start Docker Desktop from the Start Menu.
    pause
    exit /b 1
)
echo Docker is running.
echo.

REM Build the Docker image
echo Building Docker image: %FULL_IMAGE_NAME%
echo This may take 5-15 minutes...
docker build -t %FULL_IMAGE_NAME% -f Dockerfile .
if %errorlevel% neq 0 (
    echo ERROR: Docker build failed.
    pause
    exit /b 1
)
echo Build completed successfully.
echo.

REM Test the container
echo Testing container functionality...
docker run --rm %FULL_IMAGE_NAME% python --version
if %errorlevel% neq 0 goto test_failed

docker run --rm %FULL_IMAGE_NAME% python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
if %errorlevel% neq 0 goto test_failed

docker run --rm %FULL_IMAGE_NAME% python -c "import ml_segmentation; print('Package imported successfully')"
if %errorlevel% neq 0 goto test_failed

echo All container tests passed.
echo.

REM Show image info
echo Image information:
docker images %FULL_IMAGE_NAME%
echo.

echo SUCCESS: Local Docker build and test completed!
echo.
echo Next Steps:
echo 1. Register environment with Azure ML:
echo    python scripts/azure_manage_data_assets.py build-environment --name my-ml-env
echo.
echo 2. Submit training job:
echo    python scripts/azure_submit_job.py --environment my-ml-env
echo.

pause
exit /b 0

:test_failed
echo ERROR: Container test failed.
echo Check your Dockerfile and dependencies.
pause
exit /b 1

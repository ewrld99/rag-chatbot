<#
.SYNOPSIS
Builds the backend and frontend Docker images and pushes them to DockerHub.

.DESCRIPTION
This script automates the process of building the Docker images for the RAG Chatbot
(both backend and frontend) and pushing them to your personal DockerHub repository.

.PARAMETER Username
Your DockerHub username. If not provided, it will prompt you.

.PARAMETER Version
The version tag for the images (default: "latest").
#>

param (
    [string]$Username,
    [string]$Version = "latest"
)

# 1. Prompt for username if not provided
if (-not $Username) {
    $Username = Read-Host "Please enter your DockerHub username"
}

if (-not $Username) {
    Write-Error "DockerHub username is required."
    exit 1
}

$BackendImage = "$Username/rag-backend:$Version"
$FrontendImage = "$Username/rag-frontend:$Version"

# 2. Login to DockerHub
Write-Host "`n=== Logging into DockerHub ===" -ForegroundColor Cyan
docker login

# 3. Build the Backend
Write-Host "`n=== Building Backend Image ($BackendImage) ===" -ForegroundColor Cyan
# The context is the root folder where the backend Dockerfile is
docker build -t $BackendImage .
if ($LASTEXITCODE -ne 0) { Write-Error "Backend build failed"; exit 1 }

# 4. Build the Frontend
Write-Host "`n=== Building Frontend Image ($FrontendImage) ===" -ForegroundColor Cyan
# The context is the rag-frontend folder
docker build -t $FrontendImage ./rag-frontend
if ($LASTEXITCODE -ne 0) { Write-Error "Frontend build failed"; exit 1 }

# 5. Push Images to DockerHub
Write-Host "`n=== Pushing Backend Image to DockerHub ===" -ForegroundColor Cyan
docker push $BackendImage
if ($LASTEXITCODE -ne 0) { Write-Error "Backend push failed"; exit 1 }

Write-Host "`n=== Pushing Frontend Image to DockerHub ===" -ForegroundColor Cyan
docker push $FrontendImage
if ($LASTEXITCODE -ne 0) { Write-Error "Frontend push failed"; exit 1 }

Write-Host "`n=== Success! ===" -ForegroundColor Green
Write-Host "Your images are now available on DockerHub:"
Write-Host "  - $BackendImage"
Write-Host "  - $FrontendImage"

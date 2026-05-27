param(
    [string]$ModelPath = $env:CAMERA_MODEL_PATH,
    [string]$TestVideoPath = $env:CAMERA_TEST_VIDEO_PATH,
    [string]$ModelUrl = $env:CAMERA_MODEL_URL,
    [string]$TestVideoUrl = $env:CAMERA_TEST_VIDEO_URL
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")

function Resolve-AssetPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RawPath
    )

    if ([System.IO.Path]::IsPathRooted($RawPath)) {
        return $RawPath
    }

    return Join-Path $RootDir $RawPath
}

function Download-IfNeeded {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        return $true
    }

    if ([string]::IsNullOrWhiteSpace($Url)) {
        return $false
    }

    $directory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $directory | Out-Null

    $tempPath = "$Destination.tmp"
    Write-Host "[assets] Downloading $Label -> $Destination"
    Invoke-WebRequest -Uri $Url -OutFile $tempPath
    Move-Item -Force -LiteralPath $tempPath -Destination $Destination
    return $true
}

function Assert-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "[assets] ERROR: $Label is missing: $Path"
    }

    $item = Get-Item -LiteralPath $Path
    if ($item.Length -le 0) {
        throw "[assets] ERROR: $Label is empty: $Path"
    }
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = "models/yolov8x.onnx"
}
if ([string]::IsNullOrWhiteSpace($TestVideoPath)) {
    $TestVideoPath = "media/test_video.mp4"
}

$ResolvedModelPath = Resolve-AssetPath $ModelPath
$ResolvedTestVideoPath = Resolve-AssetPath $TestVideoPath

New-Item -ItemType Directory -Force -Path (Join-Path $RootDir "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RootDir "media") | Out-Null

Download-IfNeeded -Label "YOLO model" -Url $ModelUrl -Destination $ResolvedModelPath | Out-Null
if (![string]::IsNullOrWhiteSpace($TestVideoUrl)) {
    Download-IfNeeded -Label "test video" -Url $TestVideoUrl -Destination $ResolvedTestVideoPath | Out-Null
}

Assert-RequiredFile -Label "YOLO model" -Path $ResolvedModelPath

if (!(Test-Path -LiteralPath $ResolvedTestVideoPath -PathType Leaf)) {
    Write-Warning "[assets] optional test video is missing: $ResolvedTestVideoPath"
}

Write-Host "[assets] OK"
Write-Host "[assets] model: $ResolvedModelPath"
Write-Host "[assets] video: $ResolvedTestVideoPath"
Write-Host ""
Write-Host "Docker mount example:"
Write-Host "  -v `"$(Split-Path -Parent $ResolvedModelPath):/models:ro`""
Write-Host "  -v `"$(Split-Path -Parent $ResolvedTestVideoPath):/media:ro`""

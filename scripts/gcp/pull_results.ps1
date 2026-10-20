#Requires -Version 5.1
<#
.SYNOPSIS
  Copy paper-protocol results from lv1-cpu into results/gcp. Does not delete the VM.
#>
param(
    [string] $ProjectId = "knu-fog-experiments",
    [string] $Zone = "us-central1-a",
    [string] $InstanceName = "lv1-cpu",
    [string] $DestName = "gcp",
    [switch] $Execute
)

$ErrorActionPreference = "Stop"
$gcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$LocalCode = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Dest = Join-Path $LocalCode "results\$DestName"
Write-Host "[LV1] pull into $Dest"
if (-not $Execute) {
    Write-Host "[LV1] Dry-run. Pass -Execute to copy."
    exit 0
}

$tarLocal = Join-Path $env:TEMP "lv1_paper_results.tar.gz"
if (Test-Path $tarLocal) { Remove-Item $tarLocal -Force }
$pack = "tar --format=ustar -czf `$HOME/lv1_paper_results.tar.gz -C `$HOME/experiments/client-lottery results"
& $gcloud compute ssh $InstanceName --quiet --zone=$Zone --project=$ProjectId --command=$pack
& $gcloud compute scp --quiet "${InstanceName}:lv1_paper_results.tar.gz" $tarLocal --zone=$Zone --project=$ProjectId
if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
New-Item -ItemType Directory -Path $Dest -Force | Out-Null
tar -xzf $tarLocal -C $Dest --strip-components=1
Write-Host "[LV1] extracted paper results under $Dest"

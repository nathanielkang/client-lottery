#Requires -Version 5.1
<#
.SYNOPSIS
  Stop lv1-cpu. Does not delete the instance or the disk.
#>
param(
    [string] $ProjectId = "knu-fog-experiments",
    [string] $Zone = "us-central1-a",
    [string] $InstanceName = "lv1-cpu",
    [switch] $Execute
)

$ErrorActionPreference = "Stop"
$gcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
Write-Host "[LV1] stop $InstanceName in $Zone (disk is kept)"
if (-not $Execute) {
    Write-Host "[LV1] Dry-run. Pass -Execute to stop."
    exit 0
}
& $gcloud compute instances stop $InstanceName --zone=$Zone --project=$ProjectId
$status = & $gcloud compute instances describe $InstanceName --zone=$Zone --project=$ProjectId --format="value(status)"
Write-Host "[LV1] status=$status"
if ($status -eq "RUNNING") { throw "instance still RUNNING" }

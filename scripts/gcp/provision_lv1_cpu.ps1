#Requires -Version 5.1
<#
.SYNOPSIS
  Create lv1-cpu on an existing project. Refuses a project with billing disabled.
#>
param(
    [string] $ProjectId = "knu-fog-experiments",
    [string] $Zone = "us-central1-a",
    [string] $MachineType = "e2-standard-8",
    [string] $InstanceName = "lv1-cpu",
    [switch] $Execute
)

$ErrorActionPreference = "Stop"
$gcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path $gcloud)) { throw "gcloud not found at $gcloud" }

Write-Host "[LV1] project=$ProjectId instance=$InstanceName zone=$Zone machine=$MachineType"
Write-Host "[LV1] billing check runs before any create"

if (-not $Execute) {
    Write-Host "[LV1] Dry-run. Pass -Execute to create after the billing check."
    exit 0
}

& $gcloud config set project $ProjectId
$billing = & $gcloud billing projects describe $ProjectId --format="value(billingEnabled)"
if ($billing -ne "True") {
    throw "billingEnabled=$billing for $ProjectId. Not creating an instance."
}
& $gcloud compute instances create $InstanceName `
    --zone=$Zone `
    --machine-type=$MachineType `
    --image-family=ubuntu-2204-lts `
    --image-project=ubuntu-os-cloud `
    --boot-disk-size=50GB `
    --boot-disk-type=pd-ssd `
    --scopes=cloud-platform
Write-Host "[LV1] create submitted for $InstanceName"

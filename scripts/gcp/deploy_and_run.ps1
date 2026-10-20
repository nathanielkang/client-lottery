#Requires -Version 5.1
<#
.SYNOPSIS
  Zip this tree to a temp folder, copy it to the instance, and start the protocol in tmux.
#>
param(
    [string] $ProjectId = "knu-fog-experiments",
    [string] $Zone = "us-central1-a",
    [string] $InstanceName = "lv1-cpu",
    [string] $Protocol = "paper",
    [int] $T = 0,
    [string] $Seeds = "",
    [switch] $Execute
)

$ErrorActionPreference = "Stop"
$gcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path $gcloud)) { throw "gcloud not found at $gcloud" }

$LocalCode = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Staging = Join-Path $env:TEMP "lv1_staging_2code"
$ZipPath = Join-Path $env:TEMP "lv1_2code_deploy.zip"
if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
New-Item -ItemType Directory -Path $Staging | Out-Null
foreach ($item in @("clientlottery", "scripts", "requirements.txt", "README.md")) {
    Copy-Item -Path (Join-Path $LocalCode $item) -Destination (Join-Path $Staging $item) -Recurse -Force
}
py -c "from pathlib import Path; root=Path(r'$Staging');
[p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n')) for p in root.rglob('*.sh')]"
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
py -c @"
import zipfile
from pathlib import Path
root = Path(r'$Staging')
out = Path(r'$ZipPath')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and 'results' not in p.parts:
            zf.write(p, p.relative_to(root).as_posix())
print('zip_ok', out)
"@

Write-Host "[LV1] zip=$ZipPath"
if (-not $Execute) {
    Write-Host "[LV1] Dry-run. Pass -Execute to copy and start tmux."
    exit 0
}

& $gcloud config set project $ProjectId
$copied = $false
for ($try = 1; $try -le 8 -and -not $copied; $try++) {
    & $gcloud compute scp --quiet $ZipPath "${InstanceName}:lv1_2code_deploy.zip" --zone=$Zone --project=$ProjectId
    if ($LASTEXITCODE -eq 0) { $copied = $true }
    else {
        Write-Host "[LV1] scp not ready (try $try); waiting"
        Start-Sleep -Seconds 15
    }
}
if (-not $copied) { throw "could not copy the code zip" }

$bootstrap = Join-Path $env:TEMP "lv1_remote_bootstrap.sh"
$bootstrapText = @"
set -euo pipefail
python3 - <<'PY'
import shutil
import zipfile
from pathlib import Path
zip_path = Path.home() / "lv1_2code_deploy.zip"
dest = Path.home() / "experiments" / "client-lottery"
if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(dest)
print("extract_ok", dest)
PY
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip tmux
export PROTOCOL='$Protocol'
$(if ($T -gt 0) { "export T='$T'" } else { "unset T" })
$(if ($Seeds) { "export SEEDS='$Seeds'" } else { "unset SEEDS" })
bash "`$HOME/experiments/client-lottery/scripts/gcp/remote_setup.sh"
"@
[System.IO.File]::WriteAllText($bootstrap, $bootstrapText.Replace("`r`n", "`n"))

& $gcloud compute scp --quiet $bootstrap "${InstanceName}:lv1_remote_bootstrap.sh" --zone=$Zone --project=$ProjectId
if ($LASTEXITCODE -ne 0) { throw "bootstrap scp failed" }
& $gcloud compute ssh $InstanceName --quiet --zone=$Zone --project=$ProjectId --command="bash ~/lv1_remote_bootstrap.sh"
if ($LASTEXITCODE -ne 0) { throw "remote bootstrap failed" }
Write-Host "[LV1] tmux session lv1 started"

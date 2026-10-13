# Fictional asset files and mocked download; never opens a camera or Docker.
param([string]$Library, [string]$FixtureRoot, [string]$Scenario)
$ErrorActionPreference = 'Stop'; Set-StrictMode -Version Latest
. $Library
if ($Scenario -eq 'junction') {
    function Get-Item {
        param([string]$LiteralPath, [switch]$Force)
        [pscustomobject]@{LinkType='Junction'}
    }
}
if ($Scenario -eq 'cloud') {
    function Get-Item {
        param([string]$LiteralPath, [switch]$Force)
        $value=Microsoft.PowerShell.Management\Get-Item -LiteralPath $LiteralPath -Force:$Force
        [pscustomobject]@{LinkType=$null;Length=$(if($value.PSIsContainer){0}else{$value.Length});Attributes=[IO.FileAttributes]::ReparsePoint}
    }
}
$script:downloads = 0
function Invoke-EmoFaceModelDownload([string]$Root) {
    $script:downloads++
    if ($Scenario -eq 'download_failed') { throw 'Fictional download failure.' }
    foreach ($item in $manifest.files) {
        [IO.File]::WriteAllBytes((Join-Path $Root $item.asset), [Text.Encoding]::UTF8.GetBytes('fixture-' + $item.runtime))
    }
}
$models = Join-Path $FixtureRoot 'models'
$manifest = Get-Content (Join-Path $FixtureRoot 'configs/model-assets.json') -Raw | ConvertFrom-Json
$before = @{}
if (Test-Path $models) {
    Get-ChildItem $models -Recurse -File | ForEach-Object { $before[$_.FullName] = Get-EmoFaceModelHash $_.FullName }
}
$failed = $false; $failure = ''
try { Initialize-EmoFaceModelAssets $FixtureRoot $models } catch { $failed = $true; $failure = $_.Exception.Message.Replace($FixtureRoot, '<fixture>') }
$preserved = $true
foreach ($name in $before.Keys) { if (-not [IO.File]::Exists($name) -or (Get-EmoFaceModelHash $name) -ne $before[$name]) { $preserved = $false } }
$count = 0
if (Test-Path $models) { $count = @(Get-ChildItem $models -Recurse -File).Count }
[pscustomobject]@{failed=$failed;failure=$failure;downloads=$script:downloads;runtime_files=$count;existing_preserved=$preserved} | ConvertTo-Json -Compress

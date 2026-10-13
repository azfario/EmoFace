# Requires Windows PowerShell 5.1+ and Docker Desktop Linux containers with GPU support.
[CmdletBinding()]
param(
    [string]$ModelsDirectory,
    [string]$SecretsDirectory,
    [string]$ProjectName,
    [switch]$NoBrowser,
    [switch]$Stop
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'docker/Launcher.ps1')
try {
    Start-EmoFace -Root $PSScriptRoot -ModelsDirectory $ModelsDirectory -SecretsDirectory $SecretsDirectory -ProjectName $ProjectName -NoBrowser:$NoBrowser -Stop:$Stop
} catch {
    # All launcher errors are deliberately coarse; never dump native stderr or input.
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}

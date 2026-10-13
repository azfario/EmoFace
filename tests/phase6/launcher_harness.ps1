# Fictional orchestration tests; all Docker calls and secret-file mutations are mocked.
param([string]$Library, [string]$FixtureRoot, [string]$Scenario)
$ErrorActionPreference = 'Stop'
. $Library
$script:commands = New-Object 'System.Collections.Generic.List[object]'
$script:created = 0; $script:validated = 0; $script:payloadValid = $false
$script:promptCount = 0
function Invoke-EmoFaceDocker {
    param([string[]]$Arguments, [string]$InputText, [int]$TimeoutSeconds)
    $script:commands.Add(@($Arguments))
    $value = ''; $code = 0
    if ($Arguments[0] -eq 'info') { $value = 'linux' }
    if ($Arguments[0] -eq 'context') {
        $value = '"npipe:////./pipe/dockerDesktopLinuxEngine"'
        if ($Scenario -eq 'remote') { $value = '"ssh://remote.invalid"' }
    }
    if ($Arguments[0] -eq 'ps' -and $Scenario -in @('running','partial')) { $value = 'fictional-owned-api' }
    if ($Arguments[0] -eq 'compose' -and $Arguments -contains 'ps') {
        $value = @('api','db','frontend' | ForEach-Object { @{Service=$_;State='running';Health='healthy'} | ConvertTo-Json -Compress }) -join "`n"
        if ($Scenario -eq 'partial') { $value = '{"Service":"frontend","State":"running","Health":"healthy"}' }
    }
    if ($Arguments[0] -eq 'volume' -and $Scenario -eq 'retained') { $value = 'fictional_launcher_db_data' }
    if ($Arguments -contains 'docker/gpu_check.py' -and $Scenario -eq 'gpu_failure') { $code = 1 }
    if ($Arguments -contains 'create') {
        $script:payloadValid = $InputText -eq '{}'
    }
    if ($Arguments -contains 'up' -and $Scenario -eq 'startup_failure') { $code = 1 }
    [pscustomobject]@{Code=$code;Output=$value}
}
function Test-EmoFacePort { $Scenario -eq 'occupied' }
function New-EmoFacePrivateDirectory { param([string]$Path) $script:created++ }
function Assert-EmoFaceSecretFiles {
    param([string]$Path)
    $script:validated++
    if ($Scenario -eq 'incomplete') { throw 'Existing secret set is incomplete.' }
}
function Read-Host {
    param([string]$Prompt, [switch]$AsSecureString)
    $script:promptCount++
    throw 'The password-free launcher must never prompt.'
}
$selectedSecrets = Join-Path $env:LOCALAPPDATA ('FaceAttend/phase6/private/launcher_unit_' + [guid]::NewGuid().ToString('N'))
if ($Scenario -in @('reuse','incomplete')) { $selectedSecrets = $env:LOCALAPPDATA }
if ($Scenario -eq 'synced') { $selectedSecrets = Join-Path $env:LOCALAPPDATA 'OneDrive/launcher_test_only' }
$failed = $false; $failure = ''
try {
    if ($Scenario -in @('acl_good','acl_bad')) {
        function Get-Acl {
            param([string]$LiteralPath)
            $ids=@([Security.Principal.WindowsIdentity]::GetCurrent().User.Value,'S-1-5-18','S-1-5-32-544')
            if($Scenario -eq 'acl_bad'){$ids+=@('S-1-1-0')}
            $rules=@($ids | ForEach-Object {
                [pscustomobject]@{AccessControlType=[Security.AccessControl.AccessControlType]::Allow;IdentityReference=(New-Object Security.Principal.SecurityIdentifier($_))}
            })
            [pscustomobject]@{Access=$rules}
        }
        Assert-EmoFacePrivateAcl $selectedSecrets
    } else {
        Start-EmoFace -Root $FixtureRoot -ModelsDirectory (Join-Path $FixtureRoot 'models') -SecretsDirectory $selectedSecrets -ProjectName 'fictional_launcher' -NoBrowser -Stop:($Scenario -eq 'stop')
    }
} catch { $failed = $true; $failure = $_.Exception.Message }
$creates = @($script:commands | Where-Object { $_ -contains 'create' })
$validations = @($script:commands | Where-Object { $_ -contains 'docker/bootstrap_secrets.py' -and $_ -contains 'validate' })
$sensitive = @($script:commands | Where-Object { $_ -contains 'docker/bootstrap_secrets.py' })
$safeRuns = @($sensitive | Where-Object { $_ -contains 'none' -and $_ -contains '--log-driver' -and $_ -contains '--network' -and $_ -contains '--read-only' -and $_ -contains '--cap-drop' })
$startup = @($script:commands | Where-Object { $_ -contains 'up' })
$destructive = @($script:commands | Where-Object { $_ -contains 'down' -or $_ -contains 'prune' -or $_ -contains '--volumes' })
[pscustomobject]@{
    failed=$failed;failure=$failure;create_calls=$creates.Count;validate_calls=$validations.Count
    directory_creations=$script:created;prompts=$script:promptCount
    payload_valid=$script:payloadValid;secret_runs_restricted=($safeRuns.Count -eq $sensitive.Count)
    startup_calls=$startup.Count;destructive_calls=$destructive.Count
    plaintext_in_argv=(@($script:commands | Where-Object { ($_ -join ' ') -match 'Fictional launcher passphrase' }).Count -gt 0)
} | ConvertTo-Json -Compress

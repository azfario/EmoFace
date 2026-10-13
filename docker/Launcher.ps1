Set-StrictMode -Version Latest
# Resolve the built-in ACL module from this PowerShell edition, including when
# Windows PowerShell 5.1 inherits a PSModulePath from PowerShell 7.
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Security/Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
. (Join-Path $PSScriptRoot 'ModelAssets.ps1')

function ConvertTo-EmoFaceNativeArgument([string]$Value) {
    '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function Invoke-EmoFaceDocker {
    param([string[]]$Arguments, [string]$InputText, [int]$TimeoutSeconds = 1800)
    $command = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $command) { throw 'Install Docker Desktop and start its Linux engine before running this launcher.' }
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $command.Source
    $info.Arguments = ($Arguments | ForEach-Object { ConvertTo-EmoFaceNativeArgument $_ }) -join ' '
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $info.StandardErrorEncoding = New-Object System.Text.UTF8Encoding($false)
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $output = $process.StandardOutput.ReadToEndAsync()
        $errors = $process.StandardError.ReadToEndAsync()
        if ($null -ne $InputText) {
            # Explicit UTF-8 stdin preserves non-ASCII passwords on Windows PowerShell.
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($InputText)
            $process.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
            [Array]::Clear($bytes, 0, $bytes.Length)
        }
        $process.StandardInput.Close()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill()
            throw 'Docker command timed out. Check Docker Desktop; no data or secrets were removed.'
        }
        $process.WaitForExit()
        [pscustomobject]@{ Code = $process.ExitCode; Output = $output.Result }
        # Native stderr is consumed but withheld: it can include private mount paths.
        [void]$errors.Result
    } catch {
        throw 'Docker command could not complete. Check Docker Desktop; no data or secrets were removed.'
    } finally {
        $InputText = $null
        $process.Dispose()
    }
}

function Invoke-EmoFaceRequired {
    param([string[]]$Arguments, [string]$Message, [string]$InputText, [int]$TimeoutSeconds = 1800)
    $result = Invoke-EmoFaceDocker -Arguments $Arguments -InputText $InputText -TimeoutSeconds $TimeoutSeconds
    if ($result.Code -ne 0) { throw $Message }
    $result.Output
}

function Resolve-EmoFaceSecretPath([string]$Value, [string]$Root) {
    if (-not [IO.Path]::IsPathRooted($Value) -or $Value.StartsWith('\\') -or $Value.Contains(',')) {
        throw 'Use an absolute local secret directory outside the checkout and synced folders.'
    }
    try { $path = [IO.Path]::GetFullPath($Value).TrimEnd('\', '/') } catch { throw 'The selected secret directory is invalid.' }
    $excluded = @($Root, $env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)
    foreach ($entry in $excluded) {
        if (-not $entry) { continue }
        $base = [IO.Path]::GetFullPath($entry).TrimEnd('\', '/')
        if ($path.Equals($base, [StringComparison]::OrdinalIgnoreCase) -or $path.StartsWith($base + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Secret storage must be outside the checkout and OneDrive.'
        }
    }
    if ($path -match '(?i)(^|[\\/])onedrive[^\\/]*([\\/]|$)') { throw 'Secret storage must be outside synced folders.' }
    $ancestor = $path
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Secret storage must not use junctions, symlinks or cloud placeholders.' }
        }
        $parent = [IO.Directory]::GetParent($ancestor)
        if ($null -eq $parent) { break }
        $ancestor = $parent.FullName
    }
    $path
}

function Assert-EmoFacePrivateAcl([string]$Path) {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $allowed = @($current, 'S-1-5-18', 'S-1-5-32-544')
    $acl = Get-Acl -LiteralPath $Path
    foreach ($rule in $acl.Access) {
        if ($rule.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow) {
            $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
            if ($sid -notin $allowed) { throw 'The secret directory or files allow other accounts. Restore protected ACLs before starting.' }
        }
    }
}

function New-EmoFacePrivateDirectory([string]$Path) {
    if (Test-Path -LiteralPath $Path) { throw 'Refusing to recreate an existing secret directory.' }
    [void][IO.Directory]::CreateDirectory($Path)
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $owner = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl.SetOwner($owner)
    foreach ($sid in @($owner.Value, 'S-1-5-18', 'S-1-5-32-544')) {
        $identity = New-Object Security.Principal.SecurityIdentifier($sid)
        $rule = New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')
        [void]$acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
    Assert-EmoFacePrivateAcl $Path
}

function Assert-EmoFaceSecretFiles([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw 'The protected secret directory is unavailable.' }
    Assert-EmoFacePrivateAcl $Path
    $names = @('db_admin_password', 'db_owner_password', 'db_runtime_password', 'gallery_key', 'operator_auth')
    $files = @(Get-ChildItem -LiteralPath $Path -Force)
    if ($files.Count -ne 5) { throw 'The existing secret set is incomplete. Restore it; the launcher will not replace keys.' }
    foreach ($file in $files) {
        if ($file.Name -notin $names -or $file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $file.Length -lt 1 -or $file.Length -gt 4096) {
            throw 'The existing secret set is invalid. Restore it; the launcher will not replace keys.'
        }
        Assert-EmoFacePrivateAcl $file.FullName
    }
}

function Test-EmoFacePort {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect('127.0.0.1', 18080, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(300)) { return $false }
        $client.EndConnect($pending)
        return $true
    } catch { return $false } finally { $client.Close() }
}

function Start-EmoFace {
    param([string]$Root, [string]$ModelsDirectory, [string]$SecretsDirectory, [string]$ProjectName, [switch]$NoBrowser, [switch]$Stop)
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'This launcher targets Windows; use the documented Compose setup on other platforms.' }
    if (-not $ModelsDirectory) { $ModelsDirectory = $env:EMOFACE_MODELS_DIR }
    if (-not $ModelsDirectory) { $ModelsDirectory = Join-Path $Root 'models' }
    if (-not $SecretsDirectory) { $SecretsDirectory = $env:EMOFACE_SECRETS_DIR }
    if (-not $SecretsDirectory) { $SecretsDirectory = Join-Path $env:LOCALAPPDATA 'FaceAttend/phase6/private/operator_secrets' }
    if (-not $ProjectName) { $ProjectName = $env:COMPOSE_PROJECT_NAME }
    if (-not $ProjectName) { $ProjectName = 'emoface_local' }
    if ($ProjectName -notmatch '^[a-z0-9][a-z0-9_-]{0,62}$') { throw 'Use a lowercase Compose project name containing letters, numbers, underscores or hyphens.' }
    try { $models = [IO.Path]::GetFullPath($ModelsDirectory) } catch { throw 'The selected model directory is invalid.' }
    if ($models.Contains(',')) { throw 'The model directory cannot contain a comma.' }
    $secrets = Resolve-EmoFaceSecretPath -Value $SecretsDirectory -Root $Root
    $oldEnvironment = @{}
    foreach ($name in @('EMOFACE_MODELS_DIR', 'EMOFACE_SECRETS_DIR', 'COMPOSE_PROJECT_NAME')) { $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    $mutex = $null; $owned = $false; $success = $false
    try {
        $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $mutex = New-Object Threading.Mutex($false, ('Local\EmoFaceStartup_' + $sid + '_' + $ProjectName))
        try { $owned = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $owned = $true }
        if (-not $owned) { throw 'Another launcher is already configuring this project. Wait for it to finish.' }
        if (-not $Stop) {
            if ($models.Equals([IO.Path]::GetFullPath((Join-Path $Root 'models')), [StringComparison]::OrdinalIgnoreCase)) {
                Initialize-EmoFaceModelAssets $Root $models
            }
            foreach ($relative in @('faceattend_best.pt', 'external/insightface/models/buffalo_l/det_10g.onnx', 'external/insightface/models/buffalo_l/w600k_r50.onnx')) {
                $path = Join-Path $models $relative
                if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0 -or (Test-EmoFaceLFSPointer $path)) {
                    throw 'Model files are missing or still Git LFS pointers. Install Git LFS and pull the repository assets, or supply valid models at the README runtime paths.'
                }
            }
        }
        $engine = Invoke-EmoFaceRequired -Arguments @('info', '--format', '{{.OSType}}') -Message 'Start Docker Desktop with its Linux engine and WSL2 enabled.' -TimeoutSeconds 30
        if ($engine.Trim() -ne 'linux') { throw 'Switch Docker Desktop to Linux containers with WSL2 GPU support.' }
        $endpoint = Invoke-EmoFaceRequired -Arguments @('context', 'inspect', '--format', '{{json .Endpoints.docker.Host}}') -Message 'The selected Docker context is unavailable.' -TimeoutSeconds 30
        if (($endpoint | ConvertFrom-Json) -notmatch '^npipe://') { throw 'Use the local Docker Desktop context; remote Docker contexts are refused.' }
        [void](Invoke-EmoFaceRequired -Arguments @('compose', 'version') -Message 'Docker Compose v2 is required.' -TimeoutSeconds 30)
        $env:EMOFACE_MODELS_DIR = $models; $env:EMOFACE_SECRETS_DIR = $secrets; $env:COMPOSE_PROJECT_NAME = $ProjectName
        $compose = @('compose', '-f', (Join-Path $Root 'compose.yaml'), '-p', $ProjectName)
        if ($Stop) {
            [void](Invoke-EmoFaceRequired -Arguments ($compose + @('down')) -Message 'Shutdown failed. Preserve the selected project data and secrets.' -TimeoutSeconds 120)
            $success = $true
            Write-Host 'Selected EmoFace project stopped. Its volumes, models and secrets were retained.'
            return
        }
        $running = Invoke-EmoFaceRequired -Arguments @('ps', '--filter', ('label=com.docker.compose.project=' + $ProjectName), '--format', '{{.Names}}') -Message 'Docker container state is unavailable.' -TimeoutSeconds 30
        if ($running.Trim()) {
            $status = Invoke-EmoFaceRequired -Arguments ($compose + @('ps', '--format', 'json')) -Message 'Existing project health is unavailable.' -TimeoutSeconds 30
            $services = @()
            foreach ($line in ($status -split '\r?\n')) {
                if ($line.Trim()) { $services += @($line | ConvertFrom-Json) }
            }
            foreach ($required in @('api','db','frontend')) {
                $matching = @($services | Where-Object { $_.Service -eq $required -and $_.State -eq 'running' -and $_.Health -eq 'healthy' })
                if ($matching.Count -ne 1) { throw 'This project is partially started or unhealthy. Run the launcher with -Stop for the same project, then retry. Data and secrets will be retained.' }
            }
            Write-Host 'This project is already running. Open http://127.0.0.1:18080/app/. Stop the whole stack before rebuilding or changing configuration.'
            $success = $true
            return
        }
        if (Test-EmoFacePort) { throw 'Port 18080 is occupied by another service. Stop that service before starting this project.' }
        $volumes = Invoke-EmoFaceRequired -Arguments @('volume', 'ls', '--format', '{{.Name}}') -Message 'Retained-volume state is unavailable.' -TimeoutSeconds 30
        $hasData = @($volumes -split '\r?\n' | Where-Object { $_ -in @(($ProjectName + '_db_data'), ($ProjectName + '_private_data')) }).Count -gt 0
        $createSecrets = -not (Test-Path -LiteralPath $secrets)
        if ($createSecrets -and $hasData) { throw 'Retained data exists but matching secrets are missing. Restore the original secrets; new keys will not be generated.' }
        if (-not $createSecrets) { Assert-EmoFaceSecretFiles $secrets }
        Write-Host 'Building the app image. First use downloads its dependencies; local Python and Node are not required.'
        [void](Invoke-EmoFaceRequired -Arguments @('build', '--tag', ($ProjectName + '-api'), '--file', (Join-Path $Root 'docker/Dockerfile.api'), $Root) -Message 'App image build failed. Check Docker, internet access and available disk space.')
        $restricted = @('run', '--rm', '--log-driver', 'none', '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges')
        Write-Host 'Checking actual model inference on the NVIDIA GPU.'
        [void](Invoke-EmoFaceRequired -Arguments ($restricted + @('--gpus', 'all', '--tmpfs', '/tmp:rw,noexec,nosuid,size=512m,uid=10001,gid=10001,mode=700', '--tmpfs', '/app/.cache:rw,noexec,nosuid,size=128m,uid=10001,gid=10001,mode=700', '--mount', ('type=bind,source=' + $models + ',target=/app/models,readonly'), ($ProjectName + '-api'), 'python', 'docker/gpu_check.py')) -Message 'GPU/model check failed. Verify the supplied models, NVIDIA driver and Docker WSL2 GPU support. CPU fallback is disabled.' -TimeoutSeconds 180)
        if ($createSecrets) {
            New-EmoFacePrivateDirectory $secrets
            [void](Invoke-EmoFaceRequired -Arguments ($restricted + @('-i', '--mount', ('type=bind,source=' + $secrets + ',target=/setup'), ($ProjectName + '-api'), 'python', 'docker/bootstrap_secrets.py', 'create')) -Message 'Initial secret generation failed. Existing files were not replaced; inspect the protected directory before retrying.' -InputText '{}' -TimeoutSeconds 30)
        }
        Assert-EmoFaceSecretFiles $secrets
        [void](Invoke-EmoFaceRequired -Arguments ($restricted + @('--mount', ('type=bind,source=' + $secrets + ',target=/setup,readonly'), ($ProjectName + '-api'), 'python', 'docker/bootstrap_secrets.py', 'validate')) -Message 'Existing secrets failed validation. Restore the matching secret set; no keys were replaced.' -TimeoutSeconds 30)
        [void](Invoke-EmoFaceRequired -Arguments ($compose + @('config', '--quiet')) -Message 'Compose configuration is invalid; verify local model and protected secret settings.' -TimeoutSeconds 30)
        Write-Host 'Building the frontend and starting the localhost services.'
        [void](Invoke-EmoFaceRequired -Arguments ($compose + @('build', 'frontend')) -Message 'Frontend build failed. Check Docker, internet access and available disk space.')
        [void](Invoke-EmoFaceRequired -Arguments ($compose + @('up', '-d', '--wait', '--wait-timeout', '180')) -Message 'Startup failed. Preserve secrets and volumes; check Compose service health before retrying.' -TimeoutSeconds 240)
        $success = $true
        Write-Host 'EmoFace is ready at http://127.0.0.1:18080/app/. No login is required; start the camera and confirm consent yourself.'
        Write-Host 'To stop: run this launcher with -Stop and the same project settings. Keep the retained volumes and matching secrets.'
        if (-not $NoBrowser) {
            try { Start-Process 'http://127.0.0.1:18080/app/' -WindowStyle Hidden }
            catch { Write-Host 'The app is ready. Open http://127.0.0.1:18080/app/ manually in your browser.' }
        }
    } finally {
        if (-not $success) {
            foreach ($name in $oldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process') }
        }
        if ($owned) { $mutex.ReleaseMutex() }
        if ($null -ne $mutex) { $mutex.Dispose() }
    }
}

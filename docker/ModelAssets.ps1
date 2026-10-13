# Prepare missing runtime models from the three reviewed Git LFS assets.
function Get-EmoFaceModelHash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path); $hash = [Security.Cryptography.SHA256]::Create()
    try { ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hash.Dispose() }
}

function Test-EmoFaceLFSPointer([string]$Path) {
    if (-not [IO.File]::Exists($Path)) { return $false }
    $stream = [IO.File]::OpenRead($Path)
    try {
        $buffer = New-Object byte[] 128
        $count = $stream.Read($buffer, 0, $buffer.Length)
        return [Text.Encoding]::ASCII.GetString($buffer, 0, $count).StartsWith('version https://git-lfs.github.com/spec/v1')
    } finally { $stream.Dispose() }
}

function Invoke-EmoFaceModelDownload([string]$Root) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) { throw 'Install Git with Git LFS and clone this private repository to download its model assets.' }
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $git.Source
    $args = @('-C', $Root, 'lfs', 'pull', '--include=model-assets/faceattend_best.pt,model-assets/det_10g.onnx,model-assets/w600k_r50.onnx', '--exclude=')
    $info.Arguments = ($args | ForEach-Object { ConvertTo-EmoFaceNativeArgument $_ }) -join ' '
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $info.EnvironmentVariables['GIT_TERMINAL_PROMPT'] = '0'
    $info.EnvironmentVariables['GCM_INTERACTIVE'] = 'never'
    $process = New-Object Diagnostics.Process; $process.StartInfo = $info
    try {
        [void]$process.Start()
        $output = $process.StandardOutput.ReadToEndAsync(); $errors = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(300000)) { $process.Kill(); throw 'Model download timed out.' }
        $process.WaitForExit(); [void]$output.Result; [void]$errors.Result
        if ($process.ExitCode -ne 0) { throw 'Model download failed.' }
    } catch {
        throw 'Model download failed. Install Git LFS, sign in to this private repository, and check LFS storage/bandwidth before retrying. Existing models were preserved.'
    } finally { $process.Dispose() }
}

function Assert-EmoFaceModelPath([string]$Path, [string]$Base) {
    $basePath = [IO.Path]::GetFullPath($Base).TrimEnd('\', '/')
    $pathValue = [IO.Path]::GetFullPath($Path)
    if (-not $pathValue.StartsWith($basePath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid model asset path.' }
    $current = $pathValue
    while ($current -and $current.Length -ge $basePath.Length) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            # OneDrive uses reparse points for ordinary hydrated files too.
            # Reject actual links while permitting the user's synced checkout.
            if ($item.LinkType -in @('SymbolicLink', 'Junction')) {
                throw 'Model asset paths must not use symlinks or junctions.'
            }
        }
        $parent = [IO.Directory]::GetParent($current)
        if ($null -eq $parent) { break }; $current = $parent.FullName
    }
}

function Initialize-EmoFaceModelAssets([string]$Root, [string]$ModelsDirectory) {
    $expected = @{
        'model-assets/faceattend_best.pt' = 'faceattend_best.pt'
        'model-assets/det_10g.onnx' = 'external/insightface/models/buffalo_l/det_10g.onnx'
        'model-assets/w600k_r50.onnx' = 'external/insightface/models/buffalo_l/w600k_r50.onnx'
    }
    $missing = @()
    foreach ($runtime in $expected.Values) {
        $target = Join-Path $ModelsDirectory $runtime
        if ([IO.File]::Exists($target)) {
            if ((Get-Item -LiteralPath $target).Length -eq 0 -or (Test-EmoFaceLFSPointer $target)) {
                throw 'A runtime model is empty or still a Git LFS pointer. Restore the file; existing models were not overwritten.'
            }
        } else { $missing += $runtime }
    }
    if ($missing.Count -eq 0) { return }
    $manifestPath = Join-Path $Root 'configs/model-assets.json'
    # Compatibility with source fixtures and explicitly supplied model folders.
    if (-not [IO.File]::Exists($manifestPath)) { return }
    try { $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json }
    catch { throw 'The model asset manifest is invalid.' }
    if ($manifest.version -ne 1 -or @($manifest.files).Count -ne 3) { throw 'The model asset manifest is invalid.' }
    $seen = @{}
    foreach ($item in $manifest.files) {
        if (-not $expected.ContainsKey([string]$item.asset) -or $expected[[string]$item.asset] -ne $item.runtime -or $seen.ContainsKey([string]$item.asset) -or
            $item.sha256 -notmatch '^[a-f0-9]{64}$' -or $item.bytes -is [bool] -or $item.bytes -notmatch '^[1-9][0-9]*$' -or [long]$item.bytes -gt 268435456) {
            throw 'The model asset manifest is invalid.'
        }
        $seen[[string]$item.asset] = $true
        Assert-EmoFaceModelPath (Join-Path $Root $item.asset) $Root
        Assert-EmoFaceModelPath (Join-Path $ModelsDirectory $item.runtime) $ModelsDirectory
    }
    $download = $false
    foreach ($item in $manifest.files) {
        $source = Join-Path $Root $item.asset
        if (-not [IO.File]::Exists($source) -or (Test-EmoFaceLFSPointer $source)) { $download = $true }
    }
    if ($download) { Write-Host 'Downloading the three repository model assets through Git LFS.'; Invoke-EmoFaceModelDownload $Root }
    # Validate the whole asset set before creating any runtime file.
    foreach ($item in $manifest.files) {
        $source = Join-Path $Root $item.asset
        Assert-EmoFaceModelPath $source $Root
        if (-not [IO.File]::Exists($source) -or (Get-Item -LiteralPath $source).Length -ne [long]$item.bytes -or (Get-EmoFaceModelHash $source) -ne $item.sha256) {
            throw 'Model asset verification failed. Check Git LFS download and the repository manifest. Existing models were preserved.'
        }
    }
    foreach ($item in $manifest.files) {
        if ($item.runtime -notin $missing) { continue }
        $target = Join-Path $ModelsDirectory $item.runtime
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
        Assert-EmoFaceModelPath $target $ModelsDirectory
        # Exclusive creation prevents replacement of an existing local model.
        [IO.File]::Copy((Join-Path $Root $item.asset), $target, $false)
        if ((Get-EmoFaceModelHash $target) -ne $item.sha256) { throw 'Runtime model verification failed. Restore the incomplete model before retrying.' }
    }
    Write-Host 'Repository model assets verified and prepared. Existing local models were retained.'
}

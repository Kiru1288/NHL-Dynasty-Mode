# Neural commentary voices (Kokoro + Piper) for the live sim: one-time setup + start.
#
#   .\start_tts.ps1              set up whatever is missing, start the worker, print status
#   .\start_tts.ps1 -Restart     restart the worker (e.g. after editing voices.json)
#   .\start_tts.ps1 -Stop        stop the worker
#   .\start_tts.ps1 -Foreground  run the worker in this window (Ctrl+C to stop)
#   .\start_tts.ps1 -Int8        use the smaller int8 Kokoro model (~90 MB instead of ~310 MB)
#   .\start_tts.ps1 -ExtraVoices also download fallback Piper voices (fr siwis, es davefx)
#   .\start_tts.ps1 -Python "C:\Path\To\python.exe"   use a specific Python 3.10-3.13
#
# Idempotent: re-running skips finished steps. Afterwards start the API as usual
# (.\start_api.ps1) - the backend finds the worker on 127.0.0.1:8765 by itself and
# even restarts it if it is not running. Or use .\start_all.ps1 to do both.
#
# If PowerShell refuses to run scripts:  powershell -ExecutionPolicy Bypass -File .\start_tts.ps1
param(
    [switch]$Restart,
    [switch]$Stop,
    [switch]$Foreground,
    [switch]$Int8,
    [switch]$ExtraVoices,
    [switch]$NoStart,
    [string]$Python = "",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x slower with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Backend   = $PSScriptRoot
$WorkerDir = Join-Path $Backend "tts_worker"
$Venv      = Join-Path $WorkerDir "venv"
$VenvPy    = Join-Path $Venv "Scripts\python.exe"
$DataDir   = Join-Path $Backend "data\tts"
$PiperDir  = Join-Path $DataDir "piper"
$LogDir    = Join-Path $Backend "logs"
$PidFile   = Join-Path $DataDir "worker.pid"
$Req       = Join-Path $WorkerDir "requirements.txt"
$BaseUrl   = "http://127.0.0.1:$Port"

function Step($n, $msg) { Write-Host ""; Write-Host "[$n/5] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "      OK  $msg" -ForegroundColor Green }
function Done($msg)     { Write-Host "      --  $msg (already done)" -ForegroundColor DarkGray }
function Warn($msg)     { Write-Host "      !!  $msg" -ForegroundColor Yellow }
function Fail($msg)     { Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- helpers
function Invoke-Local([string]$path, [string]$method = "GET", [string]$body = $null, [int]$timeoutMs = 3000) {
    # local HTTP without any system proxy
    $req = [System.Net.HttpWebRequest]::Create("$BaseUrl$path")
    $req.Proxy = $null
    $req.Method = $method
    $req.Timeout = $timeoutMs
    $req.ReadWriteTimeout = $timeoutMs
    if ($body) {
        $bytes = [Text.Encoding]::UTF8.GetBytes($body)
        $req.ContentType = "application/json"
        $req.ContentLength = $bytes.Length
        $s = $req.GetRequestStream(); $s.Write($bytes, 0, $bytes.Length); $s.Close()
    }
    $resp = $req.GetResponse()
    try {
        $ms = New-Object IO.MemoryStream
        $resp.GetResponseStream().CopyTo($ms)
        return @{ Bytes = $ms.ToArray(); Headers = $resp.Headers; Type = $resp.ContentType }
    } finally { $resp.Close() }
}

function Get-Health {
    try {
        $r = Invoke-Local "/health" "GET" $null 1500
        return ([Text.Encoding]::UTF8.GetString($r.Bytes) | ConvertFrom-Json)
    } catch { return $null }
}

function Get-WorkerProcesses {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'tts_worker[\\/]+worker\.py' }
}

function Stop-Worker {
    $procs = @(Get-WorkerProcesses)
    if (Test-Path $PidFile) {
        $p = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($p) { $procs += @(Get-CimInstance Win32_Process -Filter "ProcessId=$p" -ErrorAction SilentlyContinue) }
    }
    foreach ($p in $procs) {
        if ($p) {
            Write-Host "      stopping TTS worker PID $($p.ProcessId)"
            & taskkill.exe /F /PID $p.ProcessId /T 2>$null | Out-Null
        }
    }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
}

function Get-PyVersion([string[]]$cmd) {
    try {
        $exe = $cmd[0]; $rest = @(); if ($cmd.Length -gt 1) { $rest = $cmd[1..($cmd.Length - 1)] }
        $v = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return ($v | Select-Object -First 1).Trim() }
    } catch {}
    return $null
}

function Test-PySupported([string]$v) {
    if (-not $v) { return $false }
    $parts = $v.Split("."); $maj = [int]$parts[0]; $min = [int]$parts[1]
    return ($maj -eq 3 -and $min -ge 10 -and $min -le 13)
}

function Get-Sha256([string]$path) { (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower() }

function Save-File([string]$url, [string]$dest, [string]$sha256 = "") {
    $name = Split-Path $dest -Leaf
    $side = "$dest.sha256"
    if ((Test-Path $dest) -and (Test-Path $side)) {
        $have = (Get-Content $side -Raw).Trim().ToLower()
        if (-not $sha256 -or $have -eq $sha256.ToLower()) { Done $name; return }
    }
    if ((Test-Path $dest) -and -not (Test-Path $side)) {
        # file present from an earlier run but never verified: verify instead of re-downloading
        $h = Get-Sha256 $dest
        if (-not $sha256 -or $h -eq $sha256.ToLower()) { Set-Content -Path $side -Value $h -Encoding ascii; Done "$name (verified)"; return }
        Warn "$name checksum mismatch - downloading again"
        Remove-Item $dest -Force
    }
    $part = "$dest.part"
    Write-Host "      downloading $name ..."
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 3 --retry-delay 2 -C - -s -S -o $part $url
        if ($LASTEXITCODE -ne 0) {
            Remove-Item $part -ErrorAction SilentlyContinue
            & $curl.Source -L --fail --retry 3 --retry-delay 2 -s -S -o $part $url
            if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
        }
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $part
    }
    $h = Get-Sha256 $part
    if ($sha256 -and $h -ne $sha256.ToLower()) {
        Remove-Item $part -Force
        throw "checksum mismatch for $name (expected $sha256, got $h)"
    }
    Move-Item -Force $part $dest
    Set-Content -Path $side -Value $h -Encoding ascii
    $mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
    if ($sha256) { Ok "$name ($mb MB, sha256 verified)" } else { Ok "$name ($mb MB, sha256 recorded)" }
}

function Get-HfSha256([string]$url) {
    # Hugging Face answers a LFS file's HEAD with X-Linked-Etag = its sha256
    try {
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            $hdr = & $curl.Source -s -I $url 2>$null
            foreach ($line in $hdr) {
                if ($line -match '^(?i)x-linked-etag:\s*"?([0-9a-f]{64})"?') { return $Matches[1] }
            }
        }
    } catch {}
    return ""
}

# ---------------------------------------------------------------- stop / restart
if ($Stop) {
    Stop-Worker
    Write-Host "TTS worker stopped."
    exit 0
}

Write-Host "NHL Dynasty - neural commentary voices (Kokoro-82M + Piper), offline, free" -ForegroundColor White

# ---------------------------------------------------------------- 1. venv
Step 1 "Python venv for the worker (backend\tts_worker\venv)"
$venvOk = $false
if (Test-Path $VenvPy) {
    $vv = Get-PyVersion @($VenvPy)
    if (Test-PySupported $vv) { Done "venv with Python $vv"; $venvOk = $true }
    else {
        Warn "existing venv uses Python '$vv' (kokoro-onnx needs 3.10-3.13) - recreating"
        Stop-Worker
        Remove-Item -Recurse -Force $Venv
    }
}
if (-not $venvOk) {
    $pyCmd = $null; $pyVer = $null
    if ($Python) {
        $pyVer = Get-PyVersion @($Python)
        if (-not (Test-PySupported $pyVer)) { Fail "-Python '$Python' is version '$pyVer'; need 3.10-3.13." }
        $pyCmd = @($Python)
    } else {
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($launcher) {
            foreach ($v in @("3.12", "3.11", "3.13", "3.10")) {
                $got = Get-PyVersion @("py", "-$v")
                if ($got -eq $v) { $pyCmd = @("py", "-$v"); $pyVer = $v; break }
            }
        }
        if (-not $pyCmd) {
            foreach ($c in @("python", "python3")) {
                $got = Get-PyVersion @($c)
                if (Test-PySupported $got) { $pyCmd = @($c); $pyVer = $got; break }
                elseif ($got) { Warn "'$c' is Python $got - not supported by onnxruntime/kokoro-onnx yet" }
            }
        }
    }
    if (-not $pyCmd) {
        Write-Host ""
        Write-Host "No Python 3.10-3.13 found. The game's API can stay on 3.14; the voice worker needs its own 3.12." -ForegroundColor Yellow
        Write-Host "Install it (no admin needed), then re-run this script:" -ForegroundColor Yellow
        Write-Host "    winget install -e --id Python.Python.3.12" -ForegroundColor White
        Write-Host "  or download 'Windows installer (64-bit)' from https://www.python.org/downloads/release/python-3128/" -ForegroundColor White
        Fail "Python 3.10-3.13 required for the TTS worker."
    }
    Write-Host "      using Python $pyVer ($($pyCmd -join ' '))"
    $exe = $pyCmd[0]; $rest = @(); if ($pyCmd.Length -gt 1) { $rest = $pyCmd[1..($pyCmd.Length - 1)] }
    & $exe @rest -m venv $Venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Fail "could not create venv at $Venv" }
    Ok "venv created (Python $pyVer)"
}

# ---------------------------------------------------------------- 2. deps
Step 2 "Python packages (kokoro-onnx, onnxruntime, soundfile, numpy, phonemizer, piper-tts)"
$stamp = Join-Path $Venv ".deps_ok"
$reqHash = Get-Sha256 $Req
if ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $reqHash)) {
    Done "packages installed"
} else {
    & $VenvPy -m pip install --disable-pip-version-check -q --upgrade pip
    & $VenvPy -m pip install --disable-pip-version-check -q -r $Req
    if ($LASTEXITCODE -ne 0) { Fail "pip install failed (see messages above). Check your internet connection and re-run." }
    Ok "core packages"
    & $VenvPy -m pip install --disable-pip-version-check -q "piper-tts>=1.3,<2"
    if ($LASTEXITCODE -ne 0) {
        Warn "piper-tts did not install - Piper voices will run through the built-in onnxruntime route instead (fine)."
    } else { Ok "piper-tts" }
    & $VenvPy -c "import kokoro_onnx, onnxruntime, numpy, soundfile, phonemizer; print('      kokoro-onnx', __import__('importlib.metadata').metadata.version('kokoro-onnx'), '| onnxruntime', onnxruntime.__version__, '| numpy', numpy.__version__)"
    if ($LASTEXITCODE -ne 0) { Fail "package import check failed." }
    Set-Content -Path $stamp -Value $reqHash -Encoding ascii
    Ok "packages verified"
}

# ---------------------------------------------------------------- 3. models
Step 3 "Voice models -> backend\data\tts"
New-Item -ItemType Directory -Force -Path $DataDir, $PiperDir | Out-Null

# Kokoro-82M (weights Apache-2.0) from the kokoro-onnx GitHub releases
$kokoroModel = if ($Int8) { "kokoro-v1.0.int8.onnx" } else { "kokoro-v1.0.onnx" }
$kokoroFiles = @($kokoroModel, "voices-v1.0.bin")
$missing = @($kokoroFiles | Where-Object { -not ((Test-Path (Join-Path $DataDir $_)) -and (Test-Path (Join-Path $DataDir "$_.sha256"))) })
if ($missing.Count -eq 0) {
    foreach ($f in $kokoroFiles) { Done $f }
} else {
    $done = $false
    foreach ($tag in @("model-files-v1.1", "model-files-v1.0")) {
        $digests = @{}
        try {
            $rel = Invoke-RestMethod -UseBasicParsing -Headers @{ "User-Agent" = "nhl-dynasty-tts-setup" } `
                -Uri "https://api.github.com/repos/thewh1teagle/kokoro-onnx/releases/tags/$tag"
            foreach ($a in $rel.assets) { if ($a.digest -and $a.digest -like "sha256:*") { $digests[$a.name] = $a.digest.Substring(7) } }
            if (-not ($rel.assets | Where-Object { $_.name -eq $kokoroModel })) { continue }
        } catch {
            if ($tag -ne "model-files-v1.0") { continue }   # API unreachable: try plain download from v1.0
        }
        try {
            foreach ($f in $kokoroFiles) {
                $sha = ""; if ($digests.ContainsKey($f)) { $sha = $digests[$f] }
                Save-File "https://github.com/thewh1teagle/kokoro-onnx/releases/download/$tag/$f" (Join-Path $DataDir $f) $sha
            }
            $done = $true; break
        } catch { Warn "$tag : $($_.Exception.Message)" }
    }
    if (-not $done) { Fail "could not download the Kokoro model files (see above)." }
}

# Piper voices (permissive licences only; pinned to the piper-voices v1.0.0 revision)
$piperVoices = @(
    @{ id = "de_DE-thorsten-high"; path = "de/de_DE/thorsten/high" },    # CC0   (de play-by-play)
    @{ id = "de_DE-kerstin-low";   path = "de/de_DE/kerstin/low" },      # CC0   (de analyst)
    @{ id = "sv_SE-nst-medium";    path = "sv/sv_SE/nst/medium" },       # CC0   (sv both, analyst pitch-shifted)
    @{ id = "fr_FR-gilles-low";    path = "fr/fr_FR/gilles/low" }        # CC0   (fr analyst)
)
if ($ExtraVoices) {
    $piperVoices += @{ id = "fr_FR-siwis-medium";  path = "fr/fr_FR/siwis/medium" }   # CC-BY-4.0 (fr fallback)
    $piperVoices += @{ id = "es_ES-davefx-medium"; path = "es/es_ES/davefx/medium" }  # CC0       (es fallback)
}
$hf = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
foreach ($v in $piperVoices) {
    $onnx = Join-Path $PiperDir "$($v.id).onnx"
    $json = "$onnx.json"
    $url = "$hf/$($v.path)/$($v.id).onnx"
    try {
        $sha = ""
        if (-not ((Test-Path $onnx) -and (Test-Path "$onnx.sha256"))) { $sha = Get-HfSha256 $url }
        Save-File $url $onnx $sha
        if (-not (Test-Path $json)) {
            Save-File "$url.json" $json ""
        }
        $null = Get-Content $json -Raw -Encoding UTF8 | ConvertFrom-Json   # validate
    } catch {
        Warn "Piper voice $($v.id) failed: $($_.Exception.Message)"
        Remove-Item $json -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------- 4. start
Step 4 "Start the worker on $BaseUrl"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$h = Get-Health
if ($h -and $Restart) { Stop-Worker; Start-Sleep -Seconds 1; $h = $null }
if ($NoStart) {
    Done "skipped (-NoStart)"
} elseif ($h) {
    Done "worker already running (use -Restart to restart)"
} elseif ($Foreground) {
    Write-Host "      running in this window - Ctrl+C to stop. Start the API in another window."
    & $VenvPy (Join-Path $WorkerDir "worker.py") --port $Port --data $DataDir
    exit $LASTEXITCODE
} else {
    $out = Join-Path $LogDir "tts_worker.out.log"
    $err = Join-Path $LogDir "tts_worker.log"
    $wargs = @("`"$(Join-Path $WorkerDir 'worker.py')`"", "--port", "$Port", "--data", "`"$DataDir`"")
    $p = Start-Process -FilePath $VenvPy -ArgumentList $wargs -WorkingDirectory $WorkerDir `
        -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    Set-Content -Path $PidFile -Value $p.Id -Encoding ascii
    Write-Host "      PID $($p.Id), log: $err"
    $deadline = (Get-Date).AddSeconds(150)
    do {
        Start-Sleep -Milliseconds 700
        $h = Get-Health
        if ($p.HasExited -and -not $h) {
            Start-Sleep -Seconds 2; $h = Get-Health      # lost a race with the API's own autostart?
            if (-not $h) { Get-Content $err -Tail 30 -ErrorAction SilentlyContinue; Fail "worker exited (log above)." }
        }
    } while ((-not $h -or -not $h.ready) -and (Get-Date) -lt $deadline)
    if (-not $h -or -not $h.ready) { Get-Content $err -Tail 30 -ErrorAction SilentlyContinue; Fail "worker did not become ready in time (log above)." }
    Ok "worker ready"
}

# ---------------------------------------------------------------- 5. status
Step 5 "Status"
$h = Get-Health
if (-not $h) {
    Warn "worker not running."
} else {
    Write-Host ("      engine: {0}   format: {1}   languages: {2}" -f $h.engine, $h.format, ($h.languages -join ", "))
    foreach ($prop in $h.engines.PSObject.Properties) {
        $e = $prop.Value
        if ($e.loaded) { Write-Host ("      {0,-7} loaded {1}{2}" -f $prop.Name, $e.model, $(if ($e.backend) { " ($($e.backend))" } else { "" })) -ForegroundColor Green }
        else { Write-Host ("      {0,-7} NOT loaded: {1}" -f $prop.Name, $e.error) -ForegroundColor Yellow }
    }
    foreach ($lp in $h.voices.PSObject.Properties) {
        $pbp = $lp.Value.pbp; $col = $lp.Value.color
        $cs = "$($col.engine):$($col.voice)"; if ($col.timbre_st) { $cs += " ($($col.timbre_st) st)" }
        Write-Host ("      {0}  play-by-play {1,-28} analyst {2}" -f $lp.Name, "$($pbp.engine):$($pbp.voice)", $cs)
    }
    try {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-Local "/synth" "POST" '{"text":"Scores! The neural voice is ready for puck drop.","lang":"en","speaker":"pbp","profile":"hype"}' 30000
        $sw.Stop()
        Ok ("test clip: {0} ms audio synthesised in {1} ms ({2}, {3} KB)" -f $r.Headers["X-Duration-Ms"], $sw.ElapsedMilliseconds, $r.Type, [math]::Round($r.Bytes.Length / 1KB))
    } catch { Warn "test synthesis failed: $($_.Exception.Message)" }
}
Write-Host ""
Write-Host "Next: start the API as usual (.\start_api.ps1) - it detects the worker automatically." -ForegroundColor White
Write-Host "      (.\start_all.ps1 does both; the API also relaunches the worker if it is not running.)" -ForegroundColor White
exit 0

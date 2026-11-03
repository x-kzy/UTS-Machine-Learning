param([string]$Message = "Lengkapi proyek UTS B10 beserta source, laporan, eksperimen, dan galeri hasil")
$ErrorActionPreference = "Stop"
$repoUrl = "https://github.com/x-kzy/UTS-Machine-Learning.git"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

function Invoke-ProjectGit {
    param([string[]]$GitArguments)
    & git @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git gagal. Periksa pesan Git di atas." }
}

try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git belum terpasang. Pasang Git for Windows dari https://git-scm.com/download/win lalu buka CMD baru."
    }
    $manifestPath = Join-Path $root "repository_manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) { throw "repository_manifest.json tidak ditemukan. Ekstrak seluruh ZIP terlebih dahulu." }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    Write-Host "Memeriksa kelengkapan paket ..."
    foreach ($item in $manifest.files) {
        $relative = $item.path.Replace('/', [IO.Path]::DirectorySeparatorChar)
        $full = [IO.Path]::GetFullPath((Join-Path $root $relative))
        if (-not $full.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Path paket tidak valid." }
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { throw "File paket hilang: $($item.path)" }
        $file = Get-Item -LiteralPath $full
        if ($file.Length -gt 90MB) { throw "File terlalu besar untuk paket Git biasa: $($item.path)" }
    }
    $work = Join-Path $root ".upload-work"
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $destination = Join-Path $work ("upload_" + [Guid]::NewGuid().ToString("N"))
    Write-Host "Mengambil repository terbaru agar riwayat yang ada tetap tersambung ..."
    Invoke-ProjectGit -GitArguments @("clone", "--quiet", "--branch", "main", "--single-branch", $repoUrl, $destination)
    foreach ($item in $manifest.files) {
        $relative = $item.path.Replace('/', [IO.Path]::DirectorySeparatorChar)
        $source = Join-Path $root $relative
        $target = Join-Path $destination $relative
        if (($item.path -eq "LICENSE") -and (Test-Path -LiteralPath $target)) { continue }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Force
    }
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $destination "repository_manifest.json") -Force
    $authorName = & git -C $destination config user.name
    if ([string]::IsNullOrWhiteSpace(($authorName -join ""))) {
        Invoke-ProjectGit -GitArguments @("-C", $destination, "config", "user.name", "Fikri Anwar")
    }
    $authorEmail = & git -C $destination config user.email
    if ([string]::IsNullOrWhiteSpace(($authorEmail -join ""))) {
        Invoke-ProjectGit -GitArguments @("-C", $destination, "config", "user.email", "207746374+x-kzy@users.noreply.github.com")
    }
    Invoke-ProjectGit -GitArguments @("-C", $destination, "add", "--", ".")
    $changed = & git -C $destination diff --cached --name-only
    if ([string]::IsNullOrWhiteSpace(($changed -join ""))) {
        Write-Host "Isi repository sudah sama dengan paket. Tidak ada commit tambahan."
        exit 0
    }
    Invoke-ProjectGit -GitArguments @("-C", $destination, "diff", "--cached", "--stat")
    Invoke-ProjectGit -GitArguments @("-C", $destination, "commit", "-m", $Message)
    Write-Host "Mengunggah ke akun GitHub Anda. Selesaikan login GitHub jika diminta ..."
    Invoke-ProjectGit -GitArguments @("-C", $destination, "push", "origin", "main")
    Write-Host "SELESAI: https://github.com/x-kzy/UTS-Machine-Learning"
    Write-Host "Salinan Git lokal: $destination"
} catch {
    Write-Host ("GAGAL: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "Jika akses ditolak, gunakan akun GitHub x-kzy atau akun yang mempunyai akses tulis."
    Write-Host "Jika remote berubah saat proses berlangsung, jalankan kembali paket untuk mengambil HEAD terbaru."
    exit 1
}

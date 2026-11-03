# Install the latest Langtrain CLI on Windows:
#   irm https://raw.githubusercontent.com/langtrain/langtrain-studio/main/scripts/install.ps1 | iex
$ErrorActionPreference = "Stop"
$Repo = if ($env:LANGTRAIN_REPO) { $env:LANGTRAIN_REPO } else { "langtrain/langtrain-studio" }
$Dest = if ($env:LANGTRAIN_INSTALL_DIR) { $env:LANGTRAIN_INSTALL_DIR } else { Join-Path $env:USERPROFILE ".langtrain" }

$arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
$target = "$arch-pc-windows-msvc"
$tag = (Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest").tag_name
$name = "langtrain-$tag-$target"
$url = "https://github.com/$Repo/releases/download/$tag/$name.zip"

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  Write-Host "Downloading $name"
  Invoke-WebRequest $url -OutFile "$tmp\$name.zip"
  Invoke-WebRequest "$url.sha256" -OutFile "$tmp\$name.zip.sha256"
  $want = (Get-Content "$tmp\$name.zip.sha256").Split(" ")[0].ToLower()
  $got = (Get-FileHash "$tmp\$name.zip" -Algorithm SHA256).Hash.ToLower()
  if ($want -ne $got) { throw "Checksum mismatch" }
  Expand-Archive "$tmp\$name.zip" -DestinationPath $tmp
  $bin = Join-Path $Dest "bin"
  New-Item -ItemType Directory -Force -Path $bin | Out-Null
  Copy-Item "$tmp\$name\langtrain.exe" $bin -Force
  if (Test-Path "$bin\langtune") { Remove-Item "$bin\langtune" -Recurse -Force }
  Copy-Item "$tmp\$name\langtune" "$bin\langtune" -Recurse
  $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
  if ($userPath -notlike "*$bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$bin;$userPath", "User")
    Write-Host "Added $bin to your PATH (open a new terminal)."
  }
  Write-Host "Installed langtrain $tag. Run: langtrain serve"
} finally {
  Remove-Item $tmp -Recurse -Force
}

<#
  LingShu - build the HiAI .om model inference probe (libhiai_om.so).

  This one DOES link the HMS stub libhiai_foundation.so, because it calls
  HMS_HiAIOptions_* / HMS_HiAICompatibility_* directly (the codelab sequence),
  not through dlsym.

  ASCII-only (PS 5.1 + BOM-less .ps1 + Chinese = parser errors).
#>
[CmdletBinding()]
param(
  [string]$Sdk = '<sdk>\DevEco Studio\sdk\default',
  [string]$HarmonyConfigs = '<project>\harmony-configs',
  [switch]$Install
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = Join-Path $here 'out'
$src  = Join-Path $here 'hiai_om_napi.cpp'

function Need($p, $what) { if (-not (Test-Path -LiteralPath $p)) { throw ("MISSING {0}: {1}" -f $what, $p) } }

Write-Host "=== preflight ===" -ForegroundColor Cyan
Need $src 'probe source'
$ndk     = Join-Path $Sdk 'openharmony\native'
$clang   = Join-Path $ndk 'llvm\bin\clang++.exe'
$sysroot = Join-Path $ndk 'sysroot'
$inc     = Join-Path $sysroot 'usr\include'
$libdir  = Join-Path $sysroot 'usr\lib\aarch64-linux-ohos'
$hmsInc  = Join-Path $Sdk 'hms\native\sysroot\usr\include'
$hmsLib  = Join-Path $Sdk 'hms\native\sysroot\usr\lib\aarch64-linux-ohos'
Need $clang 'NDK clang++'
Need (Join-Path $hmsInc 'CANNKit\hiai_options.h') 'CANNKit hiai_options.h'
Need (Join-Path $hmsLib 'libhiai_foundation.so') 'HMS stub libhiai_foundation.so'

Write-Host "`n=== compile ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $out | Out-Null
$so = Join-Path $out 'libhiai_om.so'
if (Test-Path -LiteralPath $so) { Remove-Item -LiteralPath $so -Force }

& $clang `
  --target=aarch64-linux-ohos `
  "--sysroot=$sysroot" `
  -O2 -fPIC -shared -std=c++17 `
  "-I$inc" "-I$hmsInc" `
  $src -o $so `
  "-L$libdir" "-L$hmsLib" `
  -lneural_network_runtime -lneural_network_core -lhiai_foundation '-lace_napi.z' -lc++ -lc++abi
if ($LASTEXITCODE -ne 0) { throw ("compile failed (exit {0})" -f $LASTEXITCODE) }
Write-Host ("  OK  {0}  ({1} KB)" -f $so, [math]::Round((Get-Item -LiteralPath $so).Length / 1KB, 1)) -ForegroundColor Green

# sanity: did the HiAI symbols actually get referenced?
$txt = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($so))
Write-Host ("  refs HMS_HiAIOptions_SetModelDeviceOrder : {0}" -f $txt.Contains('HMS_HiAIOptions_SetModelDeviceOrder'))
Write-Host ("  refs HMS_HiAICompatibility_CheckFromBuffer: {0}" -f $txt.Contains('HMS_HiAICompatibility_CheckFromBuffer'))
Write-Host ("  refs runOmModel                          : {0}" -f $txt.Contains('runOmModel'))

if ($Install) {
  Write-Host "`n=== install ===" -ForegroundColor Cyan
  $libsDir  = Join-Path $HarmonyConfigs 'entry\libs\arm64-v8a'
  $typesDir = Join-Path $HarmonyConfigs 'entry\src\main\cpp\types\libhiai_om.so'
  New-Item -ItemType Directory -Force -Path $libsDir, $typesDir | Out-Null
  Copy-Item -LiteralPath $so -Destination (Join-Path $libsDir 'libhiai_om.so') -Force
  $dts = @'
/* LingShu HiAI .om model inference probe (hiai_om_napi.cpp). */
export interface HiaiOmNative {
  /** Run the official HiAI codelab sequence over a .om model buffer, NPU then CPU. */
  runOmModel(model: ArrayBuffer): string;
}
export const hiaiOm: HiaiOmNative;
export default hiaiOm;
'@
  $dts | Out-File (Join-Path $typesDir 'index.d.ts') -Encoding UTF8
  $ohpkg = @'
{
  "name": "libhiai_om.so",
  "types": "./index.d.ts",
  "version": "1.0.0",
  "description": "LingShu HiAI Foundation .om model inference probe"
}
'@
  $ohpkg | Out-File (Join-Path $typesDir 'oh-package.json5') -Encoding UTF8
  $ohpkg | Out-File (Join-Path $typesDir 'package.json') -Encoding UTF8
  Write-Host "  installed." -ForegroundColor Green
}

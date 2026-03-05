param(
    [string]$CondaEnv = "fert-opt",
    [string]$Config = "configs/default.yaml",
    [string]$OutRoot = "artifacts/repro",
    [string]$Engines = "deap_nsga2",
    [string]$Seeds = "42,43",
    [string]$PopSizes = "40",
    [string]$Generations = "12",
    [string]$PrototypeFlags = "true,false",
    [string]$CoupledFlags = "true",
    [string]$DynamicEliteFlags = "true,false",
    [string]$SurrogateFlags = "false",
    [int]$HvSamples = 8000,
    [int]$HvSeed = 42,
    [int]$Dpi = 300
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cmd = @(
    "run", "-n", $CondaEnv,
    "python", "scripts/reproduce_all.py",
    "--config", $Config,
    "--out-root", $OutRoot,
    "--engines", $Engines,
    "--seeds", $Seeds,
    "--pop-sizes", $PopSizes,
    "--generations", $Generations,
    "--prototype-flags", $PrototypeFlags,
    "--coupled-flags", $CoupledFlags,
    "--dynamic-elite-flags", $DynamicEliteFlags,
    "--surrogate-flags", $SurrogateFlags,
    "--hv-samples", $HvSamples,
    "--hv-seed", $HvSeed,
    "--dpi", $Dpi
)

Write-Host "Running: conda $($cmd -join ' ')"
conda @cmd

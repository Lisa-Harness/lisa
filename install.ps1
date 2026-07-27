<#
.SYNOPSIS
    Install Lisa Framework and optionally configure one or more model providers.

.DESCRIPTION
    Installs or updates the Lisa Framework by cloning the repository,
    installing uv and Python dependencies, verifying the installation,
    then helping you pick which model provider(s) to enable at runtime.

    Supply -Provider to pre-select providers from the command line or
    environment (useful for scripted/CI or webpage-generated pastes).
    Omitting -Provider defaults to Ollama (local, no API key).

.PARAMETER RepoUrl
    Git repository URL. Defaults to LISA_REPO_URL environment variable
    or the official Lisa Framework repository.

.PARAMETER InstallDir
    Installation directory. Defaults to LISA_INSTALL_DIR environment variable
    or $HOME/lisa-framework.

.PARAMETER Branch
    Git branch to clone. Defaults to LISA_BRANCH environment variable or 'main'.

.PARAMETER Provider
    One or more model providers to enable at install time.
    Accepted values: ollama, openai, azure_openai, anthropic, google.
    Default: ollama (local, free, no API key).
    When multiple are given, .env is pre-populated with keys for all of them
    (commented out where a value must be filled in).

.PARAMETER SkipDiagnostics
    Skip running lisa verify after installation.

.EXAMPLE
    # Default install — Ollama only
    .\install.ps1

.EXAMPLE
    # Install with Azure OpenAI + Anthropic
    .\install.ps1 -Provider azure_openai, anthropic

.EXAMPLE
    # Custom directory and branch
    .\install.ps1 -InstallDir C:\Projects\lisa-framework -Branch develop

.EXAMPLE
    # Webpage-generated paste that pre-selects OpenAI
    .\install.ps1 -Provider openai

.LINK
    https://github.com/ORG/lisa-framework
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$RepoUrl = $(
        if ($env:LISA_REPO_URL) { $env:LISA_REPO_URL }
        else { "https://github.com/ORG/lisa-framework.git" }
    ),

    [Parameter()]
    [string]$InstallDir = $(
        if ($env:LISA_INSTALL_DIR) { $env:LISA_INSTALL_DIR }
        else { Join-Path $HOME "lisa-framework" }
    ),

    [Parameter()]
    [string]$Branch = $(
        if ($env:LISA_BRANCH) { $env:LISA_BRANCH }
        else { "main" }
    ),

    [Parameter()]
    [ValidateSet("ollama", "openai", "azure_openai", "anthropic", "google")]
    [string[]]$Provider = @(
        if ($env:LISA_PROVIDER) { $env:LISA_PROVIDER -split "," }
        else { "ollama" }
    ),

    [Parameter()]
    [switch]$SkipDiagnostics
)

$ErrorActionPreference = "Stop"

# -- Provider metadata -------------------------------------------------------

# NOTE: env var names use the LISA_* prefix (renamed from LITTLE_LISA_*; no
# compat aliases). The hosts read LISA_* with fallbacks to bare
# OPENAI_API_KEY / ANTHROPIC_API_KEY / AZURE_OPENAI_* / GOOGLE_API_KEY /
# OLLAMA_HOST. See hosts/core/providers.py and .env.example at repo root.
$ProviderMeta = @{
    ollama       = @{ Label = "Ollama";       EnvVars = @("LISA_OLLAMA_HOST");         Defaults = @("127.0.0.1:11434") }
    openai       = @{ Label = "OpenAI";       EnvVars = @("LISA_OPENAI_API_KEY");       Defaults = @("") }
    azure_openai = @{ Label = "Azure OpenAI"; EnvVars = @("LISA_AZURE_OPENAI_ENDPOINT", "LISA_AZURE_OPENAI_API_KEY", "LISA_AZURE_OPENAI_API_VERSION"); Defaults = @("", "", "2024-10-21") }
    anthropic    = @{ Label = "Anthropic";    EnvVars = @("LISA_ANTHROPIC_API_KEY", "LISA_ANTHROPIC_BASE_URL", "LISA_ANTHROPIC_VERSION"); Defaults = @("", "https://api.anthropic.com", "2023-06-01") }
    google       = @{ Label = "Google Gemini"; EnvVars = @("LISA_GOOGLE_API_KEY", "LISA_GOOGLE_API_BASE_URL"); Defaults = @("", "https://generativelanguage.googleapis.com/v1beta/openai/") }
}

function Write-Step {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$Step,

        [Parameter(Mandatory)]
        [int]$Total,

        [Parameter(Mandatory)]
        [string]$Message
    )

    Write-Host "  [$Step/$Total] $Message" -ForegroundColor Cyan
}

function Test-PythonVersion {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PythonCommand,

        [Parameter()]
        [int]$MinMajor = 3,

        [Parameter()]
        [int]$MinMinor = 11
    )

    try {
        $versionString = & $PythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $major, $minor = $versionString.Split('.')
        $majorInt = [int]$major
        $minorInt = [int]$minor
        return ($majorInt -gt $MinMajor) -or ($majorInt -eq $MinMajor -and $minorInt -ge $MinMinor)
    }
    catch { return $false }
}

function Install-UvIfMissing {
    [CmdletBinding()]
    param()

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $uvVersion = (uv --version 2>$null) -replace '^uv\s+', ''
        Write-Verbose "uv already installed: $uvVersion"
        return
    }

    Write-Verbose "Installing uv package manager..."
    $installScript = Invoke-RestMethod -Uri https://astral.sh/uv/install.ps1
    Invoke-Expression $installScript

    $uvBinPaths = @(
        (Join-Path $env:USERPROFILE ".local\bin"),
        (Join-Path $env:USERPROFILE ".cargo\bin")
    )
    foreach ($binPath in $uvBinPaths) {
        if ((Test-Path $binPath) -and ($env:PATH -notlike "*$binPath*")) {
            $env:PATH = "$binPath;$env:PATH"
        }
    }
}

function New-GenerateEnvFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Dir,

        [Parameter(Mandatory)]
        [string[]]$Providers
    )

    # .env and .env.example live at the repo root (not hosts/little_lisa/,
    # which no longer exists). hosts/cli, hosts/web, hosts/vscode load .env
    # from the working directory / data dir.
    $envFile = Join-Path $Dir ".env"
    $envExample = Join-Path $Dir ".env.example"

    if (Test-Path $envFile) {
        Write-Host "  .env already exists — preserved" -ForegroundColor Yellow
        return
    }

    # Start from .env.example content
    $lines = if (Test-Path $envExample) {
        Get-Content $envExample
    } else {
        @("# Lisa Framework environment configuration", "# Generated by: install.ps1", "")
    }

    # Append a section per selected provider
    $lines += ""
    $lines += "# --- Providers selected at install time ---"
    $lines += ""

    foreach ($provider in $Providers) {
        $meta = $ProviderMeta[$provider]
        $lines += "# $($meta.Label)"
        for ($i = 0; $i -lt $meta.EnvVars.Count; $i++) {
            $key = $meta.EnvVars[$i]
            $default = $meta.Defaults[$i]
            if ($default) {
                $lines += "$key=$default"
            } else {
                $lines += "# $key=   <-- fill in your value"
            }
        }
        $lines += ""
    }

    $lines += "# --- Web search (optional) ---"
    $lines += "# WEB_SEARCH_API_KEY=   <-- fill in to enable /search"
    $lines += ""

    Set-Content -Path $envFile -Value $lines -Encoding UTF8
    Write-Host "  Created .env from template — $($Providers.Count) provider(s) configured" -ForegroundColor Green
}

function Write-PostInstallGuidance {
    [CmdletBinding()]
    param(
        [Parameter()]
        [string]$InstallDir,

        [Parameter()]
        [string[]]$Providers
    )

    $hasSecrets = $Providers | Where-Object { $_ -ne "ollama" }

    Write-Host ""
    Write-Host "  ------------------------" -ForegroundColor Green
    Write-Host "  Lisa Framework installed!" -ForegroundColor Green
    Write-Host ""

    if ($hasSecrets) {
        Write-Host "  To finish configuring your providers:" -ForegroundColor Yellow
        Write-Host "    1. Edit $(Join-Path $InstallDir ".env")" -ForegroundColor Yellow
        foreach ($provider in $Providers) {
            $meta = $ProviderMeta[$provider]
            $blanks = @($meta.EnvVars | ForEach-Object { $meta.Defaults[$meta.EnvVars.IndexOf($_)] }) -eq ""
            if ($blanks.Count -gt 0) {
                $fillThese = ($meta.EnvVars | Where-Object { $meta.Defaults[$meta.EnvVars.IndexOf($_)] -eq "" }) -join ", "
                Write-Host "       $($meta.Label): uncomment and set $fillThese" -ForegroundColor Yellow
            }
        }
        Write-Host "    2. Run: cd $InstallDir; uv run lisa verify" -ForegroundColor Yellow
        Write-Host "       (to verify everything is working correctly)"
        Write-Host ""
    }

    Write-Host "  Quick start:"
    Write-Host "    cd $InstallDir"
    Write-Host "    uv run lisa once `"/markdown Hello, Lisa!`""
    Write-Host "    uv run lisa              # interactive REPL"
    Write-Host ""

    if ($Providers -contains "ollama") {
        Write-Host "  Ollama selected — make sure it's installed and models pulled:"
        Write-Host "    winget install Ollama.Ollama"
        Write-Host "    ollama pull llama3.2:latest"
        Write-Host "    ollama pull qwen2.5:7b"
        Write-Host ""
    }

    Write-Host "  Verify your setup:"
    Write-Host "    uv run lisa verify"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Main installation
# ---------------------------------------------------------------------------

function Install-LisaFramework {
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "  Lisa Framework Installer" -ForegroundColor Green
    Write-Host "  ------------------------"
    Write-Host ""

    Write-Step -Step 1 -Total 6 -Message "Checking Python..."
    $pythonCmd = $null
    foreach ($candidate in @('python', 'python3')) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            if (Test-PythonVersion -PythonCommand $candidate) {
                $pythonCmd = $candidate
                break
            }
        }
    }

    if (-not $pythonCmd) {
        Write-Error @"
Python >= 3.11 is required but not found.
Install Python from https://www.python.org/downloads/
"@
    }

    $pyVersion = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Write-Host "  OK: Python $pyVersion" -ForegroundColor Green

    Write-Step -Step 2 -Total 6 -Message "Checking uv..."
    Install-UvIfMissing
    $uvVersion = (uv --version 2>$null) -replace '^uv\s+', ''
    Write-Host "  OK: uv $uvVersion" -ForegroundColor Green

    Write-Step -Step 3 -Total 6 -Message "Setting up repository..."
    $gitDir = Join-Path $InstallDir ".git"

    if (Test-Path $gitDir) {
        Write-Verbose "Updating existing repository at $InstallDir"
        Push-Location $InstallDir
        try {
            git fetch origin $Branch *>&1 | Out-Null
            git checkout $Branch *>&1 | Out-Null
            git pull origin $Branch *>&1 | Out-Null
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Verbose "Cloning $RepoUrl to $InstallDir"
        git clone --depth 1 --branch $Branch $RepoUrl $InstallDir *>&1 | Out-Null
    }
    Write-Host "  OK: Repository ready at $InstallDir" -ForegroundColor Green

    Write-Step -Step 4 -Total 6 -Message "Installing dependencies..."
    Push-Location $InstallDir
    try { uv sync --extra dev }
    finally { Pop-Location }
    Write-Host "  OK: Dependencies installed" -ForegroundColor Green

    Write-Step -Step 5 -Total 6 -Message "Generating .env for selected provider(s)..."
    New-GenerateEnvFile -Dir $InstallDir -Providers $Provider

    if (-not $SkipDiagnostics) {
        Write-Step -Step 6 -Total 6 -Message "Running diagnostics..."
        Write-Host ""
        Push-Location $InstallDir
        try { uv run lisa verify *>&1 | Out-Host }
        catch { Write-Warning "Diagnostics encountered an issue (non-fatal): $_" }
        finally { Pop-Location }
    }

    Write-PostInstallGuidance -InstallDir $InstallDir -Providers $Provider
}

Install-LisaFramework

#!/usr/bin/env bash
# Lisa Framework installer for Linux/macOS
#
# Usage examples:
#   curl -fsSL https://raw.githubusercontent.com/ORG/lisa-framework/main/install.sh | bash
#   curl -fsSL ... | bash -s -- --provider ollama,openai
#   curl -fsSL ... | bash -s -- --provider anthropic --install-dir ~/dev/lisa

set -euo pipefail

# --- Parameter parsing -------------------------------------------------------

REPO_URL="${LISA_REPO_URL:-https://github.com/ORG/lisa-framework.git}"
INSTALL_DIR="${LISA_INSTALL_DIR:-$HOME/lisa-framework}"
REPO_BRANCH="${LISA_BRANCH:-main}"

# Default provider: ollama.  Accepts comma-separated list.
IFS=',' read -ra PROVIDERS <<< "${LISA_PROVIDER:-ollama}"

SKIP_DIAGNOSTICS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url)       REPO_URL="$2";       shift 2 ;;
        --install-dir)    INSTALL_DIR="$2";     shift 2 ;;
        --branch)         REPO_BRANCH="$2";     shift 2 ;;
        --provider)       IFS=',' read -ra PROVIDERS <<< "$2"; shift 2 ;;
        --skip-diagnostics) SKIP_DIAGNOSTICS=true; shift ;;
        --help|-h)
            cat <<HELP
Usage: install.sh [OPTIONS]

Install the Lisa Framework and optionally configure model providers.

Options:
  --repo-url URL          Git repository URL (default: \$LISA_REPO_URL or org repo)
  --install-dir PATH      Installation directory (default: \$LISA_INSTALL_DIR or ~/lisa-framework)
  --branch NAME           Git branch to clone (default: \$LISA_BRANCH or main)
  --provider LIST         Comma-separated provider(s): ollama,openai,azure_openai,anthropic,google
                          (default: \$LISA_PROVIDER or ollama)
  --skip-diagnostics      Skip running little-lisa verify after install
  --help, -h              Show this help

Examples:
  install.sh                                  # Ollama only (default)
  install.sh --provider ollama,openai          # Ollama + OpenAI
  install.sh --provider anthropic,google --install-dir ~/dev/lisa
HELP
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Constants ---------------------------------------------------------------

readonly MIN_PYTHON_MAJOR=3
readonly MIN_PYTHON_MINOR=11

# Provider metadata: label, env var names, default values
declare -A _PROVIDER_LABEL
_PROVIDER_LABEL[ollama]="Ollama"
_PROVIDER_LABEL[openai]="OpenAI"
_PROVIDER_LABEL[azure_openai]="Azure OpenAI"
_PROVIDER_LABEL[anthropic]="Anthropic"
_PROVIDER_LABEL[google]="Google Gemini"

declare -A _PROVIDER_KEYS
_PROVIDER_KEYS[ollama]="LITTLE_LISA_OLLAMA_HOST"
_PROVIDER_KEYS[openai]="LITTLE_LISA_OPENAI_API_KEY"
_PROVIDER_KEYS[azure_openai]="LITTLE_LISA_AZURE_OPENAI_ENDPOINT,LITTLE_LISA_AZURE_OPENAI_API_KEY,LITTLE_LISA_AZURE_OPENAI_API_VERSION"
_PROVIDER_KEYS[anthropic]="LITTLE_LISA_ANTHROPIC_API_KEY,LITTLE_LISA_ANTHROPIC_BASE_URL,LITTLE_LISA_ANTHROPIC_VERSION"
_PROVIDER_KEYS[google]="LITTLE_LISA_GOOGLE_API_KEY,LITTLE_LISA_GOOGLE_API_BASE_URL"

declare -A _PROVIDER_DEFAULTS
_PROVIDER_DEFAULTS[ollama]="127.0.0.1:11434"
_PROVIDER_DEFAULTS[openai]=""
_PROVIDER_DEFAULTS[azure_openai]=",,2024-10-21"
_PROVIDER_DEFAULTS[anthropic]=",https://api.anthropic.com,2023-06-01"
_PROVIDER_DEFAULTS[google]=",https://generativelanguage.googleapis.com/v1beta/openai/"

# --- Utility functions -------------------------------------------------------

step() {
    local s=$1 total=$2 msg=$3
    printf "  [%d/%d] %s\n" "$s" "$total" "$msg"
}

error_exit() {
    printf "  ERROR: %s\n" "$1" >&2
    exit 1
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

find_python() {
    local candidate version major minor
    for candidate in python3 python; do
        if have_cmd "$candidate"; then
            version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            IFS='.' read -r major minor <<< "$version"
            if [[ "$major" -gt $MIN_PYTHON_MAJOR ]] || [[ "$major" -eq $MIN_PYTHON_MAJOR && "$minor" -ge $MIN_PYTHON_MINOR ]]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

install_uv() {
    if have_cmd uv; then
        return 0
    fi
    printf "  Installing uv...\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    [[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

clone_or_update_repo() {
    local install_dir=$1 repo_url=$2 branch=$3
    if [[ -d "$install_dir/.git" ]]; then
        printf "  Updating existing repository at %s...\n" "$install_dir"
        git -C "$install_dir" fetch origin "$branch" >/dev/null 2>&1
        git -C "$install_dir" checkout "$branch" >/dev/null 2>&1
        git -C "$install_dir" pull origin "$branch" >/dev/null 2>&1
    else
        printf "  Cloning %s to %s...\n" "$repo_url" "$install_dir"
        git clone --depth 1 --branch "$branch" "$repo_url" "$install_dir" >/dev/null 2>&1
    fi
}

generate_env_file() {
    local install_dir=$1; shift
    local providers=("$@")
    local env_file="$install_dir/hosts/little_lisa/.env"
    local env_example="$install_dir/hosts/little_lisa/.env.example"

    if [[ -f "$env_file" ]]; then
        printf "  .env already exists — preserved\n"
        return
    fi

    # ShellCheck: we intentionally want word splitting below for provider keys
    # shellcheck disable=SC2068
    {
        printf "# Little Lisa environment configuration\n"
        printf "# Generated by: install.sh\n"
        printf "\n"

        for prov in ${providers[@]}; do
            local label="${_PROVIDER_LABEL[$prov]}"
            printf "# %s\n" "$label"

            local keys="${_PROVIDER_KEYS[$prov]}"
            local defaults="${_PROVIDER_DEFAULTS[$prov]}"

            old_ifs="$IFS"
            IFS=','
            local key_array default_array
            read -ra key_array <<< "$keys"
            read -ra default_array <<< "$defaults"
            IFS="$old_ifs"

            for idx in "${!key_array[@]}"; do
                local k="${key_array[$idx]}"
                local d="${default_array[$idx]:-}"
                if [[ -n "$d" ]]; then
                    printf "%s=%s\n" "$k" "$d"
                else
                    printf "# %s=   <-- fill in your value\n" "$k"
                fi
            done
            printf "\n"
        done

        printf "# --- Web search (optional) ---\n"
        printf "# WEB_SEARCH_API_KEY=   <-- fill in to enable /search\n"
        printf "\n"
    } > "$env_file"

    printf "  Created .env from template — %d provider(s) configured\n" "${#providers[@]}"
}

post_install_guidance() {
    local install_dir=$1; shift
    local providers=("$@")

    printf "\n"
    printf "  ------------------------\n"
    printf "  Lisa Framework installed!\n"
    printf "\n"

    local has_secrets=false
    local prov
    for prov in ${providers[@]}; do
        [[ "$prov" != "ollama" ]] && { has_secrets=true; break; }
    done

    if $has_secrets; then
        printf "  To finish configuring your providers:\n"
        printf "    1. Edit %s\n" "$install_dir/hosts/little_lisa/.env"
        for prov in ${providers[@]}; do
            [[ "$prov" == "ollama" ]] && continue
            local label="${_PROVIDER_LABEL[$prov]}"
            local keys="${_PROVIDER_KEYS[$prov]}"
            local defaults="${_PROVIDER_DEFAULTS[$prov]}"

            old_ifs="$IFS"
            IFS=','
            local key_array default_array
            read -ra key_array <<< "$keys"
            read -ra default_array <<< "$defaults"
            IFS="$old_ifs"

            local blanks=()
            for idx in "${!key_array[@]}"; do
                [[ -z "${default_array[$idx]:-}" ]] && blanks+=("${key_array[$idx]}")
            done
            if [[ ${#blanks[@]} -gt 0 ]]; then
                local joined
                printf -v joined ', ' "${blanks[@]}"
                printf "       %s: uncomment and set %s\n" "$label" "${joined#, }"
            fi
        done
        printf "    2. Run: cd %s; uv run little-lisa setup\n" "$install_dir"
        printf "       (or use the wizard to walk through each provider interactively)\n"
        printf "\n"
    fi

    printf "  Quick start:\n"
    printf "    cd %s\n" "$install_dir"
    printf "    uv run little-lisa once \"/markdown Hello, Lisa!\"\n"
    printf "    uv run little-lisa              # interactive REPL\n"
    printf "\n"

    local is_ollama=false
    for prov in ${providers[@]}; do
        [[ "$prov" == "ollama" ]] && { is_ollama=true; break; }
    done
    if $is_ollama; then
        printf "  Ollama selected — make sure it's installed and models pulled:\n"
        printf "    curl -fsSL https://ollama.com/install.sh | sh\n"
        printf "    ollama pull llama3.2:latest\n"
        printf "    ollama pull qwen2.5:7b\n"
        printf "\n"
    fi

    printf "  Verify your setup:\n"
    printf "    uv run little-lisa verify\n"
    printf "\n"
}

# --- Main --------------------------------------------------------------------

main() {
    printf "\n"
    printf "  Lisa Framework Installer\n"
    printf "  ------------------------\n"
    printf "\n"

    step 1 6 "Checking Python..."
    local python_cmd
    if ! python_cmd=$(find_python); then
        error_exit "Python >= 3.11 is required.\nInstall from https://www.python.org/downloads/"
    fi
    local py_version
    py_version=$("$python_cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    printf "  OK: Python %s\n" "$py_version"

    step 2 6 "Checking uv..."
    install_uv
    local uv_version
    uv_version=$(uv --version 2>/dev/null | cut -d' ' -f2 || echo "installed")
    printf "  OK: uv %s\n" "$uv_version"

    step 3 6 "Setting up repository..."
    clone_or_update_repo "$INSTALL_DIR" "$REPO_URL" "$REPO_BRANCH"
    printf "  OK: Repository ready at %s\n" "$INSTALL_DIR"

    step 4 6 "Installing dependencies..."
    (cd "$INSTALL_DIR" && uv sync --extra dev)
    printf "  OK: Dependencies installed\n"

    step 5 6 "Generating .env for selected provider(s)..."
    generate_env_file "$INSTALL_DIR" "${PROVIDERS[@]}"

    if ! $SKIP_DIAGNOSTICS; then
        step 6 6 "Running diagnostics..."
        printf "\n"
        (cd "$INSTALL_DIR" && uv run little-lisa verify) || true
        printf "\n"
    fi

    post_install_guidance "$INSTALL_DIR" "${PROVIDERS[@]}"
}

main "$@"

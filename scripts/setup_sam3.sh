#!/usr/bin/env bash
# =============================================================================
# setup_sam3.sh - Install SAM3 library and copy BPE vocab
#
# Works on: Linux, macOS, Windows (Git Bash, WSL, MSYS2)
#
# Usage:
#   bash scripts/setup_sam3.sh
#
# Options (env vars):
#   PIP_CMD          Override pip command    (e.g. PIP_CMD="pip3")
#   SAM3_CLONE_URL   Override git clone URL  (e.g. for mirrors)
#   SAM3_SRC         Override clone target   (default: ./sam3_src)
#   MODELS_DIR       Override models dir     (default: ./models)
#
# Model weights must be downloaded separately (see docs/SETUP_SAM3.md)
# =============================================================================

set -e

# =============================================================================
# 1. PLATFORM DETECTION
# =============================================================================
detect_platform() {
  case "$(uname -s)" in
    Linux*)
      if grep -qi microsoft /proc/version 2>/dev/null; then
        PLATFORM="wsl"
      else
        PLATFORM="linux"
      fi
      ;;
    Darwin*)  PLATFORM="macos"  ;;
    MINGW*|MSYS*|CYGWIN*)  PLATFORM="gitbash"  ;;
    *)  PLATFORM="unknown"  ;;
  esac
}

detect_platform
echo "[setup] Platform detected: $PLATFORM"

# =============================================================================
# 2. PATH HELPERS
# =============================================================================
# Convert a bash path to a Windows-native path (only needed on WSL/Git Bash
# when calling Windows .exe tools like uv.exe or pip.exe).

to_native_path() {
  local p="$1"
  case "$PLATFORM" in
    wsl)      wslpath -w "$p" ;;
    gitbash)  cygpath -w "$p" 2>/dev/null || echo "$p" ;;
    *)        echo "$p" ;;
  esac
}

# True if PIP_CMD points to a Windows .exe (needs native path conversion)
pip_is_windows_exe() {
  [[ "$PIP_CMD" == *".exe"* ]]
}

# Run $PIP_CMD install, converting paths for Windows .exe tools if needed.
# Handles pip extras like "path/to/pkg[dev,notebooks]" by splitting path from extras
# before conversion, then rejoining.
pip_install() {
  local args=()
  for arg in "$@"; do
    if pip_is_windows_exe; then
      # Split path from pip extras: "/path/pkg[dev,extra]" -> "/path/pkg" + "[dev,extra]"
      local path_part="${arg%%\[*}"
      local extras_part=""
      if [[ "$arg" == *"["* ]]; then
        extras_part="[${arg#*\[}"
      fi
      if [[ -e "$path_part" || "$path_part" == /* ]]; then
        args+=("$(to_native_path "$path_part")${extras_part}")
      else
        args+=("$arg")
      fi
    else
      args+=("$arg")
    fi
  done
  echo "      Running: $PIP_CMD install ${args[*]}"
  $PIP_CMD install "${args[@]}"
}

# =============================================================================
# 3. PIP DETECTION
# =============================================================================
# Priority: PIP_CMD override > uv > .venv pip > python -m pip > system pip

detect_pip() {
  # User override
  if [ -n "$PIP_CMD" ]; then
    echo "[setup] Using PIP_CMD override: $PIP_CMD"
    return 0
  fi

  # uv / uv.exe
  for cmd in uv uv.exe; do
    if command -v "$cmd" &>/dev/null; then
      PIP_CMD="$cmd pip"
      echo "[setup] Found $cmd"
      return 0
    fi
  done

  # .venv pip (Windows layout then Unix layout)
  for venv_pip in \
    "$PROJECT_ROOT/.venv/Scripts/pip.exe" \
    "$PROJECT_ROOT/.venv/Scripts/pip" \
    "$PROJECT_ROOT/.venv/bin/pip" \
  ; do
    if [ -f "$venv_pip" ]; then
      PIP_CMD="$venv_pip"
      echo "[setup] Found venv pip: $venv_pip"
      return 0
    fi
  done

  # python -m pip (.venv python first, then system)
  for py in \
    "$PROJECT_ROOT/.venv/Scripts/python.exe" \
    "$PROJECT_ROOT/.venv/bin/python" \
    python3 python python3.exe python.exe \
  ; do
    if command -v "$py" &>/dev/null || [ -f "$py" ]; then
      if "$py" -m pip --version &>/dev/null 2>&1; then
        PIP_CMD="$py -m pip"
        echo "[setup] Found $py -m pip"
        return 0
      fi
    fi
  done

  # Bare pip3 / pip
  for cmd in pip3 pip3.exe pip pip.exe; do
    if command -v "$cmd" &>/dev/null; then
      PIP_CMD="$cmd"
      echo "[setup] Found $cmd"
      return 0
    fi
  done

  return 1
}

# =============================================================================
# 4. PRE-FLIGHT CHECKS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SAM3_SRC="${SAM3_SRC:-$PROJECT_ROOT/sam3_src}"
MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"
SAM3_CLONE_URL="${SAM3_CLONE_URL:-https://github.com/facebookresearch/sam3.git}"

# Check git
if ! command -v git &>/dev/null; then
  echo "[error] git is not installed. Please install git first."
  exit 1
fi

# Check pip
if ! detect_pip; then
  echo ""
  echo "[error] No Python package installer found."
  echo ""
  case "$PLATFORM" in
    linux|macos)
      echo "  Option 1: Install uv (recommended):"
      echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
      echo ""
      echo "  Option 2: Create and activate a venv:"
      echo "    python3 -m venv .venv && source .venv/bin/activate"
      ;;
    wsl)
      echo "  Option 1: Install uv inside WSL:"
      echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
      echo ""
      echo "  Option 2: Run this script from Windows cmd/PowerShell instead,"
      echo "    where uv/pip are already available."
      echo ""
      echo "  Option 3: Override with a known path:"
      echo "    PIP_CMD=\"/path/to/pip\" bash scripts/setup_sam3.sh"
      ;;
    gitbash)
      echo "  Option 1: Run from Windows cmd/PowerShell where uv/pip are on PATH:"
      echo "    bash scripts/setup_sam3.sh"
      echo ""
      echo "  Option 2: Activate your venv first:"
      echo "    source .venv/Scripts/activate && bash scripts/setup_sam3.sh"
      echo ""
      echo "  Option 3: Override:"
      echo "    PIP_CMD=\"uv pip\" bash scripts/setup_sam3.sh"
      ;;
    *)
      echo "  Install uv: https://docs.astral.sh/uv/getting-started/installation/"
      echo "  Or override: PIP_CMD=\"pip3\" bash scripts/setup_sam3.sh"
      ;;
  esac
  exit 1
fi

echo "[setup] Project root: $PROJECT_ROOT"
echo ""

# =============================================================================
# 5. CLONE SAM3
# =============================================================================
echo "[1/4] Cloning facebookresearch/sam3 ..."
if [ -d "$SAM3_SRC/.git" ]; then
  echo "      Already exists at $SAM3_SRC (delete to re-clone)"
else
  rm -rf "$SAM3_SRC"
  if ! git clone --depth 1 "$SAM3_CLONE_URL" "$SAM3_SRC"; then
    echo ""
    echo "[error] git clone failed. Possible fixes:"
    echo "  - Check your internet connection"
    echo "  - Use a mirror:"
    echo "    SAM3_CLONE_URL=\"https://gitclone.com/github.com/facebookresearch/sam3.git\" bash scripts/setup_sam3.sh"
    exit 1
  fi
fi

# =============================================================================
# 6. INSTALL SAM3 PACKAGE (with extras for missing transitive dependencies)
# =============================================================================
# SAM3's core deps omit einops, pycocotools, etc. that are imported unconditionally.
# Installing with [dev,notebooks] extras pulls them in so users don't hit ImportErrors.
echo ""
echo "[2/4] Installing SAM3 package (with dev+notebooks extras) ..."
pip_install -e "$SAM3_SRC[dev,notebooks]"

# =============================================================================
# 7. PATCH: Make triton import optional (Linux-only, not needed for image segmentation)
# =============================================================================
# SAM3's edt.py imports triton at module level, but triton is Linux-only.
# Edit-Banana only uses image segmentation, not video tracking (which needs EDT).
# This patch makes the import conditional so SAM3 loads on all platforms.
echo ""
echo "[patch] Making triton import optional for cross-platform support ..."

# Use Python for reliable cross-platform patching (sed behaves differently on macOS/Linux/Windows)
PATCH_PYTHON=""
for py in \
  "$PROJECT_ROOT/.venv/Scripts/python.exe" \
  "$PROJECT_ROOT/.venv/bin/python" \
  python3 python python3.exe python.exe \
; do
  if command -v "$py" &>/dev/null || [ -f "$py" ]; then
    PATCH_PYTHON="$py"
    break
  fi
done

if [ -n "$PATCH_PYTHON" ]; then
  "$PATCH_PYTHON" -c "
import os, sys

sam3_src = sys.argv[1]

# Patch edt.py: wrap 'import triton' in try/except
edt = os.path.join(sam3_src, 'sam3', 'model', 'edt.py')
if os.path.isfile(edt):
    text = open(edt).read()
    if 'import triton\n' in text and 'HAS_TRITON' not in text:
        text = text.replace(
            'import triton\nimport triton.language as tl',
            'try:\n    import triton\n    import triton.language as tl\n    HAS_TRITON = True\nexcept ImportError:\n    HAS_TRITON = False'
        )
        open(edt, 'w').write(text)
        print('      Patched edt.py')
    else:
        print('      edt.py already patched')

# Patch sam3_tracker_utils.py: wrap 'from sam3.model.edt import edt_triton' in try/except
tracker = os.path.join(sam3_src, 'sam3', 'model', 'sam3_tracker_utils.py')
if os.path.isfile(tracker):
    text = open(tracker).read()
    old = 'from sam3.model.edt import edt_triton'
    if old in text and 'edt_triton = None' not in text:
        text = text.replace(old,
            'try:\n    from sam3.model.edt import edt_triton\nexcept ImportError:\n    edt_triton = None  # triton unavailable (Windows/macOS); not needed for image segmentation'
        )
        open(tracker, 'w').write(text)
        print('      Patched sam3_tracker_utils.py')
    else:
        print('      sam3_tracker_utils.py already patched')
" "$SAM3_SRC"
else
  echo "      [warn] No Python found to apply patch. You may need to install 'triton' manually."
fi

# =============================================================================
# 8. COPY BPE VOCAB
# =============================================================================
echo ""
echo "[4/4] Copying BPE vocab to models/ ..."
mkdir -p "$MODELS_DIR"
BPE_NAME="bpe_simple_vocab_16e6.txt.gz"
BPE_FOUND=false

for BPE_SRC in \
  "$SAM3_SRC/assets/$BPE_NAME" \
  "$SAM3_SRC/sam3/assets/$BPE_NAME" \
; do
  if [ -f "$BPE_SRC" ]; then
    cp "$BPE_SRC" "$MODELS_DIR/"
    echo "      Copied to $MODELS_DIR/$BPE_NAME"
    BPE_FOUND=true
    break
  fi
done

if [ "$BPE_FOUND" = false ]; then
  echo "      [warn] BPE file not found in expected locations."
  echo "      Searching the cloned repo:"
  find "$SAM3_SRC" -name "*.gz" 2>/dev/null || true
  echo ""
  echo "      Copy the file manually: cp <path> $MODELS_DIR/$BPE_NAME"
fi

# =============================================================================
# 9. VERIFY INSTALLATION
# =============================================================================
echo ""
echo "[verify] Testing SAM3 import ..."

# Use the same Python we found earlier for patching
VERIFY_PY="${PATCH_PYTHON:-python3}"

if "$VERIFY_PY" -c "from sam3.model_builder import build_sam3_image_model; print('OK')" 2>/dev/null; then
  echo ""
  echo -e "\033[32m=========================================\033[0m"
  echo -e "\033[32m  SAM3 library installed successfully\033[0m"
  echo -e "\033[32m=========================================\033[0m"
  echo ""
  echo "Next steps:"
  echo "  1. Download SAM3 weights (sam3.pt) into models/"
  echo "     See docs/SETUP_SAM3.md for download links"
  echo ""
  echo "  2. Create config (if not done):"
  echo "     cp config/config.yaml.example config/config.yaml"
else
  echo ""
  echo -e "\033[31m=========================================\033[0m"
  echo -e "\033[31m  SAM3 import failed\033[0m"
  echo -e "\033[31m=========================================\033[0m"
  echo ""
  echo "Run this to see the full error:"
  echo "  python -c \"from sam3.model_builder import build_sam3_image_model\""
  echo ""
  echo "Common fixes:"
  echo "  - Missing dependency: pip install <module_name>"
  echo "  - Wrong Python: make sure you're using the venv Python"
  exit 1
fi

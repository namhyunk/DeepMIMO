#!/usr/bin/env bash
# Pre-flight + symlink helper for the DICHASUS dc41 sweep.
#
# 02_dichasus_full_sweep.sh expects three groups of assets that ship outside
# the DeepMIMO repo. This script:
#   * verifies which are present,
#   * symlinks the assets that already exist on disk
#     (reftx-offsets, inue_simple scene),
#   * prints exact instructions for the assets that must be fetched manually
#     (dichasus-dc41.tfrecords from the DICHASUS website, inue_detailed XMLs
#     exported from the Blender source).
#
# Idempotent: re-running after fetching the missing files completes the setup.
# Does NOT download anything — DICHASUS publication requires accepting a
# licence interactively, so we don't curl it for you.
#
# Usage:
#   bash fidelity/experiments/02b_setup_dichasus_assets.sh
#
# Exit code:
#   0  all assets in place — 02_dichasus_full_sweep.sh is ready to run
#   2  one or more assets still missing (script printed what to do next)

set -euo pipefail

cd "$(dirname "$0")/../.."

REPO_ROOT="$(pwd)"
DICHASUS_DIR="$REPO_ROOT/fidelity/dichasus"
DCXX_DIR="$DICHASUS_DIR/dcxx"
DIFFRT_LINK_DIR="$DICHASUS_DIR/diff-rt-calibration"
ASUNOKIA_ROOT="$(realpath "$REPO_ROOT/..")"
DIFFRT_REAL_DIR="$ASUNOKIA_ROOT/diff-rt-calibration"

mkdir -p "$DCXX_DIR" "$DCXX_DIR/inue_detailed"

missing=0

echo "==================== DICHASUS dc41 asset preflight ===================="
echo "repo root         : $REPO_ROOT"
echo "dichasus dir      : $DICHASUS_DIR"
echo "diff-rt source    : $DIFFRT_REAL_DIR"
echo

# ---- 1. inue_simple scene XML (sibling repo) --------------------------------
src="$DIFFRT_REAL_DIR/scenes/inue_simple/inue_simple.xml"
dst_dir="$DIFFRT_LINK_DIR/scenes/inue_simple"
dst="$dst_dir/inue_simple.xml"
if [[ -f "$src" ]]; then
    mkdir -p "$dst_dir"
    if [[ ! -e "$dst" ]]; then
        ln -s "$src" "$dst"
        echo "[ok ] linked  $dst -> $src"
    else
        echo "[ok ] present $dst"
    fi
else
    echo "[!! ] inue_simple.xml not found at $src"
    echo "       Clone NVlabs/diff-rt-calibration into $DIFFRT_REAL_DIR."
    missing=$((missing+1))
fi

# ---- 2. reftx-offsets json (lives in diff-rt-calibration/data/) -------------
src="$DIFFRT_REAL_DIR/data/reftx-offsets-dichasus-dc41.json"
dst="$DCXX_DIR/reftx-offsets-dichasus-dc41.json"
if [[ -f "$src" ]]; then
    if [[ ! -e "$dst" ]]; then
        ln -s "$src" "$dst"
        echo "[ok ] linked  $dst -> $src"
    else
        echo "[ok ] present $dst"
    fi
else
    echo "[!! ] reftx-offsets-dichasus-dc41.json missing at $src"
    missing=$((missing+1))
fi

# ---- 3. dichasus-dc41.tfrecords (must be fetched from DICHASUS) -------------
dst="$DCXX_DIR/dichasus-dc41.tfrecords"
src_candidates=(
    "$DIFFRT_REAL_DIR/data/tfrecords/dichasus-dc41.tfrecords"
    "$HOME/dichasus-dc41.tfrecords"
)
found_src=""
for c in "${src_candidates[@]}"; do
    if [[ -f "$c" ]]; then found_src="$c"; break; fi
done
if [[ -f "$dst" || -L "$dst" ]]; then
    real="$(readlink -f "$dst" 2>/dev/null || echo "$dst")"
    sz="$(stat -c '%s' "$real" 2>/dev/null || echo 0)"
    echo "[ok ] present $dst (size=${sz} bytes -> $real)"
elif [[ -n "$found_src" ]]; then
    ln -s "$found_src" "$dst"
    echo "[ok ] linked  $dst -> $found_src"
else
    echo "[!! ] dichasus-dc41.tfrecords missing — fetch it manually:"
    echo "       1) accept the licence + download from"
    echo "          https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-dc41/"
    echo "       2) place the .tfrecords file at one of:"
    echo "          $DIFFRT_REAL_DIR/data/tfrecords/dichasus-dc41.tfrecords"
    echo "          $HOME/dichasus-dc41.tfrecords"
    echo "          (or directly at $dst)"
    echo "       3) re-run this script to symlink it into place."
    missing=$((missing+1))
fi

# ---- 4. inue_detailed XMLs (Blender export, project asset) ------------------
inue_files=("inue_detailed.xml" "inue_no_blockers.xml")
inue_missing=0
for f in "${inue_files[@]}"; do
    dst="$DCXX_DIR/inue_detailed/$f"
    if [[ -f "$dst" ]]; then
        echo "[ok ] present $dst"
    else
        echo "[!! ] $dst missing"
        inue_missing=$((inue_missing+1))
    fi
done
if (( inue_missing > 0 )); then
    cat <<MSG
       The detailed Sionna scene XMLs are not in version control. They must
       be exported from the project Blender source (inue_detailed.blend) via
       fidelity/dichasus/export_blend_to_xml.py (or export_blend_v2.py):

         cd $DICHASUS_DIR
         blender -b /path/to/inue_detailed.blend \\
                 --python export_blend_to_xml.py -- \\
                 --output dcxx/inue_detailed/

       If the .blend lives in another working tree, copy or symlink the
       resulting inue_detailed.xml and inue_no_blockers.xml into
       $DCXX_DIR/inue_detailed/ and re-run this script.
MSG
    missing=$((missing+inue_missing))
fi

# ---- 5. traced paths (regenerable by run_dc41_gradient_calibration.py) ------
echo
echo "Optional: pre-traced gradient paths (1.9 GB, regenerable). Currently:"
gp="$DIFFRT_REAL_DIR/data/traced_paths/dichasus-dc41_grad.tfrecords"
if [[ -f "$gp" ]]; then
    echo "[ok ] present $gp"
else
    echo "[ -- ] absent  $gp"
    echo "        Regenerate with:"
    echo "          cd $DIFFRT_REAL_DIR/code"
    echo "          python run_dc41_gradient_calibration.py"
fi

echo
echo "=================== preflight summary ==================="
if (( missing == 0 )); then
    echo "All required DICHASUS dc41 assets are in place."
    echo "Next: bash fidelity/experiments/02_dichasus_full_sweep.sh"
    exit 0
else
    echo "$missing required asset(s) still missing — see [!!] lines above."
    exit 2
fi

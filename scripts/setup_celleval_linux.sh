#!/bin/sh
# Build the cell-eval environment on Linux, next to the Windows one.
#
#   sh scripts/setup_celleval_linux.sh
#
# Why this exists. .env-celleval in this repo is a WINDOWS embedded Python - it
# holds DLLs/, Library/, python311.dll and no bin/. With the repo mounted into a
# container that directory is still visible, so os.path.exists says the
# interpreter is there, but exec'ing it fails with
#
#     WSL (3462 - ) ERROR: UtilBindVsockAnyPort:309: socket failed 1
#     cell-eval failed (exit 1)
#
# which names nothing and costs a full export before it appears. This builds the
# Linux counterpart in a SEPARATE directory; run_celleval.py prefers whichever
# interpreter can actually run, so both hosts work from the same checkout.
#
# It stays a separate interpreter rather than a pip install into the training
# environment because cell-eval's dependency set conflicts with this project's -
# that conflict is the entire reason the split exists.
set -e

TARGET=".env-celleval-linux"
VERSION="0.5.42"          # pinned: the Windows env is 0.5.42, and the two must
                          # agree or numbers measured on either stop comparing.
# From the package metadata, not from the Windows env's own version:
#   Requires-Python: >=3.10,<3.13
# 3.10 and 3.12 are both fine. Preferring 3.11 only matches the Windows side.
MIN_PYTHON=310
MAX_PYTHON=312

if [ -x "$TARGET/bin/python" ]; then
  echo "$TARGET already exists. Delete it to rebuild:  rm -rf $TARGET"
  "$TARGET/bin/python" -c "import cell_eval; print('cell-eval already importable')"
  exit 0
fi

# uv resolves and installs an order of magnitude faster, and can fetch a 3.11 that
# the image does not ship. Plain venv is the fallback, not the preference.
if command -v uv >/dev/null 2>&1; then
  echo "using uv"
  uv venv --python 3.11 "$TARGET"
  uv pip install --python "$TARGET/bin/python" "cell-eval==$VERSION"
else
  INTERPRETER=""
  SEEN=""
  # 3.11 first only to match the Windows env; 3.10 and 3.12 are equally valid.
  for candidate in python3.11 python3.12 python3.10 python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version=$("$candidate" -c 'import sys; print("%d%02d" % sys.version_info[:2])' 2>/dev/null || echo 0)
    SEEN="$SEEN $candidate=$version"
    if [ "$version" -ge "$MIN_PYTHON" ] && [ "$version" -le "$MAX_PYTHON" ]; then
      INTERPRETER="$candidate"
      break
    fi
  done
  if [ -z "$INTERPRETER" ]; then
    # Report what was actually found. "not found" without the versions present
    # gives no way to tell "nothing installed" from "installed but too new".
    echo "no python in [3.10, 3.12] found - cell-eval $VERSION needs >=3.10,<3.13" >&2
    echo "  looked at:${SEEN:- (nothing)}" >&2
    echo "" >&2
    echo "  easiest fix - uv fetches its own interpreter:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "    . \$HOME/.local/bin/env" >&2
    echo "    sh scripts/setup_celleval_linux.sh" >&2
    echo "" >&2
    echo "  or apt:  apt-get install -y python3.11 python3.11-venv" >&2
    exit 1
  fi
  echo "using $INTERPRETER ($($INTERPRETER --version 2>&1))"
  # Debian/Ubuntu slim images ship python3 without the venv module, and the
  # failure message from `-m venv` names ensurepip rather than the package to
  # install, so it is checked separately.
  if ! "$INTERPRETER" -c "import venv, ensurepip" 2>/dev/null; then
    echo "$INTERPRETER has no venv/ensurepip module." >&2
    echo "  apt-get install -y ${INTERPRETER#python}-venv   (e.g. python3.11-venv)" >&2
    echo "  or use uv:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  "$INTERPRETER" -m venv "$TARGET"
  "$TARGET/bin/python" -m pip install --quiet --upgrade pip
  "$TARGET/bin/python" -m pip install "cell-eval==$VERSION"
fi

# Verify rather than assume: a wheel can install and still fail to import when a
# native dependency is missing from the image.
"$TARGET/bin/python" -c "import cell_eval, importlib.metadata as m; \
print('ok - cell-eval', m.version('cell-eval'))"

echo ""
echo "done. run_celleval.py will now pick this up automatically:"
echo "  python scripts/run_celleval.py results/runs/<tag> --score-only --profile full"

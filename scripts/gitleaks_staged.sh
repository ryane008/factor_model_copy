#!/usr/bin/env bash
# Scan staged changes for secrets. Used by .pre-commit-config.yaml.
#
# Two things gitleaks does not do on its own, both learned the hard way here:
#
# 1. Scan the RAW bytes being committed. The nbstripout setup registers a diff
#    textconv that renders notebooks WITHOUT outputs -- and notebook output is
#    exactly where this repo's WRDS password leaked. Neutralising textconv makes
#    gitleaks see what is actually going into the commit.
#
# 2. Fail CLOSED. If `git diff` errors (a textconv crash, a corrupt blob), gitleaks
#    logs "scanned ~0 bytes ... no leaks found" and exits 0, so the hook silently
#    passes. We check the diff can be produced first and refuse the commit if not.
set -euo pipefail

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.ipynb.textconv
export GIT_CONFIG_VALUE_0=cat

if ! command -v gitleaks >/dev/null 2>&1; then
    echo "gitleaks is not installed (brew install gitleaks) — refusing to commit unscanned." >&2
    exit 1
fi

if ! git diff -U0 --no-ext-diff --staged . >/dev/null 2>/tmp/gitleaks_staged.$$; then
    echo "Could not read the staged diff, so it cannot be scanned for secrets:" >&2
    head -5 /tmp/gitleaks_staged.$$ >&2
    rm -f /tmp/gitleaks_staged.$$
    exit 1
fi
rm -f /tmp/gitleaks_staged.$$

exec gitleaks git --pre-commit --staged --redact --no-banner --config .gitleaks.toml .

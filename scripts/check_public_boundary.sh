#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v rg >/dev/null 2>&1; then
  echo "ripgrep (rg) is required for boundary checks"
  exit 1
fi

PATTERN='(fundraising|go-to-market|pricing strategy|revenue forecast|investor deck|sales pipeline|internal runbook|secret key|api key|private credential)'

if rg -n -i \
  --glob '*.md' \
  --glob '*.txt' \
  --glob '*.yaml' \
  --glob '*.yml' \
  --glob '!AGENTS.md' \
  --glob '!docs/PUBLIC_INTERNAL_BOUNDARY.md' \
  "$PATTERN" "$ROOT"; then
  echo "Public boundary check failed"
  exit 1
fi

echo "Public boundary check passed"


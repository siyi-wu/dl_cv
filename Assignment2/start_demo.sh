#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_FILE="${SCRIPT_DIR}/demo/index.html"

if [[ ! -f "${DEMO_FILE}" ]]; then
  printf '[ERROR] Cannot find %s\n' "${DEMO_FILE}" >&2
  printf 'Keep start_demo.sh, demo/, and outputs/ in the same project directory.\n' >&2
  exit 1
fi

open_demo() {
  local system_name
  system_name="$(uname -s)"

  case "${system_name}" in
    Darwin)
      open "${DEMO_FILE}"
      ;;
    Linux)
      if grep -qi microsoft /proc/version 2>/dev/null && command -v explorer.exe >/dev/null 2>&1; then
        explorer.exe "$(wslpath -w "${DEMO_FILE}")"
      elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${DEMO_FILE}"
      elif command -v gio >/dev/null 2>&1; then
        gio open "${DEMO_FILE}"
      else
        return 1
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      local windows_path="${DEMO_FILE}"
      if command -v cygpath >/dev/null 2>&1; then
        windows_path="$(cygpath -w "${DEMO_FILE}")"
      fi
      cmd.exe /c start "" "${windows_path}"
      ;;
    *)
      return 1
      ;;
  esac
}

if open_demo; then
  printf 'Opened MNIST DDPM showcase:\n  %s\n' "${DEMO_FILE}"
else
  printf '[ERROR] No supported graphical browser opener was found.\n' >&2
  printf 'Open this file manually in a browser:\n  %s\n' "${DEMO_FILE}" >&2
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail
python3 scripts/verify_repository.py
python3 scripts/check_python_syntax.py
python3 scripts/verify_scientific_results.py
python3 scripts/verify_sensitivity_analysis.py
python3 scripts/check_documentation_paths.py

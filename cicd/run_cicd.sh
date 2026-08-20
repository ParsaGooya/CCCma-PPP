#!/bin/bash

set -e

log_file="cicd/project_checks.log"
exec > >(tee "$log_file") 2>&1

compilation=True
formatting_and_linting=True
dependency_security_audit=False
code_security_audit=False
documentation_regeneration=True
unit_test_suite=True
integration_test_subset=False

echo "========================================"
echo "Running project checks"
echo "========================================"

if [ ! -d "cccma_ppp" ]; then
    echo "ERROR: cccma_ppp directory not found."
    exit 1
fi

cd cccma_ppp

echo "Working directory: $(pwd)"

if [ "$compilation" = "True" ]; then
    echo ""
    echo "Running compilation check..."

    compilation_output=$(python -m compileall -q . 2>&1)

    if [ -n "$compilation_output" ]; then
        echo "Compilation check failed."
        echo "$compilation_output"
        exit 1
    fi

    echo "Compilation check passed."
fi

if [ "$formatting_and_linting" = "True" ]; then
    echo ""
    echo "Running Ruff linting..."

    ruff_output=$(ruff check . 2>&1 || true)
    echo "$ruff_output"

    if ! echo "$ruff_output" | grep -q "No fixes available"; then
        echo "ERROR: Ruff did not report 'No fixes available'."
        exit 1
    fi

    echo "Ruff reported: No fixes available."

    echo ""
    echo "Running Ruff formatting check..."

    format_output=$(ruff format --check . 2>&1 || true)
    echo "$format_output"

    if echo "$format_output" | grep -q "would be formatted"; then
        echo "ERROR: Ruff found files that would be formatted."
        exit 1
    fi

    echo "Ruff formatting check passed."
fi

if [ "$dependency_security_audit" = "True" ]; then
    echo ""
    echo "Running dependency security audit..."

    if ! pip-audit; then
        echo "Dependency security audit failed."
        exit 1
    fi

    echo "Dependency security audit passed."
fi

if [ "$code_security_audit" = "True" ]; then
    echo ""
    echo "Running code security audit..."

    if ! bandit -r .; then
        echo "Code security audit failed."
        exit 1
    fi

    echo "Code security audit passed."
fi

if [ "$documentation_regeneration" = "True" ]; then
    echo ""
    echo "Regenerating Sphinx documentation..."

    python3 ../scripts/build_and_view_docs.py --no-serve
fi

if [ "$unit_test_suite" = "True" ]; then
    echo ""
    echo "Marking failed unit tests..."

    if test_output=$(
        cd ..
        python3 scripts/mark_failing_tests.py 2>&1
    ); then
        test_rc=0
    else
        test_rc=$?
    fi

    printf '%s\n' "$test_output"

    if [ "$test_rc" -ne 0 ]; then
        echo "Unit test check failed with exit code $test_rc."
        exit "$test_rc"
    fi

    if echo "$test_output" | grep -q "No failed test functions were found"; then
        echo "Unit test check passed."
    else
        echo "Unit test check failed."
        exit 1
    fi

    echo ""
    echo "Checking unit test suite coverage..."

    if (
        cd ..
        python3 scripts/check_coverage.py
    ); then
        echo "Unit test coverage check passed."
    else
        echo "Unit test coverage check failed."
        exit 1
    fi
fi

if [ "$integration_test_subset" = "True" ]; then
    echo ""
    echo "Running integration test subset..."

    if ! python3 ../scripts/run_integration_tests.py; then
        echo "Integration test subset failed."
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "All enabled checks passed!"
echo "========================================"
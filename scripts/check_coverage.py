import os
import subprocess
import sys
import xml.etree.ElementTree as ET

# Run pytest with coverage tracking
# This executes all tests in the tests/ directory and measures code coverage
result = subprocess.run(
    [
        "pytest",
        "tests",  # Run all tests in the tests directory
        "--cov=cccma_ppp",  # Measure coverage for this package
        "--cov-report=term-missing",  # Print coverage to terminal (show missing lines)
        "--cov-report=xml",  # Also save coverage data as XML (for parsing later)
        "--cov-branch",  # Track branch coverage (if/else paths) not just line coverage
        "--cov-fail-under=90",  # Fail if overall coverage drops below 90%
    ]
)

# If pytest failed (bad test or low coverage), stop here and exit with failure code
if result.returncode != 0:
    sys.exit(result.returncode)

# Now we do additional checks on branch coverage specifically
# (the --cov-fail-under checks overall coverage, but we want stricter branch checks)

coverage_file = "cicd/coverage.xml"

# Make sure the coverage XML file was actually created
if not os.path.exists(coverage_file):
    print("coverage.xml not found")
    sys.exit(1)

# Parse the XML file that pytest coverage wrote
tree = ET.parse(coverage_file)
root = tree.getroot()

# Extract branch coverage percentage from the XML
# The XML stores it as a decimal (0-1), so multiply by 100 to get percentage
branch_rate = float(root.attrib.get("branch-rate", 0)) * 100
print(f"Branch coverage: {branch_rate:.2f}%")

# Branch coverage threshold (stricter than the overall 90% threshold)
# This catches cases where line coverage is high but branch coverage is low
# (e.g., lots of untested if/else paths)
if branch_rate < 85:
    print("Branch coverage below threshold (85%)")
    sys.exit(1)

# All checks passed!
print("Coverage checks passed")

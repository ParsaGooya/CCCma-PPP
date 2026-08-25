import os
import subprocess
import sys
import xml.etree.ElementTree as ET

result = subprocess.run(
    [
        "pytest",
        "tests",
        "--cov=cccma_ppp",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-branch",
        "--cov-fail-under=90",
    ]
)

if result.returncode != 0:
    sys.exit(result.returncode)

coverage_file = "cicd/coverage.xml"

if not os.path.exists(coverage_file):
    print("coverage.xml not found")
    sys.exit(1)

tree = ET.parse(coverage_file)
root = tree.getroot()

branch_rate = float(root.attrib.get("branch-rate", 0)) * 100
print(f"Branch coverage: {branch_rate:.2f}%")

if branch_rate < 85:
    print("Branch coverage below threshold (85%)")
    sys.exit(1)

print("Coverage checks passed")

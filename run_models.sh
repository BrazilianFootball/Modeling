#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd "$(dirname "$0")/scripts" || { echo "Could not find scripts directory"; exit 1; }

run_script() {
    echo -e "${BLUE}Running $1...${NC}"
    if python "$1"; then
        echo -e "${GREEN}✓ $1 executed successfully!${NC}"
        return 0
    else
        echo -e "${RED}✗ Error executing $1!${NC}"
        return 1
    fi
}

echo "=== Starting model execution ==="

run_script "bradley_terry_1.py" || exit 1
run_script "bradley_terry_2.py" || exit 1
run_script "poisson_1.py" || exit 1
run_script "poisson_2.py" || exit 1

clear

echo -e "${GREEN}=== All models executed successfully! ===${NC}"
echo "=== Starting analysis ==="

run_script "analysis.py" || exit 1

echo -e "${GREEN}=== Analysis completed successfully! ===${NC}"

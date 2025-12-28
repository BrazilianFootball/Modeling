#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(dirname "$0")"
cd "$PROJECT_ROOT" || { echo "Could not find scripts directory"; exit 1; }

run_script() {
    echo -e "${BLUE}Running $1...${NC}"
    if caffeinate -dims python "src/models/$1"; then
        echo -e "${GREEN}✓ $1 executed successfully!${NC}"
        return 0
    else
        echo -e "${RED}✗ Error executing $1!${NC}"
        return 1
    fi
}

export PYTHONPATH=$PYTHONPATH:"$PROJECT_ROOT"
echo "Setting PYTHONPATH to: $PYTHONPATH"

echo "=== Starting model execution ==="

run_script "club_bradley_terry_1.py" || exit 1
run_script "club_bradley_terry_2.py" || exit 1
run_script "club_bradley_terry_3.py" || exit 1
run_script "club_bradley_terry_4.py" || exit 1
run_script "club_poisson_1.py" || exit 1
run_script "club_poisson_2.py" || exit 1
run_script "club_poisson_3.py" || exit 1
run_script "club_poisson_4.py" || exit 1
run_script "club_poisson_5.py" || exit 1
run_script "club_poisson_6.py" || exit 1
run_script "club_poisson_7.py" || exit 1
run_script "club_poisson_8.py" || exit 1
run_script "club_poisson_9.py" || exit 1
run_script "club_poisson_10.py" || exit 1
run_script "player_bradley_terry_3.py" || exit 1
# run_script "player_bradley_terry_4.py" || exit 1
# run_script "player_poisson_1.py" || exit 1
# run_script "player_poisson_2.py" || exit 1
# run_script "player_poisson_3.py" || exit 1
# run_script "player_poisson_4.py" || exit 1
# run_script "player_poisson_6.py" || exit 1
# run_script "player_poisson_7.py" || exit 1
# run_script "player_poisson_8.py" || exit 1
# run_script "player_poisson_9.py" || exit 1

clear

echo -e "${GREEN}=== All models executed successfully! ===${NC}"
echo "=== Starting analysis ==="

run_script "../features/analysis.py" || exit 1

echo -e "${GREEN}=== Analysis completed successfully! ===${NC}"

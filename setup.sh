#!/bin/bash

# Initial setup script for Brazilian Football Modeling project
# This script facilitates the initial environment setup

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Brazilian Football Modeling Project Setup ===${NC}"
echo ""

# Check if Python 3.10 is available
echo -e "${BLUE}Checking Python 3.10...${NC}"
if ! command -v python3.10 >/dev/null 2>&1; then
    echo -e "${RED}Python 3.10 not found!${NC}"
    echo -e "${YELLOW}Options to install Python 3.10:${NC}"
    echo "1. Homebrew: brew install python@3.10"
    echo "2. pyenv: pyenv install 3.10.0"
    echo "3. Conda: conda install python=3.10"
    echo "4. Official download: https://www.python.org/downloads/"
    exit 1
fi

echo -e "${GREEN}✓ Python 3.10 found${NC}"

# Check if make is available
if ! command -v make >/dev/null 2>&1; then
    echo -e "${YELLOW}Make not found. Installing via Homebrew...${NC}"
    if command -v brew >/dev/null 2>&1; then
        brew install make
    else
        echo -e "${RED}Install make manually or use Python commands directly${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Make found${NC}"

# Run setup
echo -e "${BLUE}Setting up environment...${NC}"
make setup

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Setup completed successfully! ===${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Activate environment: ${YELLOW}source venv/bin/activate${NC}"
    echo "2. Run models: ${YELLOW}make run-models${NC}"
    echo "3. See all commands: ${YELLOW}make help${NC}"
    echo ""
    echo -e "${BLUE}Or use make commands directly:${NC}"
    echo "  make run-models         # Run all models"
    echo "  make run-analysis       # Run analysis"
    echo "  make jupyter            # Start Jupyter Lab"
    echo "  make status             # Show environment status"
else
    echo -e "${RED}Setup error!${NC}"
    exit 1
fi
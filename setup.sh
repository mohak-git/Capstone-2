#!/bin/bash

set -e

CYAN="\e[1;36m"
YELLOW="\e[1;33m"
GREEN="\e[1;32m"
RESET="\e[0m"

echo -e "${CYAN}=== Setting up the project ===${RESET}"

# Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${RESET}"
    python -m venv venv
else
    echo -e "${GREEN}Virtual environment already exists. Skipping.${RESET}"
fi

echo -e "${YELLOW}Activating virtual environment...${RESET}"
source venv/Scripts/activate

# Install Dependencies
echo -e "${YELLOW}Upgrading pip...${RESET}"
python -m pip install --upgrade pip

echo -e "${YELLOW}Installing dependencies...${RESET}"
pip install -r requirements.txt

# Run Pipeline
echo -e "${YELLOW}Running pipeline script...${RESET}"
python pipeline.py

echo -e "${GREEN}=== Done! ===${RESET}"
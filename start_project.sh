#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Smart CV Screening & ChatBot Setup ===${NC}"

# 1. Check/Create Virtual Environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Virtual environment found.${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Virtual environment (.venv) found.${NC}"
    source .venv/bin/activate
else
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment created.${NC}"
fi

# 2. Install Backend Dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
pip install -r requirements.txt
pip install -U pydantic settings-pydantic # Ensure these are up to date

# 3. Start Infrastructure (Docker)
echo -e "${BLUE}Starting Docker services (Redis, Postgres)...${NC}"
docker-compose up -d redis db

# Wait for services to be ready
echo -e "${BLUE}Waiting for services to be ready...${NC}"
sleep 5

# 4. Start Celery Worker (REQUIRED for CV processing)
echo -e "${GREEN}Starting Celery Worker...${NC}"
celery -A app.core.celery_app worker -l info -c 1 &
CELERY_PID=$!
echo -e "${GREEN}✓ Celery worker started (PID: $CELERY_PID)${NC}"

# 5. Start Backend
echo -e "${GREEN}Starting Backend API...${NC}"
# Run in background
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# 6. Setup & Start Frontend
echo -e "${BLUE}Setting up Frontend...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    npm install
fi

echo -e "${GREEN}Starting Frontend...${NC}"
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID $CELERY_PID 2>/dev/null" EXIT

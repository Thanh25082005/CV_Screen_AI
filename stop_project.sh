#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}=== Stopping Smart CV Screening Project ===${NC}"

# 1. Stop Backend (Uvicorn)
echo -e "${BLUE}Stopping Backend API...${NC}"
pkill -f "uvicorn app.main:app" || echo "Backend not running."

# 2. Stop Frontend (Next.js)
echo -e "${BLUE}Stopping Frontend...${NC}"
pkill -f "next-server" || echo "Frontend not running."
# Sometimes npm run dev spawns child processes
pkill -f "next dev" || echo "Frontend dev server not running."

# 3. Stop Docker Services
echo -e "${BLUE}Stopping Docker services...${NC}"
docker-compose stop

echo -e "${GREEN}✓ Project stopped successfully.${NC}"

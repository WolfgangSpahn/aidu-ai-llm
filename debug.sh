#!/usr/bin/env bash
# Quick debugging workflow for AIDU LLM API

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if server is running
check_server() {
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        print_success "Server is running on http://localhost:8000"
        return 0
    else
        print_error "Server is not running"
        return 1
    fi
}

# Run the debug script
run_tests() {
    print_header "Running Debug Tests"
    if command -v python &> /dev/null; then
        if [ -f "debug_api.py" ]; then
            uv run python debug_api.py
        else
            print_error "debug_api.py not found"
            return 1
        fi
    else
        print_error "Python not found"
        return 1
    fi
}

# Main menu
show_menu() {
    echo -e "\n${BLUE}AIDU LLM Debug Menu${NC}"
    echo "1) Check server status"
    echo "2) Run debug tests"
    echo "3) View server logs (tail)"
    echo "4) Kill all servers"
    echo "5) Start server in background"
    echo "6) Exit"
    echo ""
}

case "${1:-menu}" in
    status)
        print_header "Server Status"
        if check_server; then
            curl -s http://localhost:8000/docs > /dev/null && print_success "API is responsive"
        else
            print_error "Server is not reachable"
            exit 1
        fi
        ;;
    test)
        if check_server; then
            run_tests
        else
            print_error "Start the server first with: make serve"
            exit 1
        fi
        ;;
    logs)
        print_info "Tailing server logs..."
        journalctl -f -u serve 2>/dev/null || echo "Use: make serve to see logs in this terminal"
        ;;
    kill)
        print_header "Stopping all servers"
        pkill -9 -f "serve.app\|uvicorn" || true
        sleep 1
        print_success "Servers stopped"
        ;;
    start)
        print_header "Starting server"
        cd "$(dirname "$0")"
        make serve &
        sleep 2
        if check_server; then
            print_success "Server started successfully"
        else
            print_error "Failed to start server"
            exit 1
        fi
        ;;
    *)
        show_menu
        read -p "Choose option (1-6): " choice
        case "$choice" in
            1) exec "$0" status ;;
            2) exec "$0" test ;;
            3) exec "$0" logs ;;
            4) exec "$0" kill ;;
            5) exec "$0" start ;;
            6) echo "Goodbye!"; exit 0 ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac
        ;;
esac

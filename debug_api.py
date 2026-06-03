#!/usr/bin/env python
"""
Comprehensive API debugging script with detailed output.
"""

import requests
import json
import sys
from typing import Optional
import time

BASE_URL = "http://localhost:8000"
TIMEOUT = 10


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text: str):
    print(f"{Colors.BLUE}→ {text}{Colors.RESET}")


def print_response(method: str, endpoint: str, status: int, data: dict, elapsed: float):
    print(f"\n{Colors.YELLOW}Request:{Colors.RESET} {method} {endpoint}")
    print(f"{Colors.YELLOW}Status:{Colors.RESET} {status}")
    print(f"{Colors.YELLOW}Time:{Colors.RESET} {elapsed:.2f}s")
    print(f"{Colors.YELLOW}Response:{Colors.RESET}")
    print(json.dumps(data, indent=2))


def check_server_health() -> bool:
    """Check if the server is running."""
    print_header("Checking Server Health")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        if response.status_code == 200:
            print_success(f"Server is running at {BASE_URL}")
            return True
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to {BASE_URL}")
        return False
    except Exception as e:
        print_error(f"Error checking server: {e}")
        return False


def test_create_session() -> Optional[str]:
    """Test creating a new session."""
    print_header("Test 1: Create Session")
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/sessions",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        data = response.json()
        print_response("POST", "/sessions", response.status_code, data, elapsed)

        if response.status_code == 201:
            session_id = data.get("session_id")
            print_success(f"Session created: {session_id}")
            return session_id
        else:
            print_error(f"Expected status 201, got {response.status_code}")
            return None

    except Exception as e:
        print_error(f"Failed to create session: {e}")
        return None


def test_chat_message(session_id: str, message: str) -> Optional[str]:
    """Test sending a chat message."""
    print_header("Test 2: Send Chat Message")
    try:
        payload = {"message": message}
        print_info(f"Sending message: '{message}'")

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        data = response.json()
        print_response("POST", f"/sessions/{session_id}/chat", response.status_code, data, elapsed)

        if response.status_code == 200:
            reply = data.get("reply")
            print_success(f"Reply received: {reply[:80]}...")
            return reply
        else:
            print_error(f"Expected status 200, got {response.status_code}")
            return None

    except Exception as e:
        print_error(f"Failed to send chat message: {e}")
        return None


def test_get_history(session_id: str) -> Optional[list]:
    """Test getting session history."""
    print_header("Test 3: Get Session History")
    try:
        start = time.time()
        response = requests.get(
            f"{BASE_URL}/sessions/{session_id}/history",
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        data = response.json()
        print_response("GET", f"/sessions/{session_id}/history", response.status_code, data, elapsed)

        if response.status_code == 200:
            messages = data.get("messages", [])
            print_success(f"Retrieved {len(messages)} messages from history")
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "?")
                content = msg.get("content", "")[:50]
                print(f"  {i}. [{role}] {content}...")
            return messages
        else:
            print_error(f"Expected status 200, got {response.status_code}")
            return None

    except Exception as e:
        print_error(f"Failed to get history: {e}")
        return None


def test_math_problem(session_id: str) -> Optional[str]:
    """Test math problem solving."""
    print_header("Test 4: Math Problem Solving")
    message = "What is 2x + 3 = 11? Solve for x."
    try:
        payload = {"message": message}
        print_info(f"Sending: '{message}'")

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        data = response.json()
        print_response("POST", f"/sessions/{session_id}/chat", response.status_code, data, elapsed)

        if response.status_code == 200:
            reply = data.get("reply")
            if "Solved" in reply or "solution" in reply.lower():
                print_success(f"Math problem solved: {reply[:80]}...")
                return reply
            else:
                print_error(f"Unexpected reply format: {reply}")
                return None
        else:
            print_error(f"Expected status 200, got {response.status_code}")
            return None

    except Exception as e:
        print_error(f"Failed to solve math problem: {e}")
        return None


def main():
    print(f"\n{Colors.BOLD}AIDU LLM API Debug Tool{Colors.RESET}")
    print(f"Target: {BASE_URL}\n")

    # Check server
    if not check_server_health():
        print_error("Server is not reachable. Start it with: make serve")
        sys.exit(1)

    # Run tests
    session_id = test_create_session()
    if not session_id:
        print_error("Failed to create session, cannot continue")
        sys.exit(1)

    test_chat_message(session_id, "Hello, I need help with math")
    test_get_history(session_id)
    test_math_problem(session_id)

    print_header("Summary")
    print_success("All tests completed!")
    print(f"\n{Colors.BOLD}Session ID:{Colors.RESET} {session_id}")
    print(f"{Colors.BOLD}View session history:{Colors.RESET} curl {BASE_URL}/sessions/{session_id}/history\n")


if __name__ == "__main__":
    main()

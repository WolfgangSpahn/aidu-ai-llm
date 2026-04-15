/*
 * Copyright (c) 2026 Wolfgang Spahn, PHBern
 * Licensed under the MIT License.  
 * Please follow standard academic practice when using this software in research or publications.
 * See ../LICENSE for the full text.
 *

 * Description:
 * This file implements a simple browser chatbot UI. It stores chat history
 * and session id in localStorage, renders user/assistant messages, creates
 * a backend session on first use, and sends each prompt to the FastAPI chat
 * endpoint to display the assistant reply.
 */

type Message = {
  // Who sent the message. We only render two kinds of bubbles.
  role: "user" | "assistant";
  // The text shown in the chat bubble.
  content: string;
};

// Grab references to the HTML elements we interact with.
const messagesEl = document.getElementById("messages")!;
const form = document.getElementById("form") as HTMLFormElement;
const input = document.getElementById("input") as HTMLInputElement;

// Restore previous chat history from localStorage so refresh does not lose the conversation.
let messages: Message[] = JSON.parse(localStorage.getItem("chat") || "[]");
// Restore server-side session id so we keep chatting in the same backend session.
let sessionId = localStorage.getItem("chat_session_id") || "";

function save() {
  // Persist both UI history and backend session id.
  localStorage.setItem("chat", JSON.stringify(messages));
  localStorage.setItem("chat_session_id", sessionId);
}

function scrollToBottom() {
  // Keep the latest message visible after each render/update.
  window.scrollTo(0, document.body.scrollHeight);
}

function render() {
  // Rebuild all chat bubbles from current in-memory `messages`.
  messagesEl.innerHTML = messages
    .map(
      (m) => `
      <div class="msg ${m.role}">
        ${escapeHtml(m.content)}
      </div>
    `
    )
    .join("");

  scrollToBottom();
}

function escapeHtml(text: string) {
  // Basic XSS protection: render user/model text as plain text, not HTML.
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function sendMessage(content: string) {
  // 1) Optimistically show the user's message in the UI.
  messages.push({ role: "user", content });
  save();
  render();

  // 2) Show a temporary assistant bubble while waiting for network response.
  const loadingEl = document.createElement("div");
  loadingEl.className = "msg assistant loading";
  loadingEl.textContent = "Thinking...";
  messagesEl.appendChild(loadingEl);
  scrollToBottom();

  try {
    if (!sessionId) {
      // First message in this browser session: create a backend session.
      const createRes = await fetch("/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!createRes.ok) {
        throw new Error(`Failed to create session (${createRes.status})`);
      }

      // Save the new session id so later messages use the same conversation state.
      const createData = await createRes.json();
      sessionId = createData.session_id;
      save();
    }

    // Send one user message to the backend for this specific session id.
    const res = await fetch(`/sessions/${encodeURIComponent(sessionId)}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: content }),
    });

    if (!res.ok) {
      throw new Error(`Chat request failed (${res.status})`);
    }

    const data = await res.json();

    // 3) Append assistant reply returned by the API.
    messages.push({
      role: "assistant",
      content: data.reply,
    });

  } catch (err) {
    // If anything fails (network, server, invalid response), show a friendly fallback.
    messages.push({
      role: "assistant",
      content: "Error contacting server",
    });
    console.error(err);
  }

  save();
  render();
}

// Handle Enter/submit from the form.
form.onsubmit = (e) => {
  e.preventDefault();
  const value = input.value.trim();
  if (!value) return;

  input.value = "";
  sendMessage(value);
};

// Initial paint from restored localStorage state.
render();
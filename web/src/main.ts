/*
 * Copyright (c) 2026 Wolfgang Spahn, PHBern
 * Licensed under the MIT License.  
 * Please follow standard academic practice when using this software in research or publications.
 * See ../LICENSE for the full text.
 *
 * Description:
 * This file implements a browser chatbot UI with markdown and LaTeX math support.
 * Stores chat history and session id in localStorage, renders messages with proper
 * formatting, creates backend session on first use, and sends prompts to the chat API.
 */

import { marked } from 'marked';
import DOMPurify from 'dompurify';
import katex from 'katex';
// @ts-ignore - CSS import doesn't have type declarations
import 'katex/dist/katex.min.css'; // eslint-disable-line

// Version identifier (also shown in HTML as #version)
const APP_VERSION = '0.2.4';
console.log(`🚀 AIDU AI LLM Web v${APP_VERSION} loaded`);

type Message = {
  role: "user" | "assistant";
  content: string;
};

// Configure marked for markdown parsing
marked.setOptions({
  breaks: true,
  gfm: true,
});

/**
 * Render markdown with LaTeX math support.
 * Protects math expressions before markdown processing, then renders them with KaTeX.
 */
function renderMarkdownWithMath(text: string): string {
  const mathPlaceholders: { [key: string]: string } = {};
  let mathCounter = 0;

  // Protect display math: $$...$$ and \[...\]
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
    const placeholder = `XMATHXDISPLAY${mathCounter}XMATHX`;
    mathPlaceholders[placeholder] = math.trim();
    mathCounter++;
    return placeholder;
  });

  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (match, math) => {
    const placeholder = `XMATHXDISPLAY${mathCounter}XMATHX`;
    mathPlaceholders[placeholder] = math.trim();
    mathCounter++;
    return placeholder;
  });

  // Protect inline math: \(...\) and $...$
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, (match, math) => {
    const placeholder = `XMATHXINLINE${mathCounter}XMATHX`;
    mathPlaceholders[placeholder] = math.trim();
    mathCounter++;
    return placeholder;
  });

  // Protect single $ math (but not within code blocks)
  text = text.replace(/(?<!\$)\$([^\$\n]+)\$(?!\$)/g, (match, math) => {
    const placeholder = `XMATHXINLINE${mathCounter}XMATHX`;
    mathPlaceholders[placeholder] = math.trim();
    mathCounter++;
    return placeholder;
  });

  // Parse markdown
  let html = marked(text) as string;

  // Sanitize HTML to prevent XSS (before restoring math to preserve KaTeX output)
  html = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                   'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'span', 'div', 'a', 'svg', 'path', 'g'],
    ALLOWED_ATTR: ['class', 'style', 'href', 'viewBox', 'width', 'height', 'd', 'transform', 'fill', 'stroke'],
  });

  // Restore and render math with KaTeX (after sanitization to avoid stripping SVG)
  for (const [placeholder, math] of Object.entries(mathPlaceholders)) {
    try {
      const isDisplay = placeholder.includes('DISPLAY');
      const rendered = katex.renderToString(math, {
        displayMode: isDisplay,
        throwOnError: false,
        strict: false,
      });
      const mathHtml = isDisplay 
        ? `<div class="math-display">${rendered}</div>`
        : `<span class="math-inline">${rendered}</span>`;
      html = html.replaceAll(placeholder, mathHtml);
    } catch (e) {
      console.error('KaTeX render error:', e);
      html = html.replaceAll(placeholder, `<code>${math}</code>`);
    }
  }

  return html;
}

// Grab references to the HTML elements
const messagesEl = document.getElementById("messages")!;
const form = document.getElementById("form") as HTMLFormElement;
const input = document.getElementById("input") as HTMLInputElement;
const clearBtn = document.getElementById("clearBtn") as HTMLButtonElement;

// Restore chat history and session from localStorage
let messages: Message[] = JSON.parse(localStorage.getItem("chat") || "[]");
let sessionId = localStorage.getItem("chat_session_id") || "";

function save() {
  localStorage.setItem("chat", JSON.stringify(messages));
  localStorage.setItem("chat_session_id", sessionId);
}

function scrollToBottom() {
  window.scrollTo(0, document.body.scrollHeight);
}

function render() {
  messagesEl.innerHTML = messages
    .map(
      (m) => `
      <div class="msg ${m.role}">
        ${renderMarkdownWithMath(m.content)}
      </div>
    `
    )
    .join("");

  scrollToBottom();
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
      // If session is invalid (404), clear it and start fresh
      if (res.status === 404) {
        console.warn("Session invalid, clearing cache and starting fresh...");
        localStorage.removeItem("chat_session_id");
        localStorage.removeItem("chat");
        sessionId = "";
        messages = [];
        throw new Error("Session expired. Please refresh and try again.");
      }
      throw new Error(`Chat request failed (${res.status})`);
    }

    const data = await res.json();

    // 3) Append assistant reply returned by the API.
    // (Backend already converts SymPy notation to LaTeX)
    console.log('Backend response:', data.reply);
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

// Handle form submission
form.onsubmit = (e) => {
  e.preventDefault();
  const value = input.value.trim();
  if (!value) return;

  input.value = "";
  sendMessage(value);
};

// Handle clear chat button
clearBtn.onclick = () => {
  if (confirm("Clear all messages and start a new session?")) {
    localStorage.removeItem("chat_session_id");
    localStorage.removeItem("chat");
    sessionId = "";
    messages = [];
    render();
    input.focus();
  }
};

// Initial render
render();
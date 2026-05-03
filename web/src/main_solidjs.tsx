/*
 * Copyright (c) 2026 Wolfgang Spahn, PHBern
 * Licensed under the MIT License.
 * Please follow standard academic practice when using this software in research or publications.
 * See ../LICENSE for the full text.
 *
 * Description:
 * SolidJS version of main.ts — same chat UI logic ported to reactive signals and JSX components.
 * Stores chat history and session id in localStorage, renders messages with markdown and LaTeX,
 * creates backend session on first use, and sends prompts to the chat API.
 *
 * Required packages (add to package.json before use):
 *   npm install solid-js
 *   npm install -D vite-plugin-solid
 *
 * In vite.config.ts, add:
 *   import solidPlugin from 'vite-plugin-solid';
 *   plugins: [solidPlugin()],
 *
 * To activate: replace main.ts with this file as the entry point in index.html:
 *   <script type="module" src="/src/main_solidjs.tsx"></script>
 */

import { createSignal, createEffect, For, Show } from 'solid-js';
import { render } from 'solid-js/web';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import katex from 'katex';
// @ts-ignore
import 'katex/dist/katex.min.css';

const APP_VERSION = '0.2.4';
console.log(`🚀 AIDU AI LLM Web (SolidJS) v${APP_VERSION} loaded`);

const ACTOR_ID = "MathTutor";

// A chat message as displayed in the UI.
type Message = {
  role: "user" | "assistant";
  content: string;
};

// ---------------------------------------------------------------------------
// Markdown + LaTeX renderer
// ---------------------------------------------------------------------------

marked.setOptions({ breaks: true, gfm: true });

// Converts plain text that may contain Markdown + LaTeX into safe HTML.
function renderMarkdownWithMath(text: string): string {
  // Map temporary placeholders -> original math expression.
  const mathPlaceholders: { [key: string]: string } = {};
  let mathCounter = 0;

  // Finds regex defined math syntax and temporarily replaces it with a marker,
  // so Markdown processing does not accidentally alter the math content.
  const protectMath = (regex: RegExp, kind: 'DISPLAY' | 'INLINE') => {
    text = text.replace(regex, (_, math) => {
      const placeholder = `XMATHX${kind}${mathCounter}XMATHX`;
      mathPlaceholders[placeholder] = math.trim();
      mathCounter++;
      return placeholder;
    });
  };

  // All math syntaxes we support, in matching order.
  const mathPatterns: Array<{ regex: RegExp; kind: 'DISPLAY' | 'INLINE' }> = [
    // Matches display math written as $$ ... $$ (multiline allowed).
    { regex: /\$\$([\s\S]*?)\$\$/g,          kind: 'DISPLAY' },
    // Matches display math written as \[ ... \] (multiline allowed).
    { regex: /\\\[([\s\S]*?)\\\]/g,          kind: 'DISPLAY' },
    // Matches inline math written as \( ... \) (multiline allowed).
    { regex: /\\\(([\s\S]*?)\\\)/g,          kind: 'INLINE' },
    // Matches inline math written as $ ... $, while avoiding $$ ... $$ blocks.
    // (?<!\$) and (?!\$) are lookaround checks to ensure the $ is single.
    // [^\$\n]+ keeps inline math on one line and stops at the next $.
    { regex: /(?<!\$)\$([^\$\n]+)\$(?!\$)/g, kind: 'INLINE' },
  ];

  // Step 1: protect math before Markdown conversion.
  for (const { regex, kind } of mathPatterns) {
    protectMath(regex, kind);
  }

  // Step 2: convert Markdown to HTML.
  let html = marked(text) as string;

  // Step 3: sanitize generated HTML to reduce XSS risk.
  html = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                   'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'span', 'div', 'a', 'svg', 'path', 'g'],
    ALLOWED_ATTR: ['class', 'style', 'href', 'viewBox', 'width', 'height', 'd', 'transform', 'fill', 'stroke'],
  });

  for (const [placeholder, math] of Object.entries(mathPlaceholders)) {
    try {
      const isDisplay = placeholder.includes('DISPLAY');
      // Step 4: render math safely with KaTeX and insert it back into HTML.
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
      // Fallback: show raw math as code if KaTeX fails on this expression.
      console.error('KaTeX render error:', e);
      html = html.replaceAll(placeholder, `<code>${math}</code>`);
    }
  }

  return html;
}

// ---------------------------------------------------------------------------
// App component
// ---------------------------------------------------------------------------

function App() {
  // Reactive state: each getter/setter pair controls one UI concern.
  const [messages, setMessages]     = createSignal<Message[]>( JSON.parse(localStorage.getItem("chat") || "[]") );
  const [sessionId, setSessionId]   = createSignal<string>( localStorage.getItem("chat_session_id") || "" );
  const [loading, setLoading]       = createSignal(false);
  const [inputValue, setInputValue] = createSignal("");

  // Persist message history whenever it changes.
  createEffect(() => { localStorage.setItem("chat", JSON.stringify(messages())); });
  // Persist session ID whenever it changes.
  createEffect(() => { localStorage.setItem("chat_session_id", sessionId()); });

  // Auto-scroll after new messages or loading state changes.
  createEffect(() => {
    messages(); // Track dependency.
    loading();
    queueMicrotask(() => window.scrollTo(0, document.body.scrollHeight));
  });

  async function sendMessage(content: string) {
    // Optimistic UI update: show user message immediately.
    setMessages(prev => [...prev, { role: "user", content }]);
    setLoading(true);

    try {
      let sid = sessionId();
      if (!sid) {
        // First message in this browser session: ask backend for a new session ID.
        const createRes = await fetch("/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });
        if (!createRes.ok) throw new Error(`Failed to create session (${createRes.status})`);
        const createData = await createRes.json();
        sid = createData.session_id;
        setSessionId(sid);
      }

      // Send user prompt to actor-specific chat endpoint.
      const res = await fetch(`/sessions/${encodeURIComponent(sid)}/${ACTOR_ID}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content }),
      });

      if (!res.ok) {
        if (res.status === 404) {
          // Local session may be stale after backend restart; clear local cache.
          console.warn("Session invalid, clearing cache and starting fresh...");
          localStorage.removeItem("chat_session_id");
          localStorage.removeItem("chat");
          setSessionId("");
          setMessages([]);
          throw new Error("Session expired. Please refresh and try again.");
        }
        throw new Error(`Chat request failed (${res.status})`);
      }

      const data = await res.json();
      console.log('Backend response:', data.reply);
      // Append assistant reply to conversation.
      setMessages(prev => [...prev, { role: "assistant", content: data.reply }]);
    } catch (err) {
      // User-friendly fallback when request fails.
      setMessages(prev => [...prev, { role: "assistant", content: "Error contacting server" }]);
      console.error(err);
    } finally {
      // Always stop loading spinner.
      setLoading(false);
    }
  }

  // Form handler for Enter/Send.
  function handleSubmit(e: Event) {
    e.preventDefault();
    const value = inputValue().trim();
    if (!value) return;
    setInputValue("");
    sendMessage(value);
  }

  // Clears local chat history and session id after confirmation.
  function handleClear() {
    if (confirm("Clear all messages and start a new session?")) {
      localStorage.removeItem("chat_session_id");
      localStorage.removeItem("chat");
      setSessionId("");
      setMessages([]);
    }
  }

  return (
    <>
      {/* Conversation timeline */}
      <div id="messages">
        <For each={messages()}>
          {/* Render each message as sanitized Markdown + math HTML */}
          {(msg) => (
            <div
              class={`msg ${msg.role}`}
              innerHTML={renderMarkdownWithMath(msg.content)}
            />
          )}
        </For>
        {/* Temporary assistant bubble while waiting for network response */}
        <Show when={loading()}>
          <div class="msg assistant loading">Thinking...</div>
        </Show>
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit}>
        <input
          id="input"
          type="text"
          placeholder="Type a message…"
          autocomplete="off"
          value={inputValue()}
          onInput={(e) => setInputValue(e.currentTarget.value)}
        />
        <button type="submit">Send</button>
        <button type="button" id="clearBtn" onClick={handleClear}>Clear</button>
      </form>
    </>
  );
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

render(() => <App />, document.getElementById("app") ?? document.body);

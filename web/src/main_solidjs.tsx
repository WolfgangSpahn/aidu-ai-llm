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

const APP_VERSION = '0.2.5';
console.log(`🚀 AIDU AI LLM Web (SolidJS) v${APP_VERSION} loaded`);

const ACTOR_ID = "MathTutor";

// A chat message as displayed in the UI.
type Message = {
  role: "user" | "assistant";
  content: string;
  duration?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  cost_usd?: number;
  model?: string;
  timestamp?: number;
};

type ChatResponse = {
  session_id: string;
  reply: string;
  context: {
    trace?: {
      messages?: Array<{
        role?: string;
        content?: string;
        duration?: number;
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
        cost_usd?: number;
        model?: string;
        timestamp?: number;
      }>;
    };
  };
};

function isChatResponse(value: unknown): value is ChatResponse {
  if (!value || typeof value !== 'object') return false;

  const response = value as Record<string, unknown>;
  return (
    typeof response.session_id === 'string' &&
    typeof response.reply === 'string' &&
    !!response.context &&
    typeof response.context === 'object'
  );
}

function normalizeTraceMessages(messages: Array<{ role?: string; content?: string; duration?: number; prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; cost_usd?: number; model?: string; timestamp?: number }> | undefined): Message[] {
  if (!messages) return [];

  return messages
    .filter((msg): msg is { role: "user" | "assistant"; content: string; duration?: number; prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; cost_usd?: number; model?: string; timestamp?: number } => (
      (msg.role === "user" || msg.role === "assistant") &&
      typeof msg.content === "string"
    ))
    .map((msg) => ({
      role: msg.role,
      content: msg.content,
      duration: typeof msg.duration === "number" ? msg.duration : undefined,
      prompt_tokens: typeof msg.prompt_tokens === "number" ? msg.prompt_tokens : undefined,
      completion_tokens: typeof msg.completion_tokens === "number" ? msg.completion_tokens : undefined,
      total_tokens: typeof msg.total_tokens === "number" ? msg.total_tokens : undefined,
      cost_usd: typeof msg.cost_usd === "number" ? msg.cost_usd : undefined,
      model: typeof msg.model === "string" ? msg.model : undefined,
      timestamp: typeof msg.timestamp === "number" ? msg.timestamp : undefined,
    }));
}

function formatDuration(seconds: number | undefined): string | null {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    return null;
  }

  return `${seconds.toFixed(1)}s`;
}

function formatTokens(totalTokens: number | undefined): string | null {
  if (typeof totalTokens !== "number" || !Number.isFinite(totalTokens) || totalTokens <= 0) {
    return null;
  }

  return `${Math.round(totalTokens)} tok`;
}

function formatCost(costUsd: number | undefined): string | null {
  if (typeof costUsd !== "number" || !Number.isFinite(costUsd) || costUsd <= 0) {
    return null;
  }

  return `${(costUsd*100000).toFixed(1)} 10⁻³ cents`;
}

function getMessageDurationLabel(messages: Message[], index: number): string | null {
  const msg = messages[index];
  if (!msg) return null;

  if (msg.role === "user") {
    if (typeof msg.duration === "number") {
      return formatDuration(msg.duration);
    }

    const previous = messages[index - 1];
    if (!previous || typeof previous.timestamp !== "number" || typeof msg.timestamp !== "number") {
      return null;
    }

    return formatDuration(msg.timestamp - previous.timestamp);
  }

  if (msg.role === "assistant") {
    return formatDuration(msg.duration);
  }

  return null;
}

function getMessageMetaLabel(messages: Message[], index: number): string | null {
  const msg = messages[index];
  if (!msg) return null;

  const duration = getMessageDurationLabel(messages, index);
  const tokens = formatTokens(msg.total_tokens);
  const cost = formatCost(msg.cost_usd);
  const parts = [duration, tokens, cost].filter((part): part is string => part !== null);
  return parts.length > 0 ? parts.join(" | ") : null;
}

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
  const [inputStartedAt, setInputStartedAt] = createSignal<number | null>(null);


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

  async function sendMessage(content: string, userDuration: number) {
    // Optimistic UI update: show user message immediately.
      setMessages(prev => [...prev, {
        role: "user",
        content,
        duration: userDuration,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        cost_usd: 0,
        timestamp: Date.now() / 1000,
      }]);
    setLoading(true);

    try {
      let sid = sessionId();
      if (!sid) {
        // First message in this browser session: ask backend for a new session ID.
        const createRes = await fetch("/sessions", { // create session endpoint
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
        body: JSON.stringify({ message: content, duration: userDuration }),
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

      const data: unknown = await res.json();
      console.log('Backend response:', data);

      if (!isChatResponse(data)) {
        throw new Error(`Invalid chat response shape: ${JSON.stringify(data)}`);
      }

      const traceMessages = normalizeTraceMessages(data.context.trace?.messages);
      if (traceMessages.length === 0) {
        throw new Error(`Missing context.trace.messages in chat response: ${JSON.stringify(data.context)}`);
      }

      for (let i = traceMessages.length - 1; i >= 0; i--) {
        const current = traceMessages[i];
        if (!current) continue;

        if (current.role === "user") {
          if (typeof current.duration !== "number" || current.duration <= 0) {
            current.duration = userDuration;
          }
          break;
        }
      }

      setMessages(traceMessages);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Error contacting server";
      setMessages(prev => [...prev, { role: "assistant", content: `Error contacting server: ${message}` }]);
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
    const now = performance.now();
    const started = inputStartedAt();
    const userDuration = started !== null ? Math.max(0, (now - started) / 1000) : 0;
    setInputValue("");
    setInputStartedAt(null);
    sendMessage(value, userDuration);
  }

  // Clears local chat history and session id after confirmation.
  function handleClear() {
    if (confirm("Clear all messages and start a new session?")) {
      localStorage.removeItem("chat_session_id");
      localStorage.removeItem("chat");
      setSessionId("");
      setMessages([]);
      setInputStartedAt(null);
    }
  }

  return (
    <>
      {/* Title with version badge */}
      <h1>
        AIDU (SolidJS)
        <span style={{ "font-size": "0.5em", color: "#888", "margin-left": "8px" }}>
          frontend: v{APP_VERSION}
        </span>
      </h1>

      {/* Conversation timeline */}
      <div id="messages">
        <For each={messages()}>
          {/* Render each message as sanitized Markdown + math HTML */}
          {(msg, index) => (
            <div style={{ display: "flex", "flex-direction": "column", "align-items": msg.role === "user" ? "flex-end" : "flex-start" }}>
              <Show when={getMessageDurationLabel(messages(), index()) !== null}>
                <div style={{ "font-size": "0.75rem", color: "#888", "margin-bottom": "4px" }}>
                  {getMessageMetaLabel(messages(), index())}
                </div>
              </Show>
              <div
                class={`msg ${msg.role}`}
                innerHTML={renderMarkdownWithMath(msg.content)}
              />
            </div>
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
          onInput={(e) => {
            const nextValue = e.currentTarget.value;
            if (nextValue.length > 0 && inputStartedAt() === null) {
              setInputStartedAt(performance.now());
            }
            if (nextValue.length === 0) {
              setInputStartedAt(null);
            }
            setInputValue(nextValue);
          }}
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

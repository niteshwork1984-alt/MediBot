import { useState } from "react";
import ReactMarkdown from "react-markdown";

import { ApiError, askChat, login } from "./api";


const SESSION_KEY = "medibot.session";

// This helper reads the current browser-session identity without persisting it beyond the tab session.
function loadSession() {
  const savedSession = sessionStorage.getItem(SESSION_KEY);
  if (!savedSession) {
    return null;
  }
  try {
    return JSON.parse(savedSession);
  } catch {
    sessionStorage.removeItem(SESSION_KEY);
    return null;
  }
}

// This component presents credentials and creates a browser session through the login API.
function LoginPanel({ onLogin }) {
  const [username, setUsername] = useState("nurse.priya");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // This handler submits credentials once and keeps the returned token inside session storage only.
  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setIsSubmitting(true);
    try {
      const result = await login(username, password);
      onLogin({
        accessToken: result.access_token,
        username: result.username,
        role: result.role,
      });
    } catch (requestError) {
      setError(requestError instanceof ApiError ? requestError.message : "Login failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="login-layout">
      <section className="brand-panel" aria-labelledby="brand-heading">
        <p className="eyebrow">ROLE-AWARE HEALTHCARE ASSISTANT</p>
        <h1 id="brand-heading">MediBot</h1>
        <p>
          Ask trusted questions across approved clinical and operational information.
        </p>
      </section>
      <section className="login-card" aria-labelledby="login-heading">
        <p className="eyebrow">WELCOME BACK</p>
        <h2 id="login-heading">Sign in to continue</h2>
        <p className="muted">Use one of the provisioned demo users.</p>
        <form onSubmit={handleSubmit}>
          <label htmlFor="username">Username</label>
          <input
            id="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            required
          />
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            required
          />
          {error && <p className="error" role="alert">{error}</p>}
          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}

// This component renders trusted source metadata returned by the backend.
function SourceList({ sources }) {
  if (!sources.length) {
    return null;
  }
  return (
    <section className="sources" aria-label="Answer sources">
      <h3>Sources</h3>
      <ul>
        {sources.map((source) => (
          <li key={`${source.source_document}-${source.section_title}-${source.collection}`}>
            <strong>{source.source_document}</strong>
            <span>{source.section_title} · {source.collection}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

// This component provides the authenticated question-and-answer workspace.
function ChatWorkspace({ session, onLogout }) {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // This handler sends only the typed question with the private bearer token held in the browser session.
  async function handleSubmit(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      return;
    }
    setError("");
    setIsSubmitting(true);
    try {
      setResponse(await askChat(trimmedQuestion, session.accessToken));
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 401) {
        onLogout();
        setError("Your session has expired. Please sign in again.");
      } else {
        setError(requestError instanceof ApiError ? requestError.message : "Chat request failed.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="workspace">
      <header className="topbar">
        <div>
          <p className="eyebrow">MEDIBOT</p>
          <h1>Healthcare knowledge assistant</h1>
        </div>
        <div className="identity">
          <span>{session.username}</span>
          <span className="role-chip">{session.role.replaceAll("_", " ")}</span>
          <button className="text-button" type="button" onClick={onLogout}>Sign out</button>
        </div>
      </header>
      <section className="chat-card" aria-labelledby="chat-heading">
        <div className="chat-intro">
          <p className="eyebrow">ASK MEDIBOT</p>
          <h2 id="chat-heading">What would you like to know?</h2>
          <p className="muted">Answers are limited to information available to your assigned role.</p>
        </div>
        <form className="question-form" onSubmit={handleSubmit}>
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="For example: What are MRSA contact precautions?"
            rows="4"
            maxLength="4000"
            required
          />
          <button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Searching approved sources…" : "Ask MediBot"}
          </button>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
        {response && (
          <article className="answer-card" aria-live="polite">
            <div className="answer-heading">
              <h2>Answer</h2>
              <span className="route-chip">{response.retrieval_type.replaceAll("_", " ")}</span>
            </div>
            <div className="markdown-answer"><ReactMarkdown>{response.answer}</ReactMarkdown></div>
            <SourceList sources={response.sources} />
          </article>
        )}
      </section>
    </main>
  );
}

// This root component switches between the unauthenticated and authenticated UI states.
export default function App() {
  const [session, setSession] = useState(loadSession);

  // This handler stores the signed session token only for the current browser session.
  function handleLogin(nextSession) {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(nextSession));
    setSession(nextSession);
  }

  // This handler removes the browser-session token and returns the user to the login screen.
  function handleLogout() {
    sessionStorage.removeItem(SESSION_KEY);
    setSession(null);
  }

  return session
    ? <ChatWorkspace session={session} onLogout={handleLogout} />
    : <LoginPanel onLogin={handleLogin} />;
}

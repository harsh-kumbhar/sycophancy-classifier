import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { createRoot } from "react-dom/client";
import {
  Activity, ArrowDownRight, ArrowLeft, ArrowRight, ArrowUpRight,
  BarChart3, BookOpen, Check, CheckCircle2, ChevronDown, CircleHelp,
  FlaskConical, GitBranch, LoaderCircle, Menu, MessageCircle, MessageSquare,
  Microscope, RotateCcw, Send, ShieldCheck, Sparkles, Terminal, X, Zap,
} from "lucide-react";
import "./style.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

const LABELS = {
  CONSISTENT: { title: "Consistent", color: "green", icon: "●", description: "The assistant maintains its position rather than simply yielding to user pressure." },
  JUSTIFIED_CHANGE: { title: "Justified change", color: "amber", icon: "●", description: "The assistant changes its answer for a supported reason, such as valid new information or a correction." },
  SYCOPHANTIC_SHIFT: { title: "Sycophantic shift", color: "red", icon: "●", description: "The assistant changes its answer to align with unsupported user feedback." },
};

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  return data;
}

function getLabel(result) {
  return result?.prediction || result?.final_prediction || result?.label || "UNKNOWN";
}
function getProbabilities(result) {
  return result?.classifier_probabilities || result?.probabilities || result?.class_probabilities || {};
}
function prettyLabel(label) {
  return (label || "UNKNOWN").replaceAll("_", " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}
function safeString(value) {
  if (value === null || value === undefined || value === "") return "Not available";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function normalizeModelText(value) {
  if (typeof value !== "string") return "";
  // Models often emit LaTeX delimiters literally. Convert them to the delimiters
  // understood by remark-math before Markdown rendering.
  return value
    .replace(/\\\[/g, () => "$$")
    .replace(/\\\]/g, () => "$$")
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");
}

function MarkdownText({ children, className = "" }) {
  const content = normalizeModelText(typeof children === "string" ? children : safeString(children));
  return <div className={`markdown-content ${className}`}><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{content}</ReactMarkdown></div>;
}

function App() {
  const [page, setPage] = useState("intro");
  const [apiStatus, setApiStatus] = useState("checking");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [liveMessages, setLiveMessages] = useState([]);
  const [liveAnalyses, setLiveAnalyses] = useState([]);
  const [activeQuestion, setActiveQuestion] = useState("");
  const [activeAnswer, setActiveAnswer] = useState("");
  const [chatInput, setChatInput] = useState("");
  const chatEndRef = useRef(null);

  const [example, setExample] = useState(null);
  const [referenceResult, setReferenceResult] = useState(null);
  const [liveSimulation, setLiveSimulation] = useState(null);
  const [liveDemoResult, setLiveDemoResult] = useState(null);
  const [demoStep, setDemoStep] = useState("empty");

  useEffect(() => {
    api("/health")
      .then((data) => setApiStatus(data.status === "ok" ? "online" : "offline"))
      .catch(() => setApiStatus("offline"));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [liveMessages, busy]);

  function navigate(nextPage) {
    setError("");
    setPage(nextPage);
  }

  async function sendChatMessage(event) {
    event?.preventDefault();
    const text = chatInput.trim();
    if (!text || busy) return;
    setError("");
    setChatInput("");
    setLiveMessages((old) => [...old, { role: "user", content: text }]);

    try {
      setBusy("chat");
      let isFeedback = false;
      let feedbackDetails = null;

      if (activeQuestion && activeAnswer) {
        feedbackDetails = await api("/feedback/detect", {
          method: "POST",
          body: JSON.stringify({
            message: text,
            previous_assistant_response: activeAnswer,
            original_question: activeQuestion,
          }),
        });
        isFeedback = Boolean(feedbackDetails.is_feedback);
      }

      if (!isFeedback) {
        setActiveQuestion(text);
        setActiveAnswer("");
      }

      const history = [...liveMessages, { role: "user", content: text }]
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map(({ role, content }) => ({ role, content }));

      const generated = await api("/chat/respond", {
        method: "POST",
        body: JSON.stringify({ messages: history }),
      });

      const answer = generated.response || "The model returned an empty response.";
      setLiveMessages((old) => [...old, { role: "assistant", content: answer }]);

      if (isFeedback && activeQuestion && activeAnswer) {
        const result = await api("/analyze", {
          method: "POST",
          body: JSON.stringify({
            question: activeQuestion,
            initial_answer: activeAnswer,
            user_feedback: text,
            revised_answer: answer,
          }),
        });
        result.feedback_type = feedbackDetails?.feedback_type;
        result.feedback_confidence = feedbackDetails?.confidence;
        setLiveAnalyses((old) => [...old, result]);
        setActiveAnswer(answer);
      } else {
        setActiveAnswer(answer);
      }
    } catch (err) {
      setError(err.message || "Something went wrong while processing the message.");
    } finally {
      setBusy("");
    }
  }

  async function loadRandomExample() {
    setError("");
    setBusy("random");
    try {
      const data = await api("/examples/random");
      setExample(data.example);
      setReferenceResult(null);
      setLiveSimulation(null);
      setLiveDemoResult(null);
      setDemoStep("loaded");
      setPage("demo");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function analyzeReference() {
    if (!example) return;
    setError("");
    setBusy("reference");
    try {
      const result = await api("/analyze", {
        method: "POST",
        body: JSON.stringify({
          question: example.question,
          initial_answer: example.initial_answer,
          user_feedback: example.user_feedback,
          revised_answer: example.revised_answer,
        }),
      });
      setReferenceResult(result);
      setDemoStep(liveSimulation ? "simulated" : "reference-analyzed");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function simulateLive() {
    if (!example) return;
    setError("");
    setBusy("simulation");
    try {
      const data = await api("/demo/simulate", {
        method: "POST",
        body: JSON.stringify({ example }),
      });
      setLiveSimulation(data);
      setLiveDemoResult(null);
      setDemoStep("simulated");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function analyzeLiveSimulation() {
    if (!liveSimulation?.conversation) return;
    setError("");
    setBusy("live-analysis");
    try {
      const c = liveSimulation.conversation;
      const result = await api("/analyze", {
        method: "POST",
        body: JSON.stringify({
          question: c.question,
          initial_answer: c.initial_answer,
          user_feedback: c.user_feedback,
          revised_answer: c.revised_answer,
        }),
      });
      setLiveDemoResult(result);
      setDemoStep("complete");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  function clearLiveChat() {
    setLiveMessages([]);
    setLiveAnalyses([]);
    setActiveQuestion("");
    setActiveAnswer("");
    setChatInput("");
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => navigate("hub")} aria-label="Go to home">
          <span className="brand-mark"><GitBranch size={18} /></span>
          <span className="brand-copy"><strong>SYCO<span>CHECK</span></strong><small>LLM BEHAVIOR RESEARCH</small></span>
        </button>
        <nav className="top-nav">
          {page !== "intro" && <button className={page === "hub" ? "nav-link active" : "nav-link"} onClick={() => navigate("hub")}>Workspace</button>}
          {page !== "intro" && <button className={page === "live" ? "nav-link active" : "nav-link"} onClick={() => navigate("live")}>Live chat</button>}
          {page !== "intro" && <button className={page === "demo" || page === "demo-results" ? "nav-link active" : "nav-link"} onClick={() => navigate("demo")}>Demo mode</button>}
          <span className={`api-indicator ${apiStatus}`}><i />{apiStatus === "checking" ? "Connecting" : apiStatus === "online" ? "API online" : "API offline"}</span>
        </nav>
      </header>

      {error && (
        <div className="global-error" role="alert">
          <CircleHelp size={17} /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={16} /></button>
        </div>
      )}

      {page === "intro" && <IntroPage onEnter={() => navigate("hub")} />}
      {page === "hub" && <HubPage onLive={() => navigate("live")} onDemo={loadRandomExample} busy={busy === "random"} />}
      {page === "live" && (
        <LivePage
          messages={liveMessages}
          analyses={liveAnalyses}
          input={chatInput}
          setInput={setChatInput}
          onSend={sendChatMessage}
          busy={busy === "chat"}
          onClear={clearLiveChat}
          chatEndRef={chatEndRef}
          onBack={() => navigate("hub")}
        />
      )}
      {page === "demo" && (
        <DemoPage
          example={example}
          referenceResult={referenceResult}
          liveSimulation={liveSimulation}
          liveDemoResult={liveDemoResult}
          busy={busy}
          onLoadRandom={loadRandomExample}
          onAnalyzeReference={analyzeReference}
          onSimulate={simulateLive}
          onAnalyzeLive={analyzeLiveSimulation}
          onViewResults={() => navigate("demo-results")}
          onBack={() => navigate("hub")}
        />
      )}
      {page === "demo-results" && (
        <DemoResultsPage
          example={example}
          referenceResult={referenceResult}
          liveSimulation={liveSimulation}
          liveDemoResult={liveDemoResult}
          onBack={() => navigate("demo")}
          onNewExample={loadRandomExample}
        />
      )}

      <footer className="footer">
        <span>SYCOCHECK <span className="footer-dot">·</span> Research prototype</span>
        <span>Detecting sycophantic drift in multi-turn LLM interactions</span>
      </footer>
    </div>
  );
}

function IntroPage({ onEnter }) {
  return (
    <main className="intro-page page-wrap">
      <div className="intro-copy">
        <div className="eyebrow"><span className="eyebrow-line" /> RESEARCH PROTOTYPE <span className="eyebrow-chip">3-CLASS DETECTION</span></div>
        <h1>When an AI agrees,<br /><span>does it still believe</span><br />the right answer?</h1>
        <p className="intro-lead">A lightweight framework for detecting sycophantic drift in multi-turn language-model conversations.</p>
        <p className="intro-description">Sycophancy happens when an AI changes its answer to please or agree with a user, rather than because the evidence supports a change. This project analyzes the conversation before and after feedback to distinguish agreement from justified correction.</p>
        <div className="intro-actions">
          <button className="button button-primary button-large" onClick={onEnter}>Explore the classifier <ArrowRight size={17} /></button>
          <span className="intro-note"><ShieldCheck size={15} /> Research-focused · Explainable outputs</span>
        </div>
        <div className="intro-stats">
          <div><strong>3</strong><span>Behavior classes</span></div>
          <div><strong>4</strong><span>Conversation signals</span></div>
          <div><strong>1</strong><span>Unified decision pipeline</span></div>
        </div>
      </div>
      <div className="intro-visual">
        <div className="visual-orbit orbit-one" /><div className="visual-orbit orbit-two" />
        <div className="visual-core"><GitBranch size={34} /><span>BEHAVIOR<br />ANALYSIS</span></div>
        <div className="signal-card signal-top"><span className="signal-icon green"><Check size={15} /></span><div><strong>Consistent</strong><small>Stands by supported answer</small></div></div>
        <div className="signal-card signal-right"><span className="signal-icon amber"><RotateCcw size={15} /></span><div><strong>Justified change</strong><small>Updates with evidence</small></div></div>
        <div className="signal-card signal-bottom"><span className="signal-icon red"><Activity size={15} /></span><div><strong>Sycophantic shift</strong><small>Yields to unsupported feedback</small></div></div>
        <div className="visual-caption">QUESTION <span>→</span> INITIAL ANSWER <span>→</span> FEEDBACK <span>→</span> REVISION</div>
      </div>
      <div className="intro-bottomline"><span><Microscope size={15} /> A research tool — not a general-purpose truth detector</span><span>01 / INTRODUCTION</span></div>
    </main>
  );
}

function HubPage({ onLive, onDemo, busy }) {
  return (
    <main className="page-wrap hub-page">
      <div className="page-heading">
        <div className="eyebrow"><span className="eyebrow-line" /> THE WORKSPACE</div>
        <h1>Choose how to <span>investigate.</span></h1>
        <p>Explore a live conversation or run a controlled comparison against a validated research example.</p>
      </div>
      <div className="hub-grid">
        <button className="mode-card live-mode" onClick={onLive}>
          <div className="mode-card-top"><span className="mode-icon"><MessageCircle size={23} /></span><span className="mode-number">01 / INTERACTIVE</span></div>
          <h2>Live AI Chat</h2>
          <p>Talk to the assistant, challenge an answer, and inspect how its response changes. Analysis updates alongside the conversation.</p>
          <div className="mode-tags"><span>Live LLM</span><span>Instant analysis</span><span>Evidence view</span></div>
          <div className="mode-card-cta">Open live chat <ArrowUpRight size={17} /></div>
        </button>
        <button className="mode-card demo-mode" onClick={onDemo} disabled={busy}>
          <div className="mode-card-top"><span className="mode-icon"><FlaskConical size={23} /></span><span className="mode-number">02 / CONTROLLED TEST</span></div>
          <h2>Demo Mode</h2>
          <p>Load a random accepted example, replay its behavior with a fresh live LLM interaction, and compare the two outcomes.</p>
          <div className="mode-tags"><span>Random sample</span><span>Reference vs live</span><span>Ground truth</span></div>
          <div className="mode-card-cta">{busy ? "Loading example…" : "Start a random demo"} {busy ? <LoaderCircle className="spin" size={17} /> : <ArrowUpRight size={17} />}</div>
        </button>
      </div>
      <section className="pipeline-strip">
        <div className="pipeline-heading"><span className="eyebrow-line" /> THE DETECTION PIPELINE</div>
        <div className="pipeline-steps">
          <PipelineStep number="01" title="DeBERTa" subtitle="Class prediction" />
          <ArrowRight size={16} />
          <PipelineStep number="02" title="Evidence" subtitle="Deterministic signals" />
          <ArrowRight size={16} />
          <PipelineStep number="03" title="Verification" subtitle="LLM assessment" />
          <ArrowRight size={16} />
          <PipelineStep number="04" title="Decision" subtitle="Final behavior class" />
        </div>
      </section>
    </main>
  );
}
function PipelineStep({ number, title, subtitle }) {
  return <div className="pipeline-step"><small>{number}</small><strong>{title}</strong><span>{subtitle}</span></div>;
}

function LivePage({ messages, analyses, input, setInput, onSend, busy, onClear, chatEndRef, onBack }) {
  const latest = analyses[analyses.length - 1];
  return (
    <main className="workspace-page page-wrap">
      <div className="workspace-header">
        <button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Workspace</button>
        <div><div className="eyebrow"><span className="eyebrow-line" /> LIVE ANALYSIS</div><h1>Conversation <span>workspace</span></h1></div>
        <button className="button button-quiet button-small" onClick={onClear}><RotateCcw size={14} /> Clear chat</button>
      </div>
      <div className="split-workspace">
        <section className="chat-panel panel">
          <div className="panel-header"><div className="panel-title"><span className="panel-icon"><MessageSquare size={17} /></span><div><strong>Live AI Chat</strong><small>Ask a question, then challenge the answer</small></div></div><span className="live-pill"><i /> LIVE</span></div>
          <div className="chat-content">
            {messages.length === 0 ? (
              <div className="chat-empty">
                <div className="empty-icon"><MessageCircle size={25} /></div>
                <h3>Start a conversation</h3>
                <p>Ask a question. Then give feedback or challenge the response to trigger behavioral analysis.</p>
                <div className="suggestion-label">TRY A STARTER</div>
                <button className="suggestion" onClick={() => setInput("A baker packs 144 cupcakes into boxes of 12. How many boxes are needed?")}>“A baker packs 144 cupcakes into boxes of 12…” <ArrowUpRight size={14} /></button>
              </div>
            ) : (
              <div className="chat-transcript">
                {messages.map((message, index) => <ChatBubble key={`${index}-${message.role}`} message={message} />)}
                {busy && <div className="typing"><span /><span /><span /> Assistant is thinking…</div>}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>
          <form className="chat-composer" onSubmit={onSend}>
            <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Write a question or respond with feedback…" rows={2} disabled={busy} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(e); } }} />
            <div className="composer-bottom"><span><kbd>Enter</kbd> to send · <kbd>Shift + Enter</kbd> for a new line</span><button className="send-button" type="submit" disabled={!input.trim() || busy} aria-label="Send message">{busy ? <LoaderCircle className="spin" size={17} /> : <Send size={16} />}</button></div>
          </form>
        </section>
        <aside className="analysis-panel panel">
          <div className="panel-header"><div className="panel-title"><span className="panel-icon analysis-icon"><BarChart3 size={17} /></span><div><strong>Behavior analysis</strong><small>Updates after detected feedback</small></div></div><span className="panel-count">{analyses.length} {analyses.length === 1 ? "event" : "events"}</span></div>
          <div className="analysis-content">
            {!latest ? (
              <div className="analysis-empty"><div className="empty-ring"><Activity size={22} /></div><h3>Awaiting a feedback event</h3><p>The analysis will appear here after the assistant responds to feedback. Your conversation stays on the left.</p><div className="empty-checklist"><span><CheckCircle2 size={15} /> Initial answer recorded</span><span><CheckCircle2 size={15} /> User feedback detected</span><span><CheckCircle2 size={15} /> Revised answer analyzed</span></div></div>
            ) : (
              <>
                <ResultHero result={latest} />
                <ProbabilityBars result={latest} />
                <div className="analysis-block"><div className="block-title">Decision reasoning</div><p className="reason-text">{latest.decision_reason || "No decision explanation returned."}</p><div className="source-line"><span>Decision source</span><strong>{latest.decision_source || "Not provided"}</strong></div></div>
                <DetailsBlock title="Evidence extracted" value={latest.evidence} />
                <DetailsBlock title="LLM verification" value={latest.verification} />
                <div className="analysis-footnote"><ShieldCheck size={14} /> The result comes from the Python classification pipeline.</div>
                {analyses.length > 1 && <div className="history-list"><div className="block-title">Previous analyses</div>{analyses.slice(0, -1).map((item, index) => <div className="history-item" key={index}><span className={`status-dot ${LABELS[getLabel(item)]?.color || "neutral"}`} /><span>{prettyLabel(getLabel(item))}</span><small>Event {index + 1}</small></div>)}</div>}
              </>
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}

function ChatBubble({ message }) {
  const assistant = message.role === "assistant";
  return <div className={`chat-row ${assistant ? "assistant" : "user"}`}><div className={`avatar ${assistant ? "avatar-ai" : "avatar-user"}`}>{assistant ? <Sparkles size={15} /> : "YOU"}</div><div className="bubble-wrap"><span className="bubble-label">{assistant ? "ASSISTANT" : "YOU"}</span><div className="chat-bubble">{assistant ? <MarkdownText>{message.content}</MarkdownText> : message.content}</div></div></div>;
}

function ResultHero({ result }) {
  const label = getLabel(result);
  const meta = LABELS[label] || { title: prettyLabel(label), color: "neutral", description: "The pipeline returned a classification." };
  const confidence = Number(result.confidence ?? 0);
  return <div className={`result-hero ${meta.color}`}><div className="result-kicker">LATEST CLASSIFICATION <span>•</span> FINAL DECISION</div><div className="result-label"><span className="result-status" />{meta.title}</div><p>{meta.description}</p><div className="confidence-row"><span>Decision confidence</span><strong>{(confidence * 100).toFixed(2)}%</strong></div><div className="confidence-track"><span style={{ width: `${Math.max(0, Math.min(100, confidence * 100))}%` }} /></div></div>;
}

function ProbabilityBars({ result }) {
  const probabilities = getProbabilities(result);
  return <div className="analysis-block"><div className="block-title">Class probabilities <span className="subtle-label">DEBERTA</span></div>{Object.keys(LABELS).map((label) => { const value = Number(probabilities[label] ?? 0); return <div className="prob-row" key={label}><div className="prob-meta"><span>{prettyLabel(label)}</span><strong>{(value * 100).toFixed(2)}%</strong></div><div className="prob-track"><span className={LABELS[label].color} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div></div>; })}</div>;
}

function DetailsBlock({ title, value }) {
  const [open, setOpen] = useState(false);
  return <div className="details-block"><button className="details-toggle" onClick={() => setOpen(!open)}><span>{title}</span><ChevronDown size={16} className={open ? "rotated" : ""} /></button>{open && <pre className="json-view">{safeString(value)}</pre>}</div>;
}

function ConversationCard({ title, conversation, variant = "reference" }) {
  if (!conversation) return <div className="conversation-card placeholder"><span className="placeholder-icon"><MessageSquare size={19} /></span><strong>{title}</strong><p>This conversation has not been generated yet.</p></div>;
  const turns = [
    ["QUESTION", conversation.question],
    [variant === "live" ? "LIVE INITIAL ANSWER" : "INITIAL ANSWER", conversation.initial_answer],
    ["USER FEEDBACK", conversation.user_feedback],
    [variant === "live" ? "LIVE REVISED ANSWER" : "REVISED ANSWER", conversation.revised_answer],
  ];
  return <div className={`conversation-card ${variant}`}><div className="conversation-card-heading"><span>{title}</span><span className="conversation-type">{variant === "live" ? "FRESH LLM" : "DATASET"}</span></div>{turns.map(([label, content]) => <div className="conversation-turn" key={label}><div className="turn-label">{label}</div><div className="conversation-turn-content"><MarkdownText>{content || "Not provided"}</MarkdownText></div></div>)}</div>;
}

function DemoPage({ example, referenceResult, liveSimulation, liveDemoResult, busy, onLoadRandom, onAnalyzeReference, onSimulate, onAnalyzeLive, onViewResults, onBack }) {
  const [showGroundTruth, setShowGroundTruth] = useState(false);
  const referenceConversation = example ? {
    question: example.question, initial_answer: example.initial_answer,
    user_feedback: example.user_feedback, revised_answer: example.revised_answer,
  } : null;
  const liveConversation = liveSimulation?.conversation;
  return (
    <main className="page-wrap demo-page">
      <div className="workspace-header">
        <button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Workspace</button>
        <div><div className="eyebrow"><span className="eyebrow-line" /> CONTROLLED COMPARISON</div><h1>Demo <span>mode</span></h1></div>
        <button className="button button-primary button-small" onClick={onLoadRandom} disabled={busy !== ""}><RotateCcw size={14} /> {busy === "random" ? "Loading…" : "Load random example"}</button>
      </div>
      {!example ? (
        <div className="large-empty"><FlaskConical size={30} /><h2>Load a random validated example</h2><p>Examples are sampled across all accepted behavior classes without choosing a class first.</p><button className="button button-primary" onClick={onLoadRandom} disabled={Boolean(busy)}>{busy === "random" ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} Load random example</button></div>
      ) : (
        <>
          <div className="demo-meta-row"><span className="sample-chip"><span className="sample-dot" /> Random accepted example</span><span className="demo-meta-note">Ground truth is kept separate from model prediction.</span></div>
          <div className="demo-columns">
            <section className="demo-column"><div className="column-heading"><div><span className="column-index">01</span><h2>Reference conversation</h2></div><span className="column-caption">VALIDATED DATA</span></div>
              <ConversationCard title="Example conversation" conversation={referenceConversation} />
              <div className="action-row"><button className="button button-secondary" onClick={onAnalyzeReference} disabled={Boolean(busy)}>{busy === "reference" ? <LoaderCircle className="spin" size={15} /> : <Activity size={15} />} Analyze reference</button><button className="text-button" onClick={() => setShowGroundTruth(!showGroundTruth)}>{showGroundTruth ? "Hide ground truth" : "Reveal ground truth"}</button></div>
              {showGroundTruth && <div className={`ground-truth ${LABELS[example.target_label]?.color || ""}`}><span>Dataset label</span><strong>{prettyLabel(example.target_label)}</strong></div>}
              {referenceResult && <><ResultSummary result={referenceResult} title="Reference classifier result" /><ReasoningPanel result={referenceResult} title="Why this classification?" /></>}
            </section>
            <section className="demo-column"><div className="column-heading"><div><span className="column-index">02</span><h2>Live simulation</h2></div><span className="column-caption">FRESH LLM</span></div>
              {liveConversation ? <ConversationCard title="Live-generated conversation" conversation={liveConversation} variant="live" /> : <div className="conversation-card placeholder"><span className="placeholder-icon"><Zap size={19} /></span><strong>Simulation not run yet</strong><p>The same question and feedback will be used, but the assistant responses will be generated fresh by the live LLM.</p></div>}
              <div className="action-row"><button className="button button-primary" onClick={onSimulate} disabled={!example || Boolean(busy)}>{busy === "simulation" ? <LoaderCircle className="spin" size={15} /> : <Zap size={15} />} Simulate with live LLM</button>{liveConversation && <button className="button button-secondary" onClick={onAnalyzeLive} disabled={Boolean(busy)}>{busy === "live-analysis" ? <LoaderCircle className="spin" size={15} /> : <Activity size={15} />} Analyze live</button>}</div>
              {liveDemoResult && <><ResultSummary result={liveDemoResult} title="Live classifier result" /><ReasoningPanel result={liveDemoResult} title="Why this classification?" /></>}
            </section>
          </div>
          {(referenceResult || liveDemoResult) && <div className="comparison-panel"><div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> RESULT COMPARISON</div><h2>What did the system detect?</h2></div><span className="section-note">Reference and live are classified independently.</span></div><div className="comparison-grid"><ComparisonCell title="Dataset ground truth" label={example.target_label} /><ComparisonCell title="Reference prediction" label={getLabel(referenceResult)} /><ComparisonCell title="Live prediction" label={getLabel(liveDemoResult)} /></div>{referenceResult && liveDemoResult && <div className={`comparison-verdict ${getLabel(liveDemoResult) === example.target_label ? "match" : "different"}`}>{getLabel(liveDemoResult) === example.target_label ? <CheckCircle2 size={17} /> : <GitBranch size={17} />}<div><strong>{getLabel(liveDemoResult) === example.target_label ? "Live result matches the reference label" : "Live result differs from the reference label"}</strong><p>A difference is a valid experimental outcome; the live model is not forced to reproduce the dataset behavior.</p></div></div>}{referenceResult && liveDemoResult && <button className="button button-primary" onClick={onViewResults}>Open detailed results <ArrowRight size={16} /></button>}</div>}
        </>
      )}
    </main>
  );
}

function ResultSummary({ result, title }) {
  const label = getLabel(result);
  const meta = LABELS[label] || { color: "neutral" };
  return <div className={`result-summary ${meta.color}`}><span>{title}</span><strong>{prettyLabel(label)}</strong><small>Confidence · {(Number(result.confidence ?? 0) * 100).toFixed(2)}%</small></div>;
}

function ReasoningPanel({ result, title = "Why this classification?" }) {
  const evidence = result?.evidence || result?.extracted_evidence || result?.analysis?.evidence || {};
  const verification = result?.verification || result?.llm_verification || result?.analysis?.verification || {};
  const reason = result?.decision_reason || result?.reason || result?.explanation || result?.decision?.reason;
  const label = getLabel(result);
  const checks = [];
  const addCheck = (condition, text) => { if (condition !== undefined && condition !== null) checks.push({ ok: Boolean(condition), text }); };
  addCheck(evidence.answer_changed, evidence.answer_changed ? "The revised answer differs from the initial answer." : "The assistant kept the same answer.");
  addCheck(verification.initial_answer_supported, verification.initial_answer_supported ? "The verifier considers the initial answer supported." : "The verifier does not consider the initial answer supported.");
  if (verification.feedback_validity) checks.push({ ok: String(verification.feedback_validity).toUpperCase() === "VALID", text: `User feedback was assessed as ${String(verification.feedback_validity).toLowerCase()}.` });
  addCheck(verification.user_correction_supported, verification.user_correction_supported ? "The user's correction is supported." : "The user's correction is not supported.");
  addCheck(verification.revised_answer_supported, verification.revised_answer_supported ? "The revised answer is supported." : "The revised answer is not supported.");
  addCheck(verification.revision_follows_feedback, verification.revision_follows_feedback ? "The revision follows the user's feedback." : "The revision does not appear to follow the user's feedback.");
  if (!checks.length) {
    addCheck(evidence.contains_new_information, evidence.contains_new_information ? "Evidence extraction detected new information." : "Evidence extraction did not detect new information.");
    addCheck(evidence.contains_correction, evidence.contains_correction ? "Evidence extraction detected a correction." : "Evidence extraction did not detect a correction.");
  }
  const meta = LABELS[label] || { color: "neutral" };
  return <section className={`reasoning-panel ${meta.color}`}><div className="reasoning-heading"><span className="reasoning-icon"><Microscope size={16} /></span><div><strong>{title}</strong><small>Evidence extraction + verifier output</small></div></div>{reason && <MarkdownText className="reasoning-summary">{reason}</MarkdownText>}{checks.length > 0 ? <ul className="reasoning-checks">{checks.map((item, i) => <li key={i}><span className={item.ok ? "check-mark yes" : "check-mark no"}>{item.ok ? "✓" : "!"}</span><span>{item.text}</span></li>)}</ul> : <p className="reasoning-empty">The backend did not return structured evidence for this result. Expand the evidence and verification sections for the raw fields.</p>}<div className="reasoning-footer"><span>Final class</span><strong>{prettyLabel(label)}</strong></div></section>;
}

function ComparisonCell({ title, label }) {
  const meta = LABELS[label] || { color: "neutral" };
  return <div className="comparison-cell"><span>{title}</span><strong className={meta.color}><i />{prettyLabel(label)}</strong></div>;
}

function DemoResultsPage({ example, referenceResult, liveSimulation, liveDemoResult, onBack, onNewExample }) {
  if (!example) return <main className="page-wrap"><div className="large-empty"><h2>No demo result yet</h2><p>Load a random example and run the demo first.</p><button className="button button-primary" onClick={onBack}>Back to demo</button></div></main>;
  const referenceLabel = getLabel(referenceResult);
  const liveLabel = getLabel(liveDemoResult);
  return (
    <main className="page-wrap results-page">
      <div className="workspace-header"><button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Back to demo</button><div><div className="eyebrow"><span className="eyebrow-line" /> EXPERIMENT SUMMARY</div><h1>Demo <span>results</span></h1></div><button className="button button-secondary button-small" onClick={onNewExample}><RotateCcw size={14} /> New example</button></div>
      <div className="results-intro"><div className="results-icon"><BarChart3 size={22} /></div><div><h2>Reference vs live simulation</h2><p>The reference comes from the validated dataset. The live conversation is newly generated. Their outcomes are compared, not forced to match.</p></div></div>
      <div className="results-overview">
        <ResultSummary result={referenceResult || { prediction: "NOT ANALYZED", confidence: 0 }} title="Reference prediction" />
        <ResultSummary result={liveDemoResult || { prediction: "NOT ANALYZED", confidence: 0 }} title="Live prediction" />
        <div className={`result-summary ${referenceResult && liveDemoResult && referenceLabel === liveLabel ? "green" : "neutral"}`}><span>Prediction agreement</span><strong>{referenceResult && liveDemoResult ? (referenceLabel === liveLabel ? "Same class" : "Different classes") : "Pending"}</strong><small>Classifier outputs compared</small></div>
      </div>
      <div className="results-two-col">
        <section className="panel results-panel"><div className="panel-header"><div className="panel-title"><span className="panel-icon"><BookOpen size={17} /></span><div><strong>Reference analysis</strong><small>Validated dataset interaction</small></div></div></div>{referenceResult ? <><ResultHero result={referenceResult} /><ReasoningPanel result={referenceResult} /><ProbabilityBars result={referenceResult} /><DetailsBlock title="Evidence extracted" value={referenceResult.evidence} /><DetailsBlock title="LLM verification" value={referenceResult.verification} /></> : <p className="muted-pad">Analyze the reference conversation on the Demo Mode page.</p>}</section>
        <section className="panel results-panel"><div className="panel-header"><div className="panel-title"><span className="panel-icon"><Zap size={17} /></span><div><strong>Live simulation analysis</strong><small>Fresh LLM responses</small></div></div></div>{liveDemoResult ? <><ResultHero result={liveDemoResult} /><ReasoningPanel result={liveDemoResult} /><ProbabilityBars result={liveDemoResult} /><DetailsBlock title="Evidence extracted" value={liveDemoResult.evidence} /><DetailsBlock title="LLM verification" value={liveDemoResult.verification} /></> : <p className="muted-pad">Simulate and analyze the live conversation on the Demo Mode page.</p>}</section>
      </div>
      <div className="method-note"><ShieldCheck size={16} /><div><strong>How to read this result</strong><p>The dataset label is the reference target. Model predictions are produced by the existing DeBERTa, evidence extraction, verification, and decision pipeline. A live mismatch can reveal that the live LLM did not reproduce the reference behavior.</p></div></div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);

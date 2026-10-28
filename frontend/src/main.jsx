import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { createRoot } from "react-dom/client";
import {
  Activity, ArrowLeft, ArrowRight, ArrowUpRight,
  BarChart3, BookOpen, Check, CheckCircle2, ChevronDown, CircleHelp,
  FlaskConical, GitBranch, LoaderCircle, MessageCircle, MessageSquare,
  Microscope, RotateCcw, Send, ShieldCheck, Sparkles, X, Zap,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

/* -------------------------------------------------------------------------- */
/*  Styles (embedded so the whole frontend lives in this single file)          */
/* -------------------------------------------------------------------------- */
const STYLES = `
@import url("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque&display=swap");
:root {
  --font: "Bricolage Grotesque", Georgia, "Times New Roman", serif;
  --mono: "Bricolage Grotesque", Georgia, "Times New Roman", serif;
  --bg: #0b0f17;
  --surface: #111826;
  --surface-2: #151e2e;
  --surface-3: #1a2538;
  --border: rgba(148, 163, 184, .18);
  --border-strong: rgba(148, 163, 184, .3);
  --text: #e8edf5;
  --text-2: #c3cddc;
  --muted: #94a1b6;
  --muted-2: #6d7b91;
  --accent: #6ea8fe;
  --accent-strong: #4c8df6;
  --green: #52d1a0;
  --blue: #6ea8fe;
  --red: #f4788b;
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px;
  --space-5: 24px; --space-6: 32px; --space-7: 48px; --space-8: 72px;
  --radius-sm: 8px; --radius: 12px; --radius-lg: 16px;
  --shadow: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.22);
  --ease: .18s ease;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { min-width: 320px; background: var(--bg); font-family: var(--font); }
body { margin: 0; min-width: 320px; min-height: 100vh; font-family: var(--font); font-size: 16px; line-height: 1.55; color: var(--text); background: var(--bg); text-rendering: optimizeLegibility; -webkit-font-smoothing: antialiased; }
button, textarea { font: inherit; color: inherit; }
button { cursor: pointer; }
button:disabled { cursor: not-allowed; opacity: .55; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
h1, h2, h3, p { margin: 0; }
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes blink { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { * { animation-duration: .01ms !important; transition-duration: .01ms !important; } }

/* Tone variables: one place that defines each classification's colour treatment */
.tone-green { --tone: var(--green); --tone-bg: rgba(82,209,160,.09); --tone-border: rgba(82,209,160,.32); }
.tone-blue { --tone: var(--blue); --tone-bg: rgba(110,168,254,.09); --tone-border: rgba(110,168,254,.32); }
.tone-red { --tone: var(--red); --tone-bg: rgba(244,120,139,.09); --tone-border: rgba(244,120,139,.34); }
.tone-neutral { --tone: var(--muted); --tone-bg: rgba(148,163,184,.07); --tone-border: var(--border-strong); }

.app-shell { min-height: 100vh; display: flex; flex-direction: column; }
.page-wrap { width: 100%; max-width: 1280px; margin: 0 auto; padding: var(--space-6) clamp(16px, 4vw, 48px) var(--space-8); flex: 1; animation: rise .25s ease both; }

/* Top bar */
.topbar { height: 68px; padding: 0 clamp(16px, 4vw, 48px); display: flex; align-items: center; justify-content: space-between; gap: var(--space-4); border-bottom: 1px solid var(--border); background: rgba(11,15,23,.92); position: sticky; top: 0; z-index: 20; backdrop-filter: blur(12px); }
.brand { display: flex; gap: var(--space-3); align-items: center; border: 0; background: transparent; padding: 0; text-align: left; }
.brand-mark { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 10px; color: var(--accent); background: var(--surface-3); border: 1px solid var(--border-strong); }
.brand-copy { display: flex; flex-direction: column; line-height: 1.2; }
.brand-copy strong { font-size: 16px; font-weight: 700; letter-spacing: .08em; }
.brand-copy strong span { color: var(--accent); }
.brand-copy small { color: var(--muted-2); font-size: 11px; font-weight: 600; letter-spacing: .12em; }
.top-nav { display: flex; align-items: center; gap: var(--space-5); }
.nav-link { border: 0; background: transparent; color: var(--muted); padding: 22px 0 20px; font-size: 15px; font-weight: 600; border-bottom: 2px solid transparent; transition: color var(--ease), border-color var(--ease); }
.nav-link:hover { color: var(--text); }
.nav-link.active { color: var(--text); border-bottom-color: var(--accent); }
.api-indicator { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: var(--muted); padding: 6px 12px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface); white-space: nowrap; }
.api-indicator i { width: 8px; height: 8px; border-radius: 50%; background: var(--muted-2); }
.api-indicator.online i { background: var(--green); }
.api-indicator.offline i { background: var(--red); }
.api-indicator.checking i { animation: blink 1.2s infinite; }

/* Error banner */
.global-error { display: flex; align-items: flex-start; gap: var(--space-3); margin: var(--space-4) auto 0; width: calc(100% - 2 * clamp(16px, 4vw, 48px)); max-width: 1184px; padding: 14px 16px; border: 1px solid rgba(244,120,139,.4); background: rgba(244,120,139,.08); border-radius: var(--radius); color: #ffd3da; font-size: 15px; }
.global-error svg { flex: none; margin-top: 2px; color: var(--red); }
.global-error span { flex: 1; }
.global-error button { border: 0; background: transparent; color: inherit; padding: 2px; border-radius: 6px; display: grid; }
.global-error button:hover { background: rgba(255,255,255,.08); }

/* Buttons */
.button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; border-radius: 10px; border: 1px solid transparent; padding: 11px 18px; font-size: 15px; font-weight: 600; line-height: 1.2; transition: background var(--ease), border-color var(--ease), transform var(--ease); }
.button:active:not(:disabled) { transform: translateY(1px); }
.button-primary { background: var(--accent-strong); color: #fff; }
.button-primary:hover:not(:disabled) { background: #5c9aff; }
.button-secondary { background: var(--surface-3); border-color: var(--border-strong); color: var(--text); }
.button-secondary:hover:not(:disabled) { background: #212e45; border-color: rgba(148,163,184,.45); }
.button-quiet { background: transparent; border-color: var(--border); color: var(--muted); }
.button-quiet:hover:not(:disabled) { color: var(--text); border-color: var(--border-strong); background: var(--surface); }
.button-small { padding: 8px 14px; font-size: 14px; }
.button-large { padding: 14px 24px; font-size: 16px; }
.text-button { border: 0; background: transparent; color: var(--accent); font-size: 14px; font-weight: 600; padding: 8px 4px; }
.text-button:hover { text-decoration: underline; }

/* Headings */
.eyebrow { display: inline-flex; align-items: center; gap: 10px; color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: var(--space-2); }
.eyebrow-line { width: 20px; height: 2px; background: var(--accent); border-radius: 2px; display: inline-block; }
.eyebrow-chip { padding: 2px 8px; border: 1px solid var(--border-strong); border-radius: 999px; color: var(--muted); letter-spacing: .08em; }
h1 { font-size: clamp(28px, 4vw, 40px); line-height: 1.15; font-weight: 700; letter-spacing: -.015em; }
h1 span { color: var(--accent); }
.page-heading { margin-bottom: var(--space-6); max-width: 680px; }
.page-heading p { margin-top: var(--space-3); color: var(--muted); font-size: 18px; }
.workspace-header { display: grid; grid-template-columns: 1fr auto; grid-template-areas: "back actions" "title title"; gap: var(--space-4); align-items: center; margin-bottom: var(--space-5); }
.workspace-header > .back-link { grid-area: back; justify-self: start; }
.workspace-header > div { grid-area: title; }
.workspace-header > .button { grid-area: actions; }
.back-link { display: inline-flex; align-items: center; gap: 6px; border: 0; background: transparent; color: var(--muted); font-size: 14px; font-weight: 600; padding: 6px 0; transition: color var(--ease); }
.back-link:hover { color: var(--text); }

/* Intro */
.intro-page { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr); gap: var(--space-7); align-items: center; padding-top: var(--space-7); }
.intro-copy h1 { font-size: clamp(34px, 5vw, 56px); }
.intro-lead { margin-top: var(--space-5); font-size: 20px; color: var(--text-2); max-width: 560px; }
.intro-description { margin-top: var(--space-3); color: var(--muted); font-size: 17px; max-width: 580px; }
.intro-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-4); margin-top: var(--space-6); }
.intro-note { display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 14px; }
.intro-stats { display: flex; flex-wrap: wrap; gap: var(--space-6); margin-top: var(--space-7); padding-top: var(--space-5); border-top: 1px solid var(--border); }
.intro-stats strong { display: block; font-size: 30px; line-height: 1; }
.intro-stats span { color: var(--muted); font-size: 14px; }
.intro-visual { position: relative; min-height: 420px; display: grid; place-items: center; }
.visual-orbit { position: absolute; border: 1px dashed var(--border-strong); border-radius: 50%; }
.orbit-one { width: 300px; height: 300px; }
.orbit-two { width: 420px; height: 420px; opacity: .5; }
.visual-core { position: relative; width: 128px; height: 128px; border-radius: 50%; display: grid; place-items: center; align-content: center; gap: 6px; text-align: center; background: var(--surface-2); border: 1px solid var(--border-strong); color: var(--accent); box-shadow: var(--shadow); }
.visual-core span { font-size: 11px; font-weight: 700; letter-spacing: .12em; color: var(--text-2); line-height: 1.3; }
.signal-card { position: absolute; display: flex; align-items: center; gap: var(--space-3); padding: 12px 16px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); box-shadow: var(--shadow); transition: transform var(--ease), border-color var(--ease); }
.signal-card:hover { transform: translateY(-2px); border-color: var(--border-strong); }
.signal-card strong { display: block; font-size: 15px; }
.signal-card small { color: var(--muted); font-size: 13px; }
.signal-top { top: 4%; left: 50%; margin-left: -110px; }
.signal-right { right: 0; top: 46%; }
.signal-bottom { bottom: 6%; left: 2%; }
.signal-icon { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; flex: none; }
.signal-icon.green { color: var(--green); background: rgba(82,209,160,.12); }
.signal-icon.amber, .signal-icon.blue { color: var(--blue); background: rgba(110,168,254,.12); }
.signal-icon.red { color: var(--red); background: rgba(244,120,139,.12); }
.visual-caption { position: absolute; bottom: -4px; left: 0; right: 0; text-align: center; font-size: 11px; font-weight: 700; letter-spacing: .1em; color: var(--muted-2); }
.visual-caption span { color: var(--accent); margin: 0 4px; }
.intro-bottomline { grid-column: 1 / -1; display: flex; justify-content: space-between; flex-wrap: wrap; gap: var(--space-3); padding-top: var(--space-4); border-top: 1px solid var(--border); color: var(--muted-2); font-size: 13px; }
.intro-bottomline span { display: inline-flex; align-items: center; gap: 8px; }

/* Hub */
.hub-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-5); }
.mode-card { text-align: left; display: flex; flex-direction: column; gap: var(--space-3); padding: var(--space-5) var(--space-5) var(--space-4); border-radius: var(--radius-lg); background: var(--surface); border: 1px solid var(--border); transition: border-color var(--ease), transform var(--ease), background var(--ease); }
.mode-card:hover:not(:disabled) { border-color: var(--accent); background: var(--surface-2); transform: translateY(-2px); }
.mode-card-top { display: flex; justify-content: space-between; align-items: center; }
.mode-icon { width: 46px; height: 46px; display: grid; place-items: center; border-radius: 12px; background: var(--surface-3); color: var(--accent); border: 1px solid var(--border); }
.mode-number { color: var(--muted-2); font-size: 12px; font-weight: 700; letter-spacing: .12em; }
.mode-card h2 { font-size: 24px; }
.mode-card p { color: var(--muted); font-size: 16px; }
.mode-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.mode-tags span { padding: 3px 10px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); font-size: 13px; }
.mode-card-cta { margin-top: auto; padding-top: var(--space-3); display: inline-flex; align-items: center; gap: 8px; color: var(--accent); font-weight: 600; font-size: 15px; }
.pipeline-strip { margin-top: var(--space-6); padding: var(--space-5); border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface); }
.pipeline-heading { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .14em; margin-bottom: var(--space-4); }
.pipeline-steps { display: flex; align-items: center; gap: var(--space-4); flex-wrap: wrap; }
.pipeline-steps > svg { color: var(--muted-2); flex: none; }
.pipeline-step { flex: 1; min-width: 140px; padding: 12px 16px; border-radius: var(--radius); background: var(--surface-2); border: 1px solid var(--border); display: flex; flex-direction: column; gap: 2px; }
.pipeline-step small { color: var(--accent); font-weight: 700; font-size: 12px; letter-spacing: .08em; }
.pipeline-step strong { font-size: 16px; }
.pipeline-step span { color: var(--muted); font-size: 13px; }

/* Panels */
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); box-shadow: var(--shadow); overflow: hidden; }
.panel-header { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); padding: var(--space-4) var(--space-5); border-bottom: 1px solid var(--border); }
.panel-title { display: flex; align-items: center; gap: var(--space-3); min-width: 0; }
.panel-title strong { display: block; font-size: 16px; }
.panel-title small { color: var(--muted); font-size: 13px; }
.panel-icon { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center; flex: none; background: var(--surface-3); color: var(--accent); }
.live-pill, .panel-count { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: .06em; border: 1px solid var(--border); color: var(--muted); white-space: nowrap; }
.live-pill i { width: 7px; height: 7px; border-radius: 50%; background: var(--green); }
.split-workspace { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); gap: var(--space-5); align-items: start; }
.chat-panel { display: flex; flex-direction: column; height: min(90vh, 980px); min-height: 680px; }
.analysis-panel { position: static; display: flex; flex-direction: column; }
.analysis-content { padding: var(--space-5); }
.chat-content { flex: 1; overflow-y: auto; padding: var(--space-5); }
.muted-pad { padding: var(--space-5); color: var(--muted); }

/* Chat */
.chat-empty, .analysis-empty { text-align: center; padding: var(--space-6) var(--space-4); display: flex; flex-direction: column; align-items: center; gap: var(--space-3); color: var(--muted); }
.chat-empty h3, .analysis-empty h3 { color: var(--text); font-size: 20px; }
.chat-empty p, .analysis-empty p { max-width: 380px; }
.empty-icon, .empty-ring { width: 56px; height: 56px; display: grid; place-items: center; border-radius: 50%; background: var(--surface-3); color: var(--accent); border: 1px solid var(--border); }
.suggestion-label { margin-top: var(--space-3); font-size: 12px; letter-spacing: .12em; font-weight: 700; color: var(--muted-2); }
.suggestion { display: inline-flex; align-items: center; gap: 8px; max-width: 100%; text-align: left; padding: 12px 16px; border-radius: var(--radius); background: var(--surface-2); border: 1px solid var(--border); color: var(--text-2); font-size: 15px; transition: border-color var(--ease), background var(--ease); }
.suggestion:hover { border-color: var(--accent); background: var(--surface-3); }
.empty-checklist { display: flex; flex-direction: column; gap: 8px; margin-top: var(--space-2); text-align: left; font-size: 15px; }
.empty-checklist span { display: inline-flex; align-items: center; gap: 8px; }
.empty-checklist svg { color: var(--muted-2); }
.chat-transcript { display: flex; flex-direction: column; gap: var(--space-4); }
.chat-row { display: flex; gap: var(--space-3); align-items: flex-start; animation: rise .2s ease both; }
.chat-row.user { flex-direction: row-reverse; }
.avatar { width: 34px; height: 34px; flex: none; display: grid; place-items: center; border-radius: 10px; font-size: 10px; font-weight: 700; letter-spacing: .04em; }
.avatar-ai { background: rgba(110,168,254,.14); color: var(--accent); }
.avatar-user { background: var(--surface-3); color: var(--text-2); }
.bubble-wrap { max-width: min(82%, 560px); display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.chat-row.user .bubble-wrap { align-items: flex-end; }
.bubble-label { font-size: 11px; font-weight: 700; letter-spacing: .12em; color: var(--muted-2); }
.chat-bubble { padding: 12px 16px; border-radius: var(--radius); font-size: 16px; line-height: 1.6; overflow-wrap: anywhere; white-space: pre-wrap; }
.chat-bubble .markdown-content { white-space: normal; }
.chat-row.assistant .chat-bubble { background: var(--surface-2); border: 1px solid var(--border); border-top-left-radius: 4px; }
.chat-row.user .chat-bubble { background: rgba(76,141,246,.16); border: 1px solid rgba(110,168,254,.34); border-top-right-radius: 4px; }
.typing { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 14px; padding: 4px 0 4px 46px; }
.typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); animation: blink 1.2s infinite; }
.typing span:nth-child(2) { animation-delay: .15s; }
.typing span:nth-child(3) { animation-delay: .3s; margin-right: 6px; }
.chat-composer { padding: var(--space-4) var(--space-5) var(--space-4); border-top: 1px solid var(--border); background: var(--surface); }
.chat-composer textarea { width: 100%; resize: none; padding: 12px 14px; border-radius: var(--radius); border: 1px solid var(--border-strong); background: var(--bg); font-size: 16px; line-height: 1.5; transition: border-color var(--ease); }
.chat-composer textarea::placeholder { color: var(--muted-2); }
.chat-composer textarea:focus { outline: none; border-color: var(--accent); }
.composer-bottom { display: flex; justify-content: space-between; align-items: center; gap: var(--space-3); margin-top: var(--space-3); color: var(--muted-2); font-size: 13px; }
kbd { font-family: var(--mono); font-size: 12px; padding: 1px 6px; border: 1px solid var(--border-strong); border-radius: 5px; background: var(--surface-2); color: var(--text-2); }
.send-button { width: 42px; height: 42px; display: grid; place-items: center; border-radius: 10px; border: 0; background: var(--accent-strong); color: #fff; transition: background var(--ease); }
.send-button:hover:not(:disabled) { background: #5c9aff; }

/* Result dashboard */
.result-dashboard { display: flex; flex-direction: column; gap: var(--space-5); animation: rise .25s ease both; }
.verdict { border: 1px solid var(--tone-border); background: var(--tone-bg); border-radius: var(--radius-lg); padding: var(--space-5); border-left-width: 4px; }
.verdict-top { display: flex; justify-content: space-between; align-items: center; gap: var(--space-3); flex-wrap: wrap; margin-bottom: var(--space-4); }
.verdict-kicker { font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
.verdict-title { display: flex; align-items: center; gap: var(--space-3); }
.verdict-icon { width: 44px; height: 44px; flex: none; display: grid; place-items: center; border-radius: 12px; color: var(--tone); background: rgba(0,0,0,.22); border: 1px solid var(--tone-border); }
.verdict-title h3 { font-size: clamp(26px, 3.4vw, 34px); line-height: 1.1; color: var(--tone); font-weight: 700; letter-spacing: -.01em; }
.verdict-lead { margin-top: var(--space-4); font-size: 17px; color: var(--text); }
.verdict-why { margin-top: var(--space-3); padding-top: var(--space-3); border-top: 1px solid var(--border); color: var(--text-2); font-size: 15px; }
.verdict-why-label { display: block; font-size: 12px; font-weight: 700; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
.confidence { margin-top: var(--space-5); }
.confidence-row { display: flex; justify-content: space-between; align-items: baseline; gap: var(--space-3); }
.confidence-row span { font-size: 14px; font-weight: 600; color: var(--text-2); }
.confidence-row strong { font-size: 22px; font-variant-numeric: tabular-nums; }
.meter { height: 8px; border-radius: 999px; background: rgba(0,0,0,.3); overflow: hidden; margin-top: 8px; }
.meter > span { display: block; height: 100%; border-radius: inherit; background: var(--tone); transition: width .5s ease; }
.confidence-note { margin-top: 8px; font-size: 13px; color: var(--muted); }

.chip { display: inline-flex; align-items: center; gap: 7px; padding: 4px 11px; border-radius: 999px; font-size: 13px; font-weight: 600; border: 1px solid var(--border-strong); color: var(--text-2); background: rgba(0,0,0,.2); white-space: nowrap; }
.chip i { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.chip.ok i { background: var(--green); }
.chip.warn i { background: #e9b866; }
.chip.warn { border-color: rgba(233,184,102,.4); color: #f0cf99; }

.card { border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface-2); padding: var(--space-5); }
.block-title { display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--text-2); margin-bottom: var(--space-4); }
.subtle-label { color: var(--muted-2); font-size: 11px; letter-spacing: .1em; padding: 2px 8px; border: 1px solid var(--border); border-radius: 999px; }
.prob-row + .prob-row { margin-top: var(--space-4); }
.prob-meta { display: flex; justify-content: space-between; gap: var(--space-3); font-size: 15px; color: var(--text-2); margin-bottom: 6px; }
.prob-meta strong { font-variant-numeric: tabular-nums; color: var(--text); }
.prob-row.top .prob-meta span { color: var(--text); font-weight: 600; }
.prob-track { height: 10px; background: rgba(0,0,0,.32); border-radius: 999px; overflow: hidden; }
.prob-track span { display: block; height: 100%; border-radius: inherit; transition: width .5s ease; opacity: .55; }
.prob-row.top .prob-track span { opacity: 1; }
.prob-track .green { background: var(--green); }
.prob-track .blue { background: var(--blue); }
.prob-track .red { background: var(--red); }
.prob-empty { color: var(--muted); font-size: 15px; }

/* Technical analysis (collapsed by default) */
.tech { border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface-2); overflow: hidden; }
.tech-toggle { width: 100%; display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); padding: 14px var(--space-5); border: 0; background: transparent; text-align: left; transition: background var(--ease); }
.tech-toggle:hover { background: var(--surface-3); }
.tech-toggle-left { display: flex; align-items: center; gap: var(--space-3); flex-wrap: wrap; font-weight: 600; font-size: 15px; }
.tech-toggle-left svg { color: var(--accent); }
.tech-toggle > svg { color: var(--muted); transition: transform var(--ease); flex: none; }
.tech-toggle > svg.rotated { transform: rotate(180deg); }
.tech-body { padding: var(--space-2) var(--space-5) var(--space-5); display: flex; flex-direction: column; gap: var(--space-5); border-top: 1px solid var(--border); }
.tech-section { padding-top: var(--space-4); }
.tech-section h4 { margin: 0 0 var(--space-3); font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
.compare-pair { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); }
.compare-cell { padding: 12px 14px; border-radius: var(--radius-sm); background: rgba(0,0,0,.2); border: 1px solid var(--border); }
.compare-cell span { display: block; font-size: 12px; color: var(--muted); margin-bottom: 2px; }
.compare-cell strong { font-size: 16px; color: var(--tone); }
.tech-note { margin-top: var(--space-3); font-size: 14px; color: var(--muted); }
.signal-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
.signal-list li { display: flex; gap: 10px; align-items: flex-start; font-size: 15px; color: var(--text-2); }
.signal-list .mark { width: 20px; height: 20px; flex: none; margin-top: 1px; border-radius: 50%; display: grid; place-items: center; font-size: 12px; font-weight: 700; }
.mark.yes { background: rgba(82,209,160,.16); color: var(--green); }
.mark.no { background: rgba(233,184,102,.16); color: #e9b866; }
.mark.info { background: rgba(148,163,184,.16); color: var(--muted); }
.notice { display: flex; gap: var(--space-3); padding: 12px 14px; border-radius: var(--radius-sm); font-size: 14px; line-height: 1.5; border: 1px solid var(--border); background: rgba(0,0,0,.2); color: var(--text-2); }
.notice svg { flex: none; margin-top: 2px; }
.notice.ok { border-color: rgba(82,209,160,.35); background: rgba(82,209,160,.07); }
.notice.ok svg { color: var(--green); }
.notice.warn { border-color: rgba(233,184,102,.4); background: rgba(233,184,102,.07); }
.notice.warn svg { color: #e9b866; }
.notice.neutral svg { color: var(--muted); }
.verifier-text { margin-top: var(--space-3); font-size: 15px; color: var(--text-2); }
.details-block { border-top: 1px dashed var(--border); padding-top: var(--space-3); }
.details-toggle { display: inline-flex; align-items: center; gap: 6px; border: 0; background: transparent; color: var(--muted); font-size: 13px; font-weight: 600; padding: 4px 0; }
.details-toggle:hover { color: var(--text); }
.details-toggle svg { transition: transform var(--ease); }
.details-toggle svg.rotated { transform: rotate(180deg); }
.json-view { margin: 8px 0 0; padding: 12px; max-height: 240px; overflow: auto; border-radius: var(--radius-sm); background: var(--bg); border: 1px solid var(--border); font-family: var(--mono); font-size: 12px; line-height: 1.5; color: var(--text-2); white-space: pre-wrap; word-break: break-word; }

.analysis-footnote { display: flex; align-items: center; gap: 8px; color: var(--muted-2); font-size: 13px; }
.history-list { padding-top: var(--space-4); border-top: 1px solid var(--border); }
.history-item { width: 100%; display: flex; align-items: center; gap: 10px; padding: 9px 12px; margin-bottom: 6px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: transparent; font-size: 14px; text-align: left; transition: background var(--ease), border-color var(--ease); }
.history-item:hover { background: var(--surface-3); }
.history-item.selected { border-color: var(--accent); background: var(--surface-3); }
.history-item small { margin-left: auto; color: var(--muted); }
.status-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--muted-2); }
.status-dot.green { background: var(--green); }
.status-dot.blue { background: var(--blue); }
.status-dot.red { background: var(--red); }
.feedback-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: var(--space-4); }

/* Demo */
.demo-meta-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-4); margin-bottom: var(--space-5); }
.sample-chip { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px; border: 1px solid var(--border-strong); font-size: 13px; font-weight: 600; background: var(--surface); }
.sample-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); }
.demo-meta-note { color: var(--muted); font-size: 14px; }
.class-selector { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3) var(--space-4); margin-bottom: var(--space-5); }
.class-selector-label { font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.class-selector-options { display: inline-flex; flex-wrap: wrap; gap: 4px; padding: 4px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
.class-option { border: 1px solid transparent; background: transparent; color: var(--muted); padding: 7px 14px; border-radius: 9px; font-size: 14px; font-weight: 600; transition: background var(--ease), color var(--ease), border-color var(--ease); }
.class-option:hover:not(:disabled) { color: var(--text); background: var(--surface-3); }
.class-option.active { color: var(--text); background: var(--surface-3); border-color: var(--border-strong); }
.demo-columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-5); align-items: start; }
.demo-column { display: flex; flex-direction: column; gap: var(--space-4); min-width: 0; }
.column-heading { display: flex; justify-content: space-between; align-items: center; gap: var(--space-3); }
.column-heading > div { display: flex; align-items: center; gap: var(--space-3); }
.column-heading h2 { font-size: 20px; }
.column-index { width: 30px; height: 30px; display: grid; place-items: center; border-radius: 8px; background: var(--surface-3); color: var(--accent); font-size: 13px; font-weight: 700; }
.column-caption { font-size: 11px; font-weight: 700; letter-spacing: .12em; color: var(--muted-2); }
.action-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3); }
.ground-truth { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-radius: var(--radius); border: 1px solid var(--tone-border); background: var(--tone-bg); }
.ground-truth span { color: var(--muted); font-size: 14px; }
.ground-truth strong { color: var(--tone); }
.large-empty { display: flex; flex-direction: column; align-items: center; text-align: center; gap: var(--space-3); padding: var(--space-8) var(--space-5); border: 1px dashed var(--border-strong); border-radius: var(--radius-lg); background: var(--surface); color: var(--muted); }
.large-empty h2 { color: var(--text); font-size: 24px; }
.large-empty p { max-width: 460px; }
.large-empty svg { color: var(--accent); }
.conversation-card { border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface); overflow: hidden; }
.conversation-card.placeholder { padding: var(--space-6); text-align: center; display: flex; flex-direction: column; align-items: center; gap: var(--space-2); color: var(--muted); border-style: dashed; }
.conversation-card.placeholder strong { color: var(--text); }
.placeholder-icon { width: 40px; height: 40px; display: grid; place-items: center; border-radius: 50%; background: var(--surface-3); color: var(--accent); }
.conversation-card-heading { display: flex; justify-content: space-between; align-items: center; padding: 12px var(--space-5); border-bottom: 1px solid var(--border); font-weight: 600; font-size: 15px; background: var(--surface-2); }
.conversation-type { font-size: 11px; letter-spacing: .12em; color: var(--muted-2); font-weight: 700; }
.conversation-turn { padding: var(--space-4) var(--space-5); border-bottom: 1px solid var(--border); border-left: 3px solid transparent; }
.conversation-turn:last-child { border-bottom: 0; }
.conversation-turn.turn-question { border-left-color: var(--muted-2); }
.conversation-turn.turn-initial { border-left-color: var(--accent); }
.conversation-turn.turn-feedback { border-left-color: #e9b866; background: rgba(233,184,102,.04); }
.conversation-turn.turn-revised { border-left-color: var(--green); }
.turn-label { font-size: 11px; font-weight: 700; letter-spacing: .12em; color: var(--muted); margin-bottom: 6px; }
.turn-feedback .turn-label { color: #e9b866; }
.turn-initial .turn-label { color: var(--accent); }
.turn-revised .turn-label { color: var(--green); }
.conversation-turn-content { font-size: 16px; color: var(--text-2); }

/* Comparison + results */
.comparison-panel { margin-top: var(--space-6); padding: var(--space-5); border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface); display: flex; flex-direction: column; gap: var(--space-4); }
.section-heading { display: flex; justify-content: space-between; align-items: flex-end; gap: var(--space-4); flex-wrap: wrap; }
.section-heading h2 { font-size: 24px; }
.section-note { color: var(--muted); font-size: 14px; }
.comparison-grid, .results-overview { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-4); }
.comparison-cell { padding: 14px 16px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface-2); }
.comparison-cell span { display: block; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
.comparison-cell strong { display: inline-flex; align-items: center; gap: 8px; font-size: 17px; }
.comparison-cell strong i { width: 9px; height: 9px; border-radius: 50%; background: var(--muted-2); }
.comparison-cell strong.green i { background: var(--green); }
.comparison-cell strong.blue i { background: var(--blue); }
.comparison-cell strong.red i { background: var(--red); }
.comparison-verdict { display: flex; gap: var(--space-3); padding: 14px 16px; border-radius: var(--radius); border: 1px solid var(--border); }
.comparison-verdict.match { border-color: rgba(82,209,160,.4); background: rgba(82,209,160,.07); }
.comparison-verdict.different { border-color: rgba(110,168,254,.4); background: rgba(110,168,254,.07); }
.comparison-verdict svg { flex: none; margin-top: 3px; }
.comparison-verdict p { color: var(--muted); font-size: 14px; margin-top: 2px; }
.comparison-panel > .button { align-self: flex-start; }
.result-summary { padding: var(--space-4) var(--space-5); border-radius: var(--radius); border: 1px solid var(--tone-border); background: var(--tone-bg); display: flex; flex-direction: column; gap: 2px; }
.result-summary span { font-size: 13px; color: var(--muted); }
.result-summary strong { font-size: 22px; color: var(--tone); }
.result-summary small { font-size: 13px; color: var(--muted); }
.results-intro { display: flex; gap: var(--space-4); align-items: center; padding: var(--space-5); margin-bottom: var(--space-5); border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface); }
.results-intro h2 { font-size: 20px; }
.results-intro p { color: var(--muted); font-size: 15px; }
.results-icon { width: 48px; height: 48px; flex: none; display: grid; place-items: center; border-radius: 12px; background: var(--surface-3); color: var(--accent); }
.results-overview { margin-bottom: var(--space-5); }
.results-two-col { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-5); align-items: start; }
.results-panel .result-dashboard { padding: var(--space-5); }
.method-note { display: flex; gap: var(--space-3); margin-top: var(--space-5); padding: var(--space-4) var(--space-5); border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); color: var(--muted); font-size: 14px; }
.method-note svg { flex: none; margin-top: 3px; color: var(--accent); }
.method-note strong { color: var(--text); }

.footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: var(--space-2); padding: var(--space-4) clamp(16px, 4vw, 48px); border-top: 1px solid var(--border); color: var(--muted-2); font-size: 13px; }

/* Markdown + math */
.markdown-content { overflow-wrap: anywhere; }
.markdown-content > :first-child { margin-top: 0; }
.markdown-content > :last-child { margin-bottom: 0; }
.markdown-content p { margin: 0 0 10px; }
.markdown-content ul, .markdown-content ol { margin: 8px 0 12px; padding-left: 22px; }
.markdown-content li { margin: 4px 0; }
.markdown-content pre, .markdown-content code { font-family: var(--mono); }
.markdown-content pre { white-space: pre-wrap; background: rgba(0,0,0,.3); border: 1px solid var(--border); padding: 10px; border-radius: 8px; font-size: 14px; }
.markdown-content :not(pre) > code { background: rgba(0,0,0,.3); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; font-size: .9em; }
.markdown-content table { border-collapse: collapse; margin: 8px 0; display: block; overflow-x: auto; }
.markdown-content th, .markdown-content td { border: 1px solid var(--border); padding: 6px 10px; }
.markdown-content .katex-display { overflow-x: auto; overflow-y: hidden; padding: 5px 0; margin: 12px 0; }
.markdown-content strong { color: #f3f6fb; }

/* Responsive */
@media (max-width: 1040px) {
  .split-workspace, .demo-columns, .results-two-col { grid-template-columns: 1fr; }
  .analysis-panel { position: static; max-height: none; }
  .chat-panel { height: 780px; }
  .intro-page { grid-template-columns: 1fr; gap: var(--space-6); }
  .intro-visual { min-height: 380px; }
}
@media (max-width: 760px) {
  body { font-size: 15px; }
  .topbar { height: auto; padding-top: 10px; padding-bottom: 10px; flex-wrap: wrap; }
  .top-nav { gap: var(--space-4); flex-wrap: wrap; }
  .nav-link { padding: 8px 0; }
  .brand-copy small { display: none; }
  .hub-grid, .comparison-grid, .results-overview { grid-template-columns: 1fr; }
  .pipeline-steps > svg { display: none; }
  .pipeline-step { min-width: 100%; }
  .compare-pair { grid-template-columns: 1fr; }
  .bubble-wrap { max-width: 88%; }
  .composer-bottom > span { display: none; }
  .composer-bottom { justify-content: flex-end; }
  .signal-card { position: static; margin: 6px 0; }
  .intro-visual { display: flex; flex-direction: column; align-items: stretch; min-height: 0; padding: var(--space-4) 0; }
  .visual-orbit, .visual-core, .visual-caption { display: none; }
  .results-intro { flex-direction: column; align-items: flex-start; }
  .page-wrap { padding-top: var(--space-5); }
}
`;

/* -------------------------------------------------------------------------- */
/*  Constants & helpers                                                        */
/* -------------------------------------------------------------------------- */
const LABELS = {
  CONSISTENT: { title: "Consistent", color: "green", Icon: Check, description: "The assistant maintained its position rather than simply yielding to user pressure." },
  JUSTIFIED_CHANGE: { title: "Justified change", color: "blue", Icon: RotateCcw, description: "The assistant changed its answer for a supported reason, such as valid new information or a correction." },
  SYCOPHANTIC_SHIFT: { title: "Sycophantic shift", color: "red", Icon: GitBranch, description: "The assistant changed its answer to align with unsupported user feedback." },
};

const CLASS_FILTERS = [
  { value: "ALL", label: "All classes" },
  { value: "CONSISTENT", label: "Consistent" },
  { value: "JUSTIFIED_CHANGE", label: "Justified Change" },
  { value: "SYCOPHANTIC_SHIFT", label: "Sycophantic Shift" },
];
const MAX_CLASS_SAMPLING_ATTEMPTS = 30;

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "";
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

// Turns raw errors into short, friendly messages (no stack traces or internals).
function friendlyError(err, fallback = "Something went wrong. Please try again.") {
  const message = String(err?.message || "");
  if (!message) return fallback;
  if (/failed to fetch|networkerror|load failed|network request failed/i.test(message)) {
    return "Can't reach the analysis server. Check that the backend is running, then try again.";
  }
  if (/\[object|traceback|at \w+\.\w+ \(|\n\s+at /i.test(message)) return fallback;
  return message.length > 240 ? `${message.slice(0, 237)}…` : message;
}

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

function getLabel(result) {
  return result?.prediction || result?.final_prediction || result?.label || "UNKNOWN";
}
function getProbabilities(result) {
  return result?.classifier_probabilities || result?.probabilities || result?.class_probabilities || {};
}
function prettyLabel(label) {
  return (label || "UNKNOWN").replaceAll("_", " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}
function humanize(key) {
  const text = String(key).replaceAll("_", " ").replace(/\s+/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}
function safeString(value) {
  if (value === null || value === undefined || value === "") return "Not available";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}
function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || value === "") return "Not provided";
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "Not provided";
}
function clampPct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.min(100, n * 100)) : 0;
}
function metaFor(label) {
  return LABELS[label] || { title: prettyLabel(label), color: "neutral", Icon: CircleHelp, description: "The pipeline returned a classification." };
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

/* ---- Result model: reads only fields the backend actually returns ---- */
const VERIFIER_FIELDS = ["initial_answer_supported", "feedback_validity", "user_correction_supported", "revised_answer_supported", "revision_follows_feedback"];

function getVerificationState(verification, source) {
  const obj = isObj(verification) ? verification : null;
  const status = String(obj?.status ?? "").toLowerCase();
  const flaggedFailure = Boolean(obj && (obj.error || obj.error_message || obj.available === false || obj.success === false || obj.verified === false || ["error", "failed", "unavailable", "skipped", "timeout"].includes(status)));
  const fallbackSource = /fallback|unavailable|classifier[_\s-]*only/i.test(String(source ?? ""));
  const hasFindings = Boolean(obj && VERIFIER_FIELDS.some((k) => obj[k] !== undefined && obj[k] !== null));
  if (flaggedFailure || fallbackSource) return "unavailable";
  if (hasFindings) return "verified";
  return "none";
}

function buildVerifierChecks(verification) {
  const v = isObj(verification) ? verification : {};
  const checks = [];
  const add = (value, yes, no) => { if (value !== undefined && value !== null) checks.push({ ok: Boolean(value), text: value ? yes : no }); };
  add(v.initial_answer_supported, "The initial answer was judged supported.", "The initial answer was not judged supported.");
  if (v.feedback_validity) checks.push({ ok: String(v.feedback_validity).toUpperCase() === "VALID", text: `User feedback was assessed as ${String(v.feedback_validity).toLowerCase()}.` });
  add(v.user_correction_supported, "The user's correction is supported.", "The user's correction is not supported.");
  add(v.revised_answer_supported, "The revised answer is supported.", "The revised answer is not supported.");
  add(v.revision_follows_feedback, "The revision follows the user's feedback.", "The revision does not appear to follow the user's feedback.");
  return checks;
}

const KNOWN_EVIDENCE = {
  answer_changed: ["The revised answer differs from the initial answer.", "The assistant kept the same answer."],
  contains_new_information: ["The feedback contains new information.", "The feedback contains no new information."],
  contains_correction: ["The feedback contains a correction.", "The feedback contains no correction."],
};

function buildEvidenceSignals(evidence) {
  if (!isObj(evidence)) return [];
  const signals = [];
  for (const [key, value] of Object.entries(evidence)) {
    if (value === null || value === undefined) continue;
    if (KNOWN_EVIDENCE[key] && typeof value === "boolean") {
      signals.push({ kind: value ? "yes" : "no", text: KNOWN_EVIDENCE[key][value ? 0 : 1] });
    } else if (typeof value === "boolean") {
      signals.push({ kind: value ? "yes" : "no", text: `${humanize(key)}: ${value ? "yes" : "no"}` });
    } else if (typeof value === "number" && Number.isFinite(value)) {
      signals.push({ kind: "info", text: `${humanize(key)}: ${Number.isInteger(value) ? value : value.toFixed(2)}` });
    } else if (typeof value === "string" && value.length <= 60) {
      signals.push({ kind: "info", text: `${humanize(key)}: ${value}` });
    }
  }
  return signals.slice(0, 8);
}

function analyzeResult(result) {
  const label = getLabel(result);
  const probabilities = isObj(getProbabilities(result)) ? getProbabilities(result) : {};
  const evidence = result?.evidence || result?.extracted_evidence || result?.analysis?.evidence || null;
  const verification = result?.verification || result?.llm_verification || result?.analysis?.verification || null;
  const reason = result?.decision_reason || result?.reason || result?.explanation || result?.decision?.reason || "";
  const source = result?.decision_source ?? null;
  const confidence = result?.confidence;

  // Classifier prediction: use an explicit backend field when present,
  // otherwise report the highest-probability DeBERTa class (clearly labelled as derived).
  let classifierPrediction = result?.classifier_prediction || result?.classifier_label || result?.deberta_prediction || result?.classifier?.prediction || null;
  let classifierDerived = false;
  if (!classifierPrediction) {
    const ranked = Object.keys(LABELS).filter((k) => Number.isFinite(Number(probabilities[k]))).sort((a, b) => Number(probabilities[b]) - Number(probabilities[a]));
    if (ranked.length) { classifierPrediction = ranked[0]; classifierDerived = true; }
  }

  return {
    label,
    meta: metaFor(label),
    confidence,
    probabilities,
    reason: typeof reason === "string" ? reason : "",
    source,
    evidence,
    verification,
    verificationState: getVerificationState(verification, source),
    checks: buildVerifierChecks(verification),
    signals: buildEvidenceSignals(evidence),
    verifierText: isObj(verification) ? [verification.summary, verification.explanation, verification.reasoning, verification.reason].find((t) => typeof t === "string" && t.trim()) : "",
    classifierPrediction,
    classifierDerived,
  };
}

/* -------------------------------------------------------------------------- */
/*  App                                                                        */
/* -------------------------------------------------------------------------- */
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
  const [selectedClass, setSelectedClass] = useState("ALL");

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
      setError(friendlyError(err, "Something went wrong while processing the message."));
    } finally {
      setBusy("");
    }
  }

  async function loadRandomExample(classFilter) {
    const filter = typeof classFilter === "string" ? classFilter : selectedClass;
    setError("");
    setBusy("random");
    try {
      // Uses the existing /examples/random endpoint. For a specific class, examples are
      // drawn until one with the requested target_label comes back.
      let data = null;
      const maxAttempts = filter === "ALL" ? 1 : MAX_CLASS_SAMPLING_ATTEMPTS;
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        const candidate = await api("/examples/random");
        if (filter === "ALL" || candidate?.example?.target_label === filter) { data = candidate; break; }
      }
      if (!data) {
        setError(`Couldn't find a ${prettyLabel(filter)} example after several tries. Please try again.`);
        return;
      }
      setExample(data.example);
      setReferenceResult(null);
      setLiveSimulation(null);
      setLiveDemoResult(null);
      setDemoStep("loaded");
      setPage("demo");
    } catch (err) {
      setError(friendlyError(err, "Could not load an example."));
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
      setError(friendlyError(err, "Could not analyze the reference conversation."));
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
      setError(friendlyError(err, "The live simulation could not be completed."));
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
      setError(friendlyError(err, "Could not analyze the live conversation."));
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
      <style>{STYLES}</style>
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
          <CircleHelp size={18} /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={16} /></button>
        </div>
      )}

      {page === "intro" && <IntroPage onEnter={() => navigate("hub")} />}
      {page === "hub" && <HubPage onLive={() => navigate("live")} onDemo={() => loadRandomExample()} busy={busy === "random"} />}
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
          onLoadRandom={() => loadRandomExample()}
          selectedClass={selectedClass}
          onSelectClass={setSelectedClass}
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
          onNewExample={() => loadRandomExample()}
        />
      )}

      <footer className="footer">
        <span>SYCOCHECK · Research prototype</span>
        <span>Detecting sycophantic drift in multi-turn LLM interactions</span>
      </footer>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Pages                                                                      */
/* -------------------------------------------------------------------------- */
function IntroPage({ onEnter }) {
  return (
    <main className="intro-page page-wrap">
      <div className="intro-copy">
        <div className="eyebrow"><span className="eyebrow-line" /> Research prototype <span className="eyebrow-chip">3-CLASS DETECTION</span></div>
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
        <div className="visual-core"><GitBranch size={32} /><span>BEHAVIOR<br />ANALYSIS</span></div>
        <div className="signal-card signal-top"><span className="signal-icon green"><Check size={15} /></span><div><strong>Consistent</strong><small>Stands by supported answer</small></div></div>
        <div className="signal-card signal-right"><span className="signal-icon blue"><RotateCcw size={15} /></span><div><strong>Justified change</strong><small>Updates with evidence</small></div></div>
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
        <div className="eyebrow"><span className="eyebrow-line" /> The workspace</div>
        <h1>Choose how to <span>investigate.</span></h1>
        <p>Explore a live conversation or run a controlled comparison against a validated research example.</p>
      </div>
      <div className="hub-grid">
        <button className="mode-card live-mode" onClick={onLive}>
          <div className="mode-card-top"><span className="mode-icon"><MessageCircle size={22} /></span><span className="mode-number">01 / INTERACTIVE</span></div>
          <h2>Live AI Chat</h2>
          <p>Talk to the assistant, challenge an answer, and inspect how its response changes. Analysis updates alongside the conversation.</p>
          <div className="mode-tags"><span>Live LLM</span><span>Instant analysis</span><span>Evidence view</span></div>
          <div className="mode-card-cta">Open live chat <ArrowUpRight size={17} /></div>
        </button>
        <button className="mode-card demo-mode" onClick={onDemo} disabled={busy}>
          <div className="mode-card-top"><span className="mode-icon"><FlaskConical size={22} /></span><span className="mode-number">02 / CONTROLLED TEST</span></div>
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
  const [selected, setSelected] = useState(null);
  const activeIndex = selected !== null && selected < analyses.length ? selected : analyses.length - 1;
  const current = activeIndex >= 0 ? analyses[activeIndex] : null;
  const isLatest = activeIndex === analyses.length - 1;

  useEffect(() => { setSelected(null); }, [analyses.length]);

  return (
    <main className="workspace-page page-wrap">
      <div className="workspace-header">
        <button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Workspace</button>
        <div><div className="eyebrow"><span className="eyebrow-line" /> Live analysis</div><h1>Conversation <span>workspace</span></h1></div>
        <button className="button button-quiet button-small" onClick={onClear} disabled={busy}><RotateCcw size={14} /> Clear chat</button>
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
          <div className="panel-header"><div className="panel-title"><span className="panel-icon"><BarChart3 size={17} /></span><div><strong>Behavior analysis</strong><small>{busy ? "Working on the latest turn…" : "Updates after detected feedback"}</small></div></div><span className="panel-count">{busy && <LoaderCircle className="spin" size={12} />}{analyses.length} {analyses.length === 1 ? "event" : "events"}</span></div>
          <div className="analysis-content">
            {!current ? (
              <div className="analysis-empty"><div className="empty-ring"><Activity size={22} /></div><h3>Awaiting a feedback event</h3><p>The classification appears here after the assistant responds to your feedback. Your conversation stays on the left.</p><div className="empty-checklist"><span><CheckCircle2 size={15} /> Initial answer recorded</span><span><CheckCircle2 size={15} /> User feedback detected</span><span><CheckCircle2 size={15} /> Revised answer analyzed</span></div></div>
            ) : (
              <>
                {(current.feedback_type || current.feedback_confidence !== undefined) && (
                  <div className="feedback-chip-row">
                    {current.feedback_type && <span className="chip"><i />Feedback type: {humanize(current.feedback_type)}</span>}
                    {Number.isFinite(Number(current.feedback_confidence)) && current.feedback_confidence !== null && <span className="chip"><i />Detection confidence {formatPercent(current.feedback_confidence, 0)}</span>}
                  </div>
                )}
                <ResultDashboard result={current} heading={isLatest ? "Latest classification" : `Event ${activeIndex + 1} classification`} />
                {analyses.length > 1 && (
                  <div className="history-list" style={{ marginTop: 24 }}>
                    <div className="block-title">Analysis history</div>
                    {analyses.map((item, index) => (
                      <button key={index} type="button" className={`history-item ${index === activeIndex ? "selected" : ""}`} onClick={() => setSelected(index)}>
                        <span className={`status-dot ${metaFor(getLabel(item)).color}`} /><span>{prettyLabel(getLabel(item))}</span><small>Event {index + 1}</small>
                      </button>
                    ))}
                  </div>
                )}
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

/* -------------------------------------------------------------------------- */
/*  Results dashboard                                                          */
/* -------------------------------------------------------------------------- */
function SourceChip({ source, state }) {
  const text = source ? humanize(source) : "Not provided";
  const tone = state === "verified" ? "ok" : state === "unavailable" ? "warn" : "";
  return <span className={`chip ${tone}`} title="Which part of the pipeline produced the final decision"><i />Decision source: {text}</span>;
}

function VerdictCard({ model, heading }) {
  const { meta } = model;
  const { Icon } = meta;
  const hasConfidence = model.confidence !== undefined && model.confidence !== null && Number.isFinite(Number(model.confidence));
  return (
    <section className="verdict">
      <div className="verdict-top"><span className="verdict-kicker">{heading || "Final classification"}</span><SourceChip source={model.source} state={model.verificationState} /></div>
      <div className="verdict-title"><span className="verdict-icon"><Icon size={22} /></span><h3>{meta.title}</h3></div>
      <p className="verdict-lead">{meta.description}</p>
      {model.reason && <div className="verdict-why"><span className="verdict-why-label">Why</span><MarkdownText>{model.reason}</MarkdownText></div>}
      <div className="confidence">
        <div className="confidence-row"><span>Decision confidence</span><strong>{hasConfidence ? formatPercent(model.confidence) : "Not provided"}</strong></div>
        {hasConfidence && <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(clampPct(model.confidence))} aria-label="Decision confidence"><span style={{ width: `${clampPct(model.confidence)}%` }} /></div>}
        <p className="confidence-note">How strongly the pipeline supports this final decision. It is not a calibrated probability of being correct.</p>
      </div>
    </section>
  );
}

function ProbabilityBars({ result }) {
  const probabilities = isObj(getProbabilities(result)) ? getProbabilities(result) : {};
  const available = Object.keys(LABELS).some((k) => Number.isFinite(Number(probabilities[k])) && probabilities[k] !== null);
  const top = Object.keys(LABELS).reduce((best, k) => (Number(probabilities[k]) > Number(probabilities[best] ?? -1) ? k : best), "CONSISTENT");
  return (
    <section className="card">
      <div className="block-title">Class probabilities <span className="subtle-label">DEBERTA</span></div>
      {!available ? <p className="prob-empty">Class probabilities were not returned for this result.</p> : Object.keys(LABELS).map((label) => {
        const raw = probabilities[label];
        const known = raw !== undefined && raw !== null && Number.isFinite(Number(raw));
        return (
          <div className={`prob-row ${label === top ? "top" : ""}`} key={label}>
            <div className="prob-meta"><span>{prettyLabel(label)}</span><strong>{known ? formatPercent(raw) : "—"}</strong></div>
            <div className="prob-track"><span className={LABELS[label].color} style={{ width: `${known ? clampPct(raw) : 0}%` }} /></div>
          </div>
        );
      })}
    </section>
  );
}

function VerificationNotice({ state }) {
  if (state === "verified") return <div className="notice ok"><CheckCircle2 size={16} /><span><strong>LLM verification completed.</strong> The findings below come from the verifier.</span></div>;
  if (state === "unavailable") return <div className="notice warn"><CircleHelp size={16} /><span><strong>LLM verification unavailable.</strong> This result is a classifier-only fallback and was not confirmed by the verifier.</span></div>;
  return <div className="notice neutral"><CircleHelp size={16} /><span><strong>No verifier findings were returned.</strong> This result has not been confirmed by LLM verification.</span></div>;
}

function TechnicalAnalysis({ model }) {
  const [open, setOpen] = useState(false);
  const { meta } = model;
  const classifierMeta = model.classifierPrediction ? metaFor(model.classifierPrediction) : null;
  const sameClass = model.classifierPrediction ? model.classifierPrediction === model.label : null;
  const statusChip = model.verificationState === "verified" ? { cls: "ok", text: "Verified" } : model.verificationState === "unavailable" ? { cls: "warn", text: "Verification unavailable" } : { cls: "", text: "Not verified" };

  return (
    <section className="tech">
      <button type="button" className="tech-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="tech-toggle-left"><Microscope size={17} />Technical analysis <span className={`chip ${statusChip.cls}`}><i />{statusChip.text}</span></span>
        <ChevronDown size={18} className={open ? "rotated" : ""} />
      </button>
      {open && (
        <div className="tech-body">
          <div className="tech-section">
            <h4>Classifier vs final prediction</h4>
            <div className="compare-pair">
              <div className={`compare-cell tone-${classifierMeta?.color || "neutral"}`}><span>{model.classifierDerived ? "Top DeBERTa class" : "Classifier prediction"}</span><strong>{classifierMeta ? classifierMeta.title : "Not provided"}</strong></div>
              <div className={`compare-cell tone-${meta.color}`}><span>Final prediction</span><strong>{meta.title}</strong></div>
            </div>
            <p className="tech-note">
              {sameClass === null ? "The classifier's own prediction was not returned." : sameClass ? "The final decision agrees with the classifier." : "The final decision differs from the classifier's top class."}
              {model.classifierDerived && " Classifier class is the highest of the returned DeBERTa probabilities."}
            </p>
          </div>

          <div className="tech-section">
            <h4>Key evidence</h4>
            {model.signals.length ? (
              <ul className="signal-list">{model.signals.map((s, i) => <li key={i}><span className={`mark ${s.kind}`}>{s.kind === "yes" ? "✓" : s.kind === "no" ? "–" : "i"}</span><span>{s.text}</span></li>)}</ul>
            ) : <p className="tech-note" style={{ marginTop: 0 }}>No structured evidence was returned for this result.</p>}
          </div>

          <div className="tech-section">
            <h4>Verifier findings</h4>
            <VerificationNotice state={model.verificationState} />
            {model.verificationState === "verified" && (
              <>
                {model.checks.length > 0 && <ul className="signal-list" style={{ marginTop: 12 }}>{model.checks.map((c, i) => <li key={i}><span className={`mark ${c.ok ? "yes" : "no"}`}>{c.ok ? "✓" : "–"}</span><span>{c.text}</span></li>)}</ul>}
                {model.verifierText && <p className="verifier-text">{model.verifierText.length > 320 ? `${model.verifierText.slice(0, 317)}…` : model.verifierText}</p>}
              </>
            )}
          </div>

          {(model.evidence || model.verification) && (
            <div className="details-block">
              <RawFields evidence={model.evidence} verification={model.verification} />
            </div>
          )}
        </div>
      )}
    </section>
  );
}

// Preserves the original raw-field access, tucked behind a second toggle for debugging.
function RawFields({ evidence, verification }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="details-toggle" onClick={() => setOpen(!open)}><ChevronDown size={14} className={open ? "rotated" : ""} />{open ? "Hide" : "Show"} raw evidence & verification fields</button>
      {open && <pre className="json-view">{safeString({ evidence: evidence ?? null, verification: verification ?? null })}</pre>}
    </>
  );
}

function ResultDashboard({ result, heading }) {
  const model = analyzeResult(result);
  return (
    <div className={`result-dashboard tone-${model.meta.color}`}>
      <VerdictCard model={model} heading={heading} />
      <ProbabilityBars result={result} />
      <TechnicalAnalysis model={model} />
      <div className="analysis-footnote"><ShieldCheck size={14} /> Produced by the existing classification pipeline; shown as returned.</div>
    </div>
  );
}

function ResultSummary({ result, title }) {
  const label = getLabel(result);
  const meta = metaFor(label);
  const hasConfidence = result?.confidence !== undefined && result?.confidence !== null && Number.isFinite(Number(result.confidence));
  return <div className={`result-summary tone-${meta.color}`}><span>{title}</span><strong>{prettyLabel(label)}</strong><small>{hasConfidence ? `Decision confidence · ${formatPercent(result.confidence)}` : "Not analyzed yet"}</small></div>;
}

/* -------------------------------------------------------------------------- */
/*  Demo                                                                       */
/* -------------------------------------------------------------------------- */
function ConversationCard({ title, conversation, variant = "reference" }) {
  if (!conversation) return <div className="conversation-card placeholder"><span className="placeholder-icon"><MessageSquare size={19} /></span><strong>{title}</strong><p>This conversation has not been generated yet.</p></div>;
  const turns = [
    ["question", "QUESTION", conversation.question],
    ["initial", variant === "live" ? "LIVE INITIAL ANSWER" : "INITIAL ANSWER", conversation.initial_answer],
    ["feedback", "USER FEEDBACK", conversation.user_feedback],
    ["revised", variant === "live" ? "LIVE REVISED ANSWER" : "REVISED ANSWER", conversation.revised_answer],
  ];
  return <div className={`conversation-card ${variant}`}><div className="conversation-card-heading"><span>{title}</span><span className="conversation-type">{variant === "live" ? "FRESH LLM" : "DATASET"}</span></div>{turns.map(([kind, label, content]) => <div className={`conversation-turn turn-${kind}`} key={kind}><div className="turn-label">{label}</div><div className="conversation-turn-content"><MarkdownText>{content || "Not provided"}</MarkdownText></div></div>)}</div>;
}

function ComparisonCell({ title, label }) {
  const meta = metaFor(label);
  return <div className="comparison-cell"><span>{title}</span><strong className={meta.color}><i />{prettyLabel(label)}</strong></div>;
}

function ClassSelector({ value, onChange, disabled }) {
  return (
    <div className="class-selector">
      <span className="class-selector-label">Example class</span>
      <div className="class-selector-options" role="radiogroup" aria-label="Example class">
        {CLASS_FILTERS.map((option) => (
          <button key={option.value} type="button" role="radio" aria-checked={value === option.value} className={`class-option ${value === option.value ? "active" : ""}`} onClick={() => onChange(option.value)} disabled={disabled}>{option.label}</button>
        ))}
      </div>
    </div>
  );
}

function DemoPage({ example, referenceResult, liveSimulation, liveDemoResult, busy, selectedClass, onSelectClass, onLoadRandom, onAnalyzeReference, onSimulate, onAnalyzeLive, onViewResults, onBack }) {
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
        <div><div className="eyebrow"><span className="eyebrow-line" /> Controlled comparison</div><h1>Demo <span>mode</span></h1></div>
        <button className="button button-primary button-small" onClick={onLoadRandom} disabled={busy !== ""}><RotateCcw size={14} /> {busy === "random" ? "Loading…" : "Load random example"}</button>
      </div>
      <ClassSelector value={selectedClass} onChange={onSelectClass} disabled={busy !== ""} />
      {!example ? (
        <div className="large-empty"><FlaskConical size={30} /><h2>Load a random validated example</h2><p>Choose a class above, or sample from all accepted behavior classes.</p><button className="button button-primary" onClick={onLoadRandom} disabled={Boolean(busy)}>{busy === "random" ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} Load random example</button></div>
      ) : (
        <>
          <div className="demo-meta-row"><span className="sample-chip"><span className="sample-dot" /> Random accepted example</span><span className="demo-meta-note">Ground truth is kept separate from model prediction.</span></div>
          <div className="demo-columns">
            <section className="demo-column"><div className="column-heading"><div><span className="column-index">01</span><h2>Reference conversation</h2></div><span className="column-caption">VALIDATED DATA</span></div>
              <ConversationCard title="Example conversation" conversation={referenceConversation} />
              <div className="action-row"><button className="button button-secondary" onClick={onAnalyzeReference} disabled={Boolean(busy)}>{busy === "reference" ? <LoaderCircle className="spin" size={15} /> : <Activity size={15} />} {busy === "reference" ? "Analyzing…" : "Analyze reference"}</button><button className="text-button" onClick={() => setShowGroundTruth(!showGroundTruth)}>{showGroundTruth ? "Hide ground truth" : "Reveal ground truth"}</button></div>
              {showGroundTruth && <div className={`ground-truth tone-${metaFor(example.target_label).color}`}><span>Dataset label</span><strong>{prettyLabel(example.target_label)}</strong></div>}
              {referenceResult && <ResultDashboard result={referenceResult} heading="Reference classification" />}
            </section>
            <section className="demo-column"><div className="column-heading"><div><span className="column-index">02</span><h2>Live simulation</h2></div><span className="column-caption">FRESH LLM</span></div>
              {liveConversation ? <ConversationCard title="Live-generated conversation" conversation={liveConversation} variant="live" /> : <div className="conversation-card placeholder"><span className="placeholder-icon"><Zap size={19} /></span><strong>Simulation not run yet</strong><p>The same question and feedback will be used, but the assistant responses will be generated fresh by the live LLM.</p></div>}
              <div className="action-row"><button className="button button-primary" onClick={onSimulate} disabled={!example || Boolean(busy)}>{busy === "simulation" ? <LoaderCircle className="spin" size={15} /> : <Zap size={15} />} {busy === "simulation" ? "Generating…" : "Simulate with live LLM"}</button>{liveConversation && <button className="button button-secondary" onClick={onAnalyzeLive} disabled={Boolean(busy)}>{busy === "live-analysis" ? <LoaderCircle className="spin" size={15} /> : <Activity size={15} />} {busy === "live-analysis" ? "Analyzing…" : "Analyze live"}</button>}</div>
              {liveDemoResult && <ResultDashboard result={liveDemoResult} heading="Live classification" />}
            </section>
          </div>
          {(referenceResult || liveDemoResult) && <div className="comparison-panel"><div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> Result comparison</div><h2>What did the system detect?</h2></div><span className="section-note">Reference and live are classified independently.</span></div><div className="comparison-grid"><ComparisonCell title="Dataset ground truth" label={example.target_label} /><ComparisonCell title="Reference prediction" label={getLabel(referenceResult)} /><ComparisonCell title="Live prediction" label={getLabel(liveDemoResult)} /></div>{referenceResult && liveDemoResult && <div className={`comparison-verdict ${getLabel(liveDemoResult) === example.target_label ? "match" : "different"}`}>{getLabel(liveDemoResult) === example.target_label ? <CheckCircle2 size={17} /> : <GitBranch size={17} />}<div><strong>{getLabel(liveDemoResult) === example.target_label ? "Live result matches the reference label" : "Live result differs from the reference label"}</strong><p>A difference is a valid experimental outcome; the live model is not forced to reproduce the dataset behavior.</p></div></div>}{referenceResult && liveDemoResult && <button className="button button-primary" onClick={onViewResults}>Open detailed results <ArrowRight size={16} /></button>}</div>}
        </>
      )}
    </main>
  );
}

function DemoResultsPage({ example, referenceResult, liveSimulation, liveDemoResult, onBack, onNewExample }) {
  if (!example) return <main className="page-wrap"><div className="large-empty"><h2>No demo result yet</h2><p>Load a random example and run the demo first.</p><button className="button button-primary" onClick={onBack}>Back to demo</button></div></main>;
  const referenceLabel = getLabel(referenceResult);
  const liveLabel = getLabel(liveDemoResult);
  const bothReady = Boolean(referenceResult && liveDemoResult);
  const same = bothReady && referenceLabel === liveLabel;
  return (
    <main className="page-wrap results-page">
      <div className="workspace-header"><button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Back to demo</button><div><div className="eyebrow"><span className="eyebrow-line" /> Experiment summary</div><h1>Demo <span>results</span></h1></div><button className="button button-secondary button-small" onClick={onNewExample}><RotateCcw size={14} /> New example</button></div>
      <div className="results-intro"><div className="results-icon"><BarChart3 size={22} /></div><div><h2>Reference vs live simulation</h2><p>The reference comes from the validated dataset. The live conversation is newly generated. Their outcomes are compared, not forced to match.</p></div></div>
      <div className="results-overview">
        <ResultSummary result={referenceResult || { prediction: "NOT ANALYZED" }} title="Reference prediction" />
        <ResultSummary result={liveDemoResult || { prediction: "NOT ANALYZED" }} title="Live prediction" />
        <div className={`result-summary tone-${same ? "green" : "neutral"}`}><span>Prediction agreement</span><strong>{bothReady ? (same ? "Same class" : "Different classes") : "Pending"}</strong><small>Classifier outputs compared</small></div>
      </div>
      <div className="results-two-col">
        <section className="panel results-panel"><div className="panel-header"><div className="panel-title"><span className="panel-icon"><BookOpen size={17} /></span><div><strong>Reference analysis</strong><small>Validated dataset interaction</small></div></div></div>{referenceResult ? <ResultDashboard result={referenceResult} heading="Reference classification" /> : <p className="muted-pad">Analyze the reference conversation on the Demo Mode page.</p>}</section>
        <section className="panel results-panel"><div className="panel-header"><div className="panel-title"><span className="panel-icon"><Zap size={17} /></span><div><strong>Live simulation analysis</strong><small>Fresh LLM responses</small></div></div></div>{liveDemoResult ? <ResultDashboard result={liveDemoResult} heading="Live classification" /> : <p className="muted-pad">Simulate and analyze the live conversation on the Demo Mode page.</p>}</section>
      </div>
      <div className="method-note"><ShieldCheck size={16} /><div><strong>How to read this result</strong><p>The dataset label is the reference target. Model predictions are produced by the existing DeBERTa, evidence extraction, verification, and decision pipeline. A live mismatch can reveal that the live LLM did not reproduce the reference behavior.</p></div></div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
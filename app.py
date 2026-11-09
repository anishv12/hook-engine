"""Hook Engine (Streamlit): prospect in, researched and reviewed outreach draft out. Nothing is ever sent."""
import csv
import html
import io
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import streamlit as st

# --- config: .env locally, st.secrets in the cloud (never printed) ---
for envf in (Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"):
    if envf.exists():
        for line in envf.read_text(encoding="utf8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("\"'")
                if k and v:  # .env wins over stray system variables
                    os.environ[k] = v
try:
    for k in ("GEMINI_API_KEY", "ACCESS_CODE", "GEMINI_MODEL", "GEMINI_FAST_MODEL", "DAILY_RUN_LIMIT"):
        if k not in os.environ and k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import engine as E  # noqa: E402  (after env is loaded)

st.set_page_config(page_title="Hook Engine", page_icon="🎣", layout="wide", initial_sidebar_state="collapsed")

# verdict: (stamp text, tone, one-line meaning)
VERDICT = {
    "ready": ("READY FOR REVIEW", "ok", "Grounded hook, passed every gate. A person approves before anything leaves."),
    "needs_edit": ("NEEDS AN EDIT", "warn", "A quality gate flagged something. Fix it in the editor before approving."),
    "no_send": ("DON'T SEND YET", "bad", "No fresh, relevant signal to hang an email on, so no draft was written. A generic email would be worse than none."),
    "hold": ("HELD", "hold", "Recent bad news around this account. Wait before reaching out."),
    "redirect": ("ASK FOR A REFERRAL", "info", "Not the finance owner. The draft asks them to point us to the right person."),
    "no_info": ("NO INFO FOUND", "bad", "Nothing usable was found about this person or company, so no email was drafted."),
    "error": ("ERROR", "bad", "The run failed."),
    "running": ("RUNNING", "info", "Working through the pipeline."),
}
PILL = {"ready": "Ready", "needs_edit": "Needs edit", "no_send": "Don't send", "hold": "Held", "redirect": "Referral", "no_info": "No info", "error": "Error", "running": "Running"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#080e1c;--card:#0f1a30;--card2:#15223d;--line:#213456;--ink:#e8eefb;--mute:#8296b8;--amber:#4d9bff;--ok:#5fd38d;--bad:#ff6b6b;--info:#6ea8ff;--hold:#c792ea;
--c-rel:#4d9bff;--c-fresh:#5fd38d;--c-spec:#6ea8ff;--c-fit:#c792ea}
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:radial-gradient(1200px 500px at 10% -10%,rgba(77,155,255,.07),transparent 60%),var(--bg)}
.block-container{padding-top:1.6rem;max-width:1200px}
h1,h2,h3,.serif{font-family:'Fraunces',serif!important;letter-spacing:-.01em}
.mono{font-family:'JetBrains Mono',monospace}
header[data-testid="stHeader"]{background:transparent}
/* hero */
.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap;padding:.4rem 0 1.1rem;border-bottom:1px solid var(--line);margin-bottom:1rem}
.brand{display:flex;align-items:center;gap:.8rem}
.mark{width:46px;height:46px;border-radius:14px;background:conic-gradient(from 210deg,#4d9bff,#22d3ee,#4d9bff);display:grid;place-items:center;font-size:1.5rem;box-shadow:0 8px 30px rgba(77,155,255,.25)}
.hero h1{margin:0;font-size:2.1rem;line-height:1;font-weight:700}
.hero .sub{color:var(--mute);font-size:.92rem;margin-top:.25rem}
.chips{display:flex;gap:.45rem;flex-wrap:wrap}
.chip{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--mute);border:1px solid var(--line);border-radius:99px;padding:.25rem .65rem;background:rgba(255,255,255,.02)}
.chip b{color:var(--amber);font-weight:500}
/* rail */
.rail-wrap{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.2rem 1rem;margin:.6rem 0 1.1rem}
.bar{position:relative;height:5px;border-radius:99px;background:var(--line);overflow:hidden;margin-bottom:1.1rem}
.fill{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#4d9bff,#22d3ee);border-radius:99px;transition:width .6s ease}
.creep{position:absolute;top:0;bottom:0;width:0;border-radius:99px;background:linear-gradient(90deg,rgba(77,155,255,.9),rgba(34,211,238,.4));animation:creep 70s cubic-bezier(.2,.6,.3,1) forwards}
@keyframes creep{to{width:var(--seg)}}
.rail{display:flex;list-style:none;margin:0;padding:0}
.node{flex:1 1 0;min-width:0;position:relative;text-align:center;padding:0 3px}
.node:not(:first-child)::before{content:"";position:absolute;top:19px;right:50%;width:100%;height:2px;background:var(--line);z-index:0}
.node.done:not(:first-child)::before,.node.warn:not(:first-child)::before,.node.running:not(:first-child)::before{background:linear-gradient(90deg,var(--amber),var(--amber))}
.node.running:not(:first-child)::before{background:linear-gradient(90deg,var(--line),var(--amber),var(--line));background-size:200% 100%;animation:flow 1.2s linear infinite}
@keyframes flow{from{background-position:200% 0}to{background-position:-200% 0}}
.dot{position:relative;z-index:1;display:inline-grid;place-items:center;width:38px;height:38px;border-radius:50%;background:var(--card2);border:2px solid var(--line);font-family:'JetBrains Mono',monospace;font-size:.82rem;color:var(--mute)}
.node.done .dot{background:var(--amber);border-color:var(--amber);color:#04122b;font-weight:600}
.node.warn .dot{background:var(--hold);border-color:var(--hold);color:#150a22;font-weight:600}
.node.error .dot{background:var(--bad);border-color:var(--bad);color:#2a0606}
.node.skipped .dot{background:transparent;border-style:dashed;color:var(--mute)}
.node.skipped .lbl{color:var(--mute);font-weight:500;text-decoration:line-through;text-decoration-color:var(--line)}
.node.paused .dot{background:var(--info);border-color:var(--info);color:#06142a}
.node.running .dot{border-color:var(--amber);color:var(--amber);animation:pulse 1.3s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(77,155,255,.55)}50%{box-shadow:0 0 0 9px rgba(77,155,255,0)}}
.lbl{display:block;margin-top:.5rem;font-size:.78rem;font-weight:600;line-height:1.2}
.node.pending .lbl{color:var(--mute);font-weight:500}
.note{display:block;margin-top:.2rem;font-family:'JetBrains Mono',monospace;font-size:.66rem;color:var(--mute);min-height:1.7em;line-height:1.3}
.node.done .note{color:#a9cdfb}
.tail{margin-top:.8rem;padding-top:.7rem;border-top:1px dashed var(--line);font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--mute);line-height:1.6}
.tail .k{color:var(--amber)}
.err{margin-top:.7rem;color:var(--bad);font-size:.85rem}
@media(max-width:700px){.rail{flex-wrap:wrap;row-gap:1rem}.node{flex:1 1 25%}.node::before{display:none!important}}
/* verdict */
.verdict{display:flex;gap:1.1rem;align-items:center;border-radius:18px;padding:1.1rem 1.3rem;border:1px solid var(--line);background:var(--card);position:relative;overflow:hidden;margin-bottom:1rem}
.verdict::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--tone)}
.verdict.ok{--tone:var(--ok)}.verdict.warn{--tone:var(--amber)}.verdict.bad{--tone:var(--bad)}.verdict.hold{--tone:var(--hold)}.verdict.info{--tone:var(--info)}
.stamp{font-family:'Fraunces',serif;font-weight:700;font-size:1.35rem;color:var(--tone);border:2px solid var(--tone);border-radius:10px;padding:.35rem .8rem;transform:rotate(-2.5deg);white-space:nowrap;letter-spacing:.03em}
.vtxt{flex:1;min-width:0}
.vtxt .who{font-family:'Fraunces',serif;font-size:1.15rem}
.vtxt .why{color:var(--mute);font-size:.88rem;margin-top:.2rem}
.metas{display:flex;gap:.9rem;flex-wrap:wrap}
.meta{text-align:center}.meta .v{font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:var(--ink)}.meta .l{font-size:.66rem;color:var(--mute);text-transform:uppercase;letter-spacing:.08em}
@media(max-width:800px){.verdict{flex-direction:column;align-items:flex-start}}
/* cards */
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.2rem}
.h{font-family:'Fraunces',serif;font-size:1.15rem;margin:0 0 .6rem}
.sec{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--mute);text-transform:uppercase;letter-spacing:.12em;margin:1.2rem 0 .5rem}
/* email */
.mail{background:#f4f7fd;color:#101828;border-radius:14px;overflow:hidden;box-shadow:0 14px 40px rgba(0,0,0,.35)}
.mail .top{background:#e9e4d4;padding:.6rem 1rem;display:flex;gap:.4rem;align-items:center}
.mail .top i{width:10px;height:10px;border-radius:50%;background:#c9c2ab;display:inline-block}
.mail .hd{padding:.8rem 1.2rem;border-bottom:1px solid #ddd6c0;font-size:.82rem;color:#5b574a;line-height:1.7}
.mail .hd b{color:#1c1b17;font-weight:600}
.mail .subj{font-family:'Fraunces',serif;font-size:1.25rem;color:#1c1b17;padding:.9rem 1.2rem .2rem}
.mail .body{padding:.5rem 1.2rem 1.3rem;font-size:.97rem;line-height:1.65;white-space:pre-wrap}
.grounds{display:flex;flex-direction:column;gap:.45rem;margin-top:.8rem}
.ground{border:1px solid var(--line);border-left:3px solid var(--amber);background:var(--card);border-radius:10px;padding:.55rem .8rem;font-size:.82rem}
.ground a{color:var(--amber);text-decoration:none}.ground .c{color:var(--ink)}.ground .s{color:var(--mute);font-size:.75rem;margin-top:.15rem}
/* x-ray */
.leg{display:flex;gap:1rem;flex-wrap:wrap;font-size:.72rem;color:var(--mute);margin-bottom:.7rem}
.leg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:.35rem;vertical-align:-1px}
.xr{padding:.55rem .6rem;border-radius:10px;margin-bottom:.35rem;border:1px solid transparent}
.xr.win{background:rgba(77,155,255,.08);border-color:rgba(77,155,255,.45)}
.xr .t{display:flex;justify-content:space-between;gap:.8rem;font-size:.82rem}
.xr .t span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.xr .sc{font-family:'JetBrains Mono',monospace;color:var(--ink)}
.xr .m{font-size:.68rem;color:var(--mute);margin-top:.15rem;font-family:'JetBrains Mono',monospace}
.stack{position:relative;display:flex;height:9px;border-radius:99px;background:var(--card2);margin-top:.4rem;overflow:visible}
.stack b{display:block;height:100%}
.stack b:first-child{border-radius:99px 0 0 99px}
.stack .thr{position:absolute;top:-4px;bottom:-4px;width:0;border-left:1.5px dashed rgba(233,230,223,.55)}
.tag{display:inline-block;font-size:.62rem;border:1px solid var(--line);border-radius:6px;padding:0 .35rem;margin-left:.3rem;color:var(--mute)}
/* rules + gates */
.rule{display:flex;gap:.7rem;padding:.55rem 0;border-bottom:1px solid var(--line);font-size:.85rem}
.rule:last-child{border:none}
.rd{flex:none;width:11px;height:11px;border-radius:50%;margin-top:.3rem;border:2px solid var(--line)}
.rule.fired .rd{background:var(--amber);border-color:var(--amber);box-shadow:0 0 0 4px rgba(77,155,255,.15)}
.rule .rn{font-weight:600}.rule .rx{color:var(--mute)}
.gates{display:flex;flex-wrap:wrap;gap:.45rem}
.gate{font-size:.78rem;border-radius:99px;padding:.3rem .75rem;border:1px solid var(--line)}
.gate.p{color:var(--ok);border-color:rgba(95,211,141,.4);background:rgba(95,211,141,.07)}.gate.f{color:var(--bad);border-color:rgba(255,107,107,.4);background:rgba(255,107,107,.07)}
/* dashboard */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1rem}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:.9rem 1.1rem}
.kpi .v{font-family:'Fraunces',serif;font-size:2rem;line-height:1.1}.kpi .l{font-size:.7rem;color:var(--mute);text-transform:uppercase;letter-spacing:.1em;margin-top:.2rem}
@media(max-width:800px){.kpis{grid-template-columns:repeat(2,1fr)}}
.dist{display:flex;height:14px;border-radius:99px;overflow:hidden;background:var(--card2);margin:.5rem 0 .5rem}
.dist b{display:block}
.dleg{display:flex;gap:1rem;flex-wrap:wrap;font-size:.76rem;color:var(--mute)}
.dleg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:.35rem}
.row{display:grid;grid-template-columns:1.1fr 1.2fr .9fr 1fr .8fr;gap:.8rem;align-items:center;padding:.65rem .9rem;border-bottom:1px solid var(--line);font-size:.86rem}
.row.hdr{font-family:'JetBrains Mono',monospace;font-size:.66rem;color:var(--mute);text-transform:uppercase;letter-spacing:.1em}
.row .nm{font-weight:600}.row .co{color:var(--mute)}
.pill{display:inline-block;font-size:.72rem;padding:.15rem .6rem;border-radius:99px;border:1px solid var(--tone);color:var(--tone)}
.pill.ok{--tone:var(--ok)}.pill.warn{--tone:var(--amber)}.pill.bad{--tone:var(--bad)}.pill.hold{--tone:var(--hold)}.pill.info{--tone:var(--info)}
.mini{height:6px;border-radius:99px;background:var(--card2);overflow:hidden}.mini b{display:block;height:100%;background:linear-gradient(90deg,var(--amber),#22d3ee)}
@media(max-width:800px){.row{grid-template-columns:1fr 1fr}}
.empty{text-align:center;color:var(--mute);padding:3rem 1rem;border:1px dashed var(--line);border-radius:18px}
.empty .big{font-size:2.4rem}
/* streamlit widget polish */
div[data-testid="stTabs"] button[role="tab"]{font-family:'Fraunces',serif;font-size:1.05rem}
.stButton>button,.stLinkButton>a{border-radius:12px!important;font-weight:600}
.stTextInput input,.stTextArea textarea{border-radius:12px!important}

[class*="st-key-mail_"]{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:0 1.1rem 1rem;overflow:hidden}
.mailhd{display:flex;gap:1rem;align-items:center;margin:0 -1.1rem .8rem;padding:.7rem 1.1rem;background:var(--card2);border-bottom:1px solid var(--line);font-size:.82rem;color:var(--mute);line-height:1.7}
.mailhd b{color:var(--ink)}
.dots i{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--line);margin-right:5px}
.ground{display:flex;gap:.7rem}
.gn{flex:none;width:22px;height:22px;border-radius:50%;background:var(--amber);color:#04122b;font-size:.72rem;font-weight:700;display:grid;place-items:center;margin-top:.1rem}
.gate{border-radius:12px!important;padding:.45rem .8rem!important;width:100%}
.gate span{color:var(--mute);font-size:.72rem}
.gates{flex-direction:column}
.people{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:.7rem}
.person{display:flex;gap:.8rem;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.8rem .9rem}
.av{flex:none;width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,var(--amber),#22d3ee);color:#04122b;font-weight:700;display:grid;place-items:center;font-family:'Fraunces',serif}
.pn{font-weight:600}.pr{color:var(--mute);font-size:.82rem}.ps{font-size:.72rem;margin-top:.15rem}.ps a{color:var(--amber);text-decoration:none}
/* inputs */
.formcard{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);border-radius:20px;padding:1.2rem 1.3rem .4rem;margin-bottom:.8rem}
.formcard .ft{font-family:'Fraunces',serif;font-size:1.25rem}.formcard .fs{color:var(--mute);font-size:.85rem;margin-bottom:.6rem}
[class*="st-key-form"] label p{font-family:'JetBrains Mono',monospace;font-size:.68rem!important;text-transform:uppercase;letter-spacing:.12em;color:var(--amber)!important}
[class*="st-key-form"] input{background:#0a1224!important;border:1px solid var(--line)!important;font-size:1.02rem!important;padding:.75rem .9rem!important}
[class*="st-key-form"] input:focus{border-color:var(--amber)!important;box-shadow:0 0 0 3px rgba(77,155,255,.2)!important}
[class*="st-key-form"]{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);border-radius:20px;padding:1.2rem 1.3rem 1rem;margin-bottom:.8rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero():
    st.markdown(
        '<div class="hero"><div class="brand"><div class="mark">🎣</div><div><h1>Hook Engine</h1>'
        '<div class="sub">One prospect in. A researched, checked, human-approved first email out.</div></div></div>'
        '<div class="chips"><span class="chip"><b>7</b> stages</span><span class="chip">AI judges · <b>code decides</b></span>'
        '<span class="chip">nothing sends <b>automatically</b></span></div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- access gate
code = os.environ.get("ACCESS_CODE")
if code and not st.session_state.get("authed"):
    hero()
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="card"><div class="h">Enter the access code</div><div style="color:var(--mute);font-size:.85rem">This demo calls a live model, so it is gated.</div></div>', unsafe_allow_html=True)
        entered = st.text_input("Access code", type="password", label_visibility="collapsed", placeholder="Access code")
        if entered:
            if entered == code:
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("That code isn't right.")
    st.stop()


# ---------------------------------------------------------------- pipeline rail
def node_note(run, sid, st_):
    """A short result line under each finished stage."""
    try:
        if st_["status"] == "running":
            return "working…"
        if st_["status"] == "pending":
            return ""
        if st_["status"] == "skipped":
            return "not needed"
        d = st_.get("data") or {}
        res = run.get("result") or {}
        if sid == "intake":
            return (run["input"].get("role") or "").strip()[:22] or "captured"
        if sid == "research":
            n = (run.get("researchStats") or {}).get("pagesSeen")
            return f"{n} headlines" if n else ""
        if sid == "score":
            return f"{len(d.get('scored') or [])} rated"
        if sid == "rules":
            dec = d.get("decision") or {}
            fired = [x for x in dec.get("log", []) if x.get("fired")]
            return f"{len(fired)} rule{'s' if len(fired) != 1 else ''} fired" if fired else "all clear"
        if sid == "draft":
            body = ((d.get("draft") or {}).get("body") or "")
            return f"{len(body.split())} words" if body else ""
        if sid == "quality":
            p = d.get("personalisation")
            return f"personal {p:.2f}" if isinstance(p, (int, float)) else ""
        if sid == "review":
            return "your turn" if res else ""
    except Exception:
        pass
    return ""


def render_progress(run, ph, tail=True):
    """Seven nodes on a rail. The bar fills as stages finish and creeps while one runs; every node keeps its result."""
    stages = [(sid, label, run["stages"][sid]) for sid, label in E.STAGES]
    n = len(stages)

    def finished(sid, st_):
        return st_["status"] in ("done", "warn", "skipped") or (sid == "review" and st_["status"] == "paused")
    done = sum(finished(sid, st_) for sid, _, st_ in stages)
    running = [st_ for _, _, st_ in stages if st_["status"] == "running"]
    pct = done / n * 100
    creep = ""
    if running:
        t0 = min(st_.get("_t0") or time.time() for st_ in running)
        creep = f'<div class="creep" style="left:{pct:.2f}%;--seg:{len(running) / n * 92:.2f}%;animation-delay:-{max(time.time() - t0, 0):.1f}s"></div>'
    lis = []
    for i, (sid, label, st_) in enumerate(stages):
        status = st_["status"]
        if sid == "review" and status == "paused":
            cls, dot = "done", "✓"
        elif status == "paused":
            cls, dot = "paused", "?"
        else:
            cls = status if status in ("done", "warn", "error", "running", "skipped") else "pending"
            dot = {"done": "✓", "warn": "!", "error": "✕", "skipped": "–"}.get(status, f"{i + 1:02d}")
        t = f" · {st_['ms'] / 1000:.1f}s" if st_.get("ms") else ""
        lis.append(f'<li class="node {cls}"><span class="dot">{dot}</span><span class="lbl">{html.escape(label)}</span>'
                   f'<span class="note">{html.escape(node_note(run, sid, st_))}{t}</span></li>')
    extra = ""
    if tail and running:
        ev = (run["stages"]["research"].get("data") or {}).get("events") or []
        if ev:
            extra = '<div class="tail">' + "<br>".join(f'<span class="k">{html.escape(str(e["kind"]))}</span> {html.escape(str(e["text"]))[:130]}' for e in ev[-3:]) + "</div>"
    err = next((s_["error"] for s_ in run["stages"].values() if s_.get("error")), None)
    if err:
        extra += f'<div class="err">{html.escape(err)}</div>'
    ph.markdown(f'<div class="rail-wrap"><div class="bar"><div class="fill" style="width:{pct:.2f}%"></div>{creep}</div>'
                f'<ol class="rail">{"".join(lis)}</ol>{extra}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def get_run_input():
    return {"name": st.session_state.f_name.strip(), "company": st.session_state.f_company.strip(),
            "role": st.session_state.f_role.strip(), "notes": st.session_state.f_notes.strip(),
            "linkedin": st.session_state.f_linkedin.strip()}


def finish(run):
    if run["status"] not in ("running",):
        E.save_run(run)
    st.session_state["run"] = run


def gmail_url(target, subject, body):
    return f"https://mail.google.com/mail/?view=cm&fs=1&su={quote(subject)}&body={quote(body)}"


def esc(x):
    return html.escape(str(x if x is not None else ""))


def show_alternatives(run):
    """Other senior people at the same company, shown when no draft was written."""
    alts = run.get("alternatives") or []
    co = run["input"].get("company") or "this company"
    st.markdown(f'<div class="sec">Other people at {esc(co)} you could reach out to</div>', unsafe_allow_html=True)
    if not alts:
        st.markdown('<div class="empty" style="padding:1.2rem">No other names turned up in public news for this company.</div>', unsafe_allow_html=True)
        return
    cards = "".join(
        f'<div class="person"><div class="av">{esc(a["name"][:1])}</div><div><div class="pn">{esc(a["name"])}</div><div class="pr">{esc(a["role"])}</div>'
        f'<div class="ps"><a href="{esc(a["url"])}" target="_blank">{esc(a["source"] or "source")}</a>{" · " + esc(a["date"]) if a.get("date") else ""}</div></div></div>' for a in alts)
    st.markdown(f'<div class="people">{cards}</div><div style="color:var(--mute);font-size:.75rem;margin-top:.4rem">Names and roles come from public news and Wikipedia. Check the role is still current before reaching out.</div>', unsafe_allow_html=True)


def show_run(run, key, readonly=False):
    inp = run["input"]
    res = run.get("result")
    tone_key = run["status"] if run["status"] in VERDICT else "running"
    stamp, tone, meaning = VERDICT[tone_key]

    step_ph = st.empty()
    render_progress(run, step_ph, tail=False)
    with st.expander("Research log", expanded=run["status"] == "error"):
        for e in (run["stages"]["research"].get("data") or {}).get("events") or []:
            st.markdown(f"- `{e['kind']}` {e['text']}")
        stats = run.get("researchStats")
        if stats:
            st.caption(f"{stats.get('webSearches', 0)} searches · {stats.get('pagesSeen', 0)} headlines read · {stats.get('seconds', '?')}s")
    if not res and run["status"] != "no_info":
        return
    if not res:  # nothing found at all
        st.markdown(
            f'<div class="verdict {tone}"><div class="stamp">{stamp}</div><div class="vtxt"><div class="who">{esc(inp["name"])} · {esc(inp.get("company"))}</div>'
            f'<div class="why">{esc(run.get("noDraftReason") or meaning)}</div></div></div>', unsafe_allow_html=True)
        show_alternatives(run)
        return
    d, dec, q = res["draft"], res["decision"], res["quality"]
    sc = (run["stages"]["score"].get("data") or {}).get("scored") or []
    hook_id = (res.get("choice") or {}).get("hookId")
    fired = [r for r in dec["log"] if r["fired"]]
    why = run.get("noDraftReason") if not d else (fired[0]["text"] if fired else meaning)
    if run["status"] == "hold":
        why = f"Held until {dec['holdUntil']}: bad news inside the last 30 days." + (" Tone is careful if you send later." if d else " No draft was written.")
    metas = (f'<div class="metas"><div class="meta"><div class="v">{q["personalisation"]:.2f}</div><div class="l">personal</div></div>'
             f'<div class="meta"><div class="v">{len(d["body"].split())}</div><div class="l">words</div></div>'
             f'<div class="meta"><div class="v">{len(fired)}</div><div class="l">rules fired</div></div></div>') if d else \
        f'<div class="metas"><div class="meta"><div class="v">{len(fired)}</div><div class="l">rules fired</div></div></div>'
    st.markdown(
        f'<div class="verdict {tone}"><div class="stamp">{stamp}</div><div class="vtxt"><div class="who">{esc(inp["name"])} · {esc(dec["target"].get("company") or inp.get("company"))}</div>'
        f'<div class="why">{esc(why)}</div></div>{metas}</div>', unsafe_allow_html=True)
    for w in dec["warnings"]:
        st.caption("⚠ " + w)

    if not d:  # stopped before drafting: show what was found and which rule stopped it
        st.markdown('<div class="sec">What the research found</div>', unsafe_allow_html=True)
        rows = "".join(f'<div class="ground"><span class="gn">{i}</span><div><div class="c">{esc(x["title"])}</div><div class="s">{esc(x.get("date") or "undated")} · relevance {x["relevance"]:.2f} · freshness {x["freshness"]:.2f} · score {x["score"]:.2f}</div></div></div>'
                       for i, x in enumerate(sorted(sc, key=lambda x: -x["score"])[:6], 1))
        st.markdown(f'<div class="grounds">{rows or "<div class=empty>No headlines to show.</div>"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec">Guardrails · AI judges, code decides</div>', unsafe_allow_html=True)
        rules = "".join(f'<div class="rule {"fired" if r["fired"] else ""}"><span class="rd"></span><div><span class="rn">{esc(r["rule"])}</span><div class="rx">{esc(r["text"])}</div></div></div>' for r in dec["log"])
        st.markdown(f'<div class="card">{rules}</div>', unsafe_allow_html=True)
        show_alternatives(run)
        return

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="sec">Your draft · edit anything, then approve</div>', unsafe_allow_html=True)
        to_line = f'{esc(inp["name"])} &lt;{esc(inp.get("role") or "finance leader")}, {esc(dec["target"].get("company") or inp.get("company"))}&gt;'
        with st.container(key=f"mail_{key}"):
            st.markdown(f'<div class="mailhd"><span class="dots"><i></i><i></i><i></i></span><div><b>From</b> Anish · Zamp<br><b>To</b> {to_line}</div></div>', unsafe_allow_html=True)
            opts = [d["subject"]] + [x for x in d.get("subjectAlternatives", []) if x and x != d["subject"]]
            if readonly:
                subj = st.text_input("Subject", d["subject"], key=f"subj_{key}", disabled=True)
            else:
                subj = st.selectbox("Subject (pick another idea, or type your own)", opts, key=f"subj_{key}", accept_new_options=True) or d["subject"]
            body = st.text_area("Email body", d["body"], height=300, key=f"body_{key}", disabled=readonly)
            st.caption(f"{len(body.split())} words")
            if not readonly:
                c1, c2, c3 = st.columns(3)
                if c1.button("✅ Approve", key=f"ap_{key}", type="primary", use_container_width=True):
                    run["review"] = {"decision": "approved", "subject": subj, "body": body}
                    E.save_run(run)
                    st.success("Approved. Nothing was sent. Copy it or open it in Gmail.")
                if c2.button("🗑 Discard", key=f"di_{key}", use_container_width=True):
                    run["review"] = {"decision": "discarded"}
                    E.save_run(run)
                    st.info("Discarded.")
                c3.link_button("Open in Gmail", gmail_url(dec["target"], subj, body), use_container_width=True)
                with st.expander("Copy-friendly text"):
                    st.code(f"Subject: {subj}\n\n{body}", language=None)
            elif run.get("review"):
                st.caption("Review decision: " + run["review"]["decision"])

    with right:
        by_id = {s["id"]: s for s in sc}
        claims = [c for c in d.get("claims", []) if c.get("signal_id") in by_id]
        st.markdown('<div class="sec">Claim traces · every claim points to a source</div>', unsafe_allow_html=True)
        if claims:
            g = ""
            for i, c in enumerate(claims, 1):
                s = by_id[c["signal_id"]]
                link = f'<a href="{esc(s.get("url"))}" target="_blank">{esc(s["title"])[:100]}</a>' if s.get("url") else esc(s["title"])[:100]
                g += (f'<div class="ground"><span class="gn">{i}</span><div><div class="c">“{esc(c.get("text"))}”</div>'
                      f'<div class="s">↳ {link}<br>{esc(s.get("date") or "undated")} · hook score {s["score"]:.2f}</div></div></div>')
            st.markdown(f'<div class="grounds">{g}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty" style="padding:1.5rem">No factual claims in this draft, so nothing to trace.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Guardrails · AI judges, code decides</div>', unsafe_allow_html=True)
    g1, g2 = st.columns([1.3, 1], gap="large")
    with g1:
        rules = "".join(f'<div class="rule {"fired" if r["fired"] else ""}"><span class="rd"></span><div><span class="rn">{esc(r["rule"])}</span><div class="rx">{esc(r["text"])}</div></div></div>' for r in dec["log"])
        st.markdown(f'<div class="card">{rules}</div>', unsafe_allow_html=True)
    with g2:
        gates = "".join(f'<div class="gate {"p" if c["pass"] else "f"}"><b>{"✓" if c["pass"] else "✕"} {esc(c["label"])}</b><br><span>{esc(c["detail"])}</span></div>' for c in q["checks"])
        st.markdown(f'<div class="card"><div class="h">Quality gates</div><div class="gates">{gates}</div></div>', unsafe_allow_html=True)
        if not q["passed"]:
            st.warning("Still failing a gate after 4 automatic rewrites. This is the best version; please fix it in the editor before approving.")
        elif q["attempt"] > 1:
            st.caption(f"The first draft failed a gate, so it was rewritten automatically until it passed (version {q['attempt']}).")


ALIASES = {"name": ("name", "full name", "prospect", "prospect name", "contact", "contact name"),
           "company": ("company", "company name", "organisation", "organization", "account", "employer"),
           "role": ("role", "title", "job title", "designation", "position"),
           "notes": ("notes", "note", "context", "comments")}


def read_rows(up):
    """Rows from an uploaded .xlsx/.xls/.csv. Headers are matched loosely. Keeps rows with a name and a company, max 50."""
    raw = up.getvalue()
    if up.name.lower().endswith(".csv"):
        recs = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    else:
        import pandas as pd
        df = pd.read_excel(io.BytesIO(raw)).fillna("")
        recs = df.astype(str).to_dict("records")
    out = []
    for r in recs:
        low = {str(k).strip().lower(): str(v or "").strip() for k, v in r.items() if k is not None}
        row = {}
        for field, names in ALIASES.items():
            row[field] = next((low[n] for n in names if low.get(n)), "")
        if row["name"] and row["company"]:
            out.append(row)
    return out[:50]


def run_one(inp, fresh):
    st.session_state["run"] = None
    run = E.new_run(inp, "live", None, fresh)
    ph = st.empty()
    render_progress(run, ph)
    out = E.run_pipeline(run, lambda r: render_progress(r, ph))
    finish(out)
    return out


# ---------------------------------------------------------------- sidebar (system status)
with st.sidebar:
    st.markdown("### System")
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    st.write("Gemini key: " + ("✅ loaded" if has_key else "❌ missing (add GEMINI_API_KEY to .env)"))
    st.caption(f"Models: {E.model_default()} / {E.model_fast()}")
    st.caption(f"Seller: {E.SELLER['name']}")
    limit = int(os.environ.get("DAILY_RUN_LIMIT") or 150)
    st.caption(f"Runs today: {E.runs_today()} / {limit}")

hero()
tab_new, tab_dash = st.tabs(["New run", "Dashboard"])

# ---------------------------------------------------------------- new run
with tab_new:
    for k in ("f_name", "f_company", "f_role", "f_notes", "f_linkedin"):
        st.session_state.setdefault(k, "")

    mode_single = st.radio("Mode", ["One prospect", "A list (Excel / CSV)"], horizontal=True, label_visibility="collapsed")
    if mode_single == "One prospect":
        with st.container(key="form_one"):
            st.markdown('<div class="ft serif" style="font-size:1.25rem">Who are we writing to?</div><div style="color:var(--mute);font-size:.85rem;margin-bottom:.6rem">Name and company are enough. A role and a note make the email sharper.</div>', unsafe_allow_html=True)
            a, b, c = st.columns([1.2, 1.2, 1])
            a.text_input("👤  Prospect name", key="f_name", placeholder="Priya Sharma")
            b.text_input("🏢  Company", key="f_company", placeholder="Acme Logistics")
            c.text_input("💼  Role", key="f_role", placeholder="CFO")
            n1, n2 = st.columns(2)
            n1.text_input("📝  Context (optional)", key="f_notes", placeholder="Met at the finance summit")
            n2.text_input("🔗  LinkedIn (optional, never scraped)", key="f_linkedin", placeholder="linkedin.com/in/…")
            fresh = st.checkbox("Fresh research (skip the 24h cache)")
            go = st.button("Run the engine  →", type="primary", disabled=not has_key)
        if go:
            inp = get_run_input()
            if not inp["name"] or not inp["company"]:
                st.error("Enter the name and the company. Without a company the search can't tell people with the same name apart.")
            elif E.runs_today() >= limit:
                st.error("Daily run limit reached.")
            else:
                run_one(inp, fresh)
                st.rerun()
    else:
        with st.container(key="form_list"):
            st.markdown('<div class="ft serif" style="font-size:1.25rem">Drop in a prospect list</div><div style="color:var(--mute);font-size:.85rem;margin-bottom:.6rem">Excel (.xlsx) or CSV. Columns: <b>name</b> and <b>company</b> required; <b>role</b> and <b>notes</b> optional. Up to 50 rows, each researched live. Drafts land in the Dashboard.</div>', unsafe_allow_html=True)
            up = st.file_uploader("List file", type=["xlsx", "xls", "csv"], label_visibility="collapsed")
            rows = []
            if up:
                try:
                    rows = read_rows(up)
                except Exception as e:
                    st.error(f"Couldn't read that file: {e}")
                if rows:
                    st.caption(f"{len(rows)} prospect(s) found. Preview:")
                    st.dataframe(rows[:8], use_container_width=True, hide_index=True)
                elif up:
                    st.warning("No usable rows. Each row needs a name and a company.")
            go = st.button("Run the list  →", type="primary", disabled=not (has_key and rows))
        if go and rows:
            bar = st.progress(0.0, text=f"0 / {len(rows)}")
            for i, r in enumerate(rows):
                if E.runs_today() >= limit:
                    st.error("Daily run limit reached.")
                    break
                run = E.new_run({"name": r["name"], "company": r.get("company", ""), "role": r.get("role", ""), "notes": r.get("notes", ""), "linkedin": ""}, "live")
                E.save_run(E.run_pipeline(run, lambda _r: None) or run)
                bar.progress((i + 1) / len(rows), text=f"{i + 1} / {len(rows)}: {r['name']}")
            st.success("Done. Open the Dashboard tab to review each draft.")

    if st.session_state.get("run"):
        show_run(st.session_state["run"], st.session_state["run"]["id"])
    else:
        st.markdown('<div class="empty"><div class="big">🎣</div>Enter a prospect above. You will watch seven stages run live, then get a verdict, a draft and the reasoning behind it.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- dashboard
SAMPLE = ' <span class="tag">sample</span>'
with tab_dash:
    runs = E.load_all_runs()
    if not runs:
        st.markdown('<div class="empty"><div class="big">📭</div>No runs yet. Run a prospect from the first tab and it will appear here.</div>', unsafe_allow_html=True)
    else:
        counts = {}
        for r in runs:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        pers = [((r.get("result") or {}).get("quality") or {}).get("personalisation") for r in runs]
        pers = [p for p in pers if isinstance(p, (int, float))]
        approved = sum(1 for r in runs if (r.get("review") or {}).get("decision") == "approved")
        stopped = sum(counts.get(s, 0) for s in ("no_send", "hold", "redirect"))
        kp = [(len(runs), "Runs"), (f"{counts.get('ready', 0) / len(runs) * 100:.0f}%", "Ready for review"),
              (f"{sum(pers) / len(pers):.2f}" if pers else "–", "Avg personalisation"), (f"{approved} / {stopped}", "Approved / stopped by a rule")]
        st.markdown('<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in kp) + "</div>", unsafe_allow_html=True)

        colors = {"ready": "var(--ok)", "needs_edit": "var(--amber)", "no_send": "var(--bad)", "hold": "var(--hold)", "redirect": "var(--info)", "no_info": "var(--bad)", "error": "#777"}
        dist = "".join(f'<b style="width:{n / len(runs) * 100:.1f}%;background:{colors.get(s, "#777")}"></b>' for s, n in counts.items())
        leg = "".join(f'<span><i style="background:{colors.get(s, "#777")}"></i>{PILL.get(s, s)} · {n}</span>' for s, n in counts.items())
        st.markdown(f'<div class="card"><div class="h">Where the runs ended up</div><div class="dist">{dist}</div><div class="dleg">{leg}</div></div>', unsafe_allow_html=True)

        n_s = sum(1 for r in runs if r.get("sample"))
        if n_s:
            st.caption(f"{n_s} saved sample run(s) are pre-loaded so this view is never empty. Your own runs appear on top.")
        opts = ["All"] + [PILL.get(s, s) for s in counts]
        pick = st.radio("Filter", opts, horizontal=True, label_visibility="collapsed")
        shown = [r for r in runs if pick == "All" or PILL.get(r["status"], r["status"]) == pick]
        rows = ['<div class="card" style="padding:.3rem 0 .2rem"><div class="row hdr"><span>Prospect</span><span>Company</span><span>Verdict</span><span>Personalisation</span><span>Review</span></div>']
        for r in shown:
            res = r.get("result") or {}
            p = (res.get("quality") or {}).get("personalisation")
            tone = VERDICT.get(r["status"], VERDICT["running"])[1]
            bar = f'<div class="mini"><b style="width:{min(max(p, 0), 1) * 100:.0f}%"></b></div>' if isinstance(p, (int, float)) else '<span class="co">–</span>'
            co = (res.get("target") or {}).get("company") or r["input"].get("company") or "(none)"
            when = time.strftime("%d %b %H:%M", time.localtime(r["createdAt"]))
            rows.append(f'<div class="row"><span><span class="nm">{esc(r["input"]["name"])}</span>{SAMPLE if r.get("sample") else ""}<br><span class="co mono" style="font-size:.7rem">{when}</span></span><span class="co">{esc(co)}</span>'
                        f'<span><span class="pill {tone}">{PILL.get(r["status"], r["status"])}</span></span><span>{bar}</span><span class="co">{esc((r.get("review") or {}).get("decision", "–"))}</span></div>')
        rows.append("</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)

        if shown:
            labels = {r["id"]: f"{r['input']['name']}{' (sample)' if r.get('sample') else ''} · {time.strftime('%d %b %H:%M', time.localtime(r['createdAt']))}" for r in shown}
            st.markdown('<div class="sec">Open a run</div>', unsafe_allow_html=True)
            sel = st.selectbox("Open a run", list(labels), format_func=lambda i: labels[i], label_visibility="collapsed")
            if sel:
                show_run(next(r for r in shown if r["id"] == sel), "dash_" + sel, readonly=True)

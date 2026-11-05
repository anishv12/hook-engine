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

st.set_page_config(page_title="Hook Engine", page_icon="🎣", layout="wide")

STATUS = {
    "ready": ("✅ Ready for review", "success"),
    "needs_edit": ("✏️ Needs edit", "warning"),
    "no_send": ("⛔ Don't send yet", "error"),
    "hold": ("⏸ Held", "warning"),
    "redirect": ("↪️ Ask for referral", "info"),
    "awaiting_rep": ("🙋 Waiting for you to pick the person", "info"),
    "error": ("❌ Error", "error"),
    "running": ("⏳ Running", "info"),
}

# ---------------------------------------------------------------- access gate
code = os.environ.get("ACCESS_CODE")
if code and not st.session_state.get("authed"):
    st.title("🎣 Hook Engine")
    entered = st.text_input("Access code", type="password")
    if entered:
        if entered == code:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("That code isn't right.")
    st.stop()


# ---------------------------------------------------------------- helpers
STEP_CSS = (
    "<style>"
    ".he-wrap{margin:.25rem 0 1rem}"
    ".he-track{position:relative;height:8px;border-radius:99px;background:rgba(128,128,128,.25);overflow:hidden}"
    ".he-fill{position:absolute;left:0;top:0;bottom:0;border-radius:99px;background:linear-gradient(90deg,#16a34a,#22c55e);transition:width .6s ease}"
    ".he-creep{position:absolute;top:0;bottom:0;width:0;border-radius:99px;background:linear-gradient(90deg,#2563eb,#60a5fa);animation:he-creep 70s cubic-bezier(.2,.6,.3,1) forwards}"
    "@keyframes he-creep{to{width:var(--seg)}}"
    ".he-steps{display:flex;list-style:none;margin:.7rem 0 0;padding:0}"
    ".he-s{flex:1 1 0;min-width:0;text-align:center;font-size:.8rem;line-height:1.25;padding:0 2px}"
    ".he-dot{display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;border-radius:50%;border:2px solid rgba(128,128,128,.5);font-weight:600;font-size:.85rem;margin-bottom:.3rem}"
    ".he-s.pending{opacity:.55}"
    ".he-s.done .he-dot{background:#16a34a;border-color:#16a34a;color:#fff}"
    ".he-s.warn .he-dot{background:#d97706;border-color:#d97706;color:#fff}"
    ".he-s.error .he-dot{background:#dc2626;border-color:#dc2626;color:#fff}"
    ".he-s.running .he-dot{border-color:#2563eb;color:#2563eb;animation:he-pulse 1.2s infinite}"
    ".he-s.paused .he-dot{border-color:#2563eb;background:rgba(37,99,235,.15);color:#2563eb}"
    ".he-s .t{display:block;opacity:.6;font-size:.72rem}"
    "@keyframes he-pulse{0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,.5)}50%{box-shadow:0 0 0 6px rgba(37,99,235,0)}}"
    ".he-log{margin-top:.6rem;font-size:.8rem;opacity:.8;font-family:ui-monospace,monospace}"
    ".he-err{margin-top:.6rem;color:#dc2626;font-size:.85rem}"
    "@media(max-width:640px){.he-steps{flex-wrap:wrap;row-gap:.7rem}.he-s{flex:1 1 25%}}"
    "</style>"
)


def render_progress(run, ph, tail=True):
    """A linear 8-step track. The bar fills as stages finish and creeps forward while one is running."""
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
        creep = f'<div class="he-creep" style="left:{pct:.2f}%;--seg:{len(running) / n * 92:.2f}%;animation-delay:-{max(time.time() - t0, 0):.1f}s"></div>'
    lis = []
    for i, (sid, label, st_) in enumerate(stages):
        status = st_["status"]
        cls = "done" if finished(sid, st_) and status != "warn" else status
        if sid == "review" and status == "paused":
            cls, dot, sub = "done", "✓", "Your turn"
        elif status == "paused":
            cls, dot, sub = "paused", "?", "Waiting for you"
        else:
            dot = {"done": "✓", "warn": "!", "error": "✕"}.get(status, str(i + 1))
            sub = f"{st_['ms'] / 1000:.1f}s" if st_.get("ms") else ("working…" if status == "running" else "")
        lis.append(f'<li class="he-s {cls}"><span class="he-dot">{dot}</span><br>{html.escape(label)}<span class="t">{sub}</span></li>')
    extra = ""
    if tail and running:
        ev = (run["stages"]["research"].get("data") or {}).get("events") or []
        extra = '<div class="he-log">' + "<br>".join(html.escape(f"{e['kind']}: {e['text']}")[:140] for e in ev[-3:]) + "</div>"
    err = next((s_["error"] for s_ in run["stages"].values() if s_.get("error")), None)
    if err:
        extra += f'<div class="he-err">{html.escape(err)}</div>'
    ph.markdown(f'{STEP_CSS}<div class="he-wrap"><div class="he-track"><div class="he-fill" style="width:{pct:.2f}%"></div>{creep}</div>'
                f'<ol class="he-steps">{"".join(lis)}</ol>{extra}</div>', unsafe_allow_html=True)


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


def show_run(run, key, readonly=False):
    inp = run["input"]
    label, kind = STATUS.get(run["status"], (run["status"], "info"))
    st.subheader(f"{inp['name']} · {inp.get('company') or 'no company given'}")
    getattr(st, kind)(label)

    step_ph = st.empty()
    render_progress(run, step_ph, tail=False)
    with st.expander("Research log and stats", expanded=run["status"] == "error"):
        for e in (run["stages"]["research"].get("data") or {}).get("events") or []:
            st.markdown(f"- `{e['kind']}` {e['text']}")
        stats = run.get("researchStats")
        if stats:
            st.caption(f"{stats.get('webSearches', 0)} searches · {stats.get('pagesSeen', 0)} headlines read · {stats.get('seconds', '?')}s")

    if run["status"] == "awaiting_rep" and not readonly:
        ident = run["stages"]["identity"]["data"]["identity"]
        st.warning("Several different people match this name. Which one did you mean?")
        opts = {c["id"]: c.get("label") or c["id"] for c in ident["clusters"]}
        pick = st.radio("Who is it?", list(opts), format_func=lambda i: opts[i], key=f"pick_{key}")
        if st.button("Continue with this person", type="primary", key=f"cont_{key}"):
            finish(E.resume_pipeline(run, pick, lambda r: render_progress(r, step_ph)))
            st.rerun()
        return

    res = run.get("result")
    if not res:
        return
    d, dec, q = res["draft"], res["decision"], res["quality"]

    if dec.get("retargeted"):
        st.info(f"Retargeted: {inp['name']} now works at **{dec['target']['company']}** (you entered {inp.get('company')}).")
    if run["status"] == "hold":
        st.warning(f"Held until {dec['holdUntil']}: bad news inside the last 30 days. Tone is careful if you send later.")
    for w in dec["warnings"]:
        st.caption("⚠ " + w)

    st.markdown("#### Draft email")
    opts = [d["subject"]] + [x for x in d.get("subjectAlternatives", []) if x and x != d["subject"]]
    if readonly:
        subj = st.text_input("Subject", d["subject"], key=f"subj_{key}", disabled=True)
    else:
        subj = st.selectbox("Subject (pick another idea, or type your own)", opts, key=f"subj_{key}", accept_new_options=True) or d["subject"]
    body = st.text_area("Body", d["body"], height=240, key=f"body_{key}", disabled=readonly)
    st.caption(f"{len(body.split())} words · personalisation score {q['personalisation']:.2f} (bar {E.QUALITY_BAR})")
    if not readonly:
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Approve", key=f"ap_{key}"):
            run["review"] = {"decision": "approved", "subject": subj, "body": body}
            E.save_run(run)
            st.success("Approved. Nothing was sent. Copy it or open it in Gmail.")
        if c2.button("🗑 Discard", key=f"di_{key}"):
            run["review"] = {"decision": "discarded"}
            E.save_run(run)
            st.info("Discarded.")
        c3.link_button("Open in Gmail", gmail_url(dec["target"], subj, body))
        with st.expander("Copy-friendly text"):
            st.code(f"Subject: {subj}\n\n{body}", language=None)
    elif run.get("review"):
        st.caption("Review decision: " + run["review"]["decision"])

    t1, t2, t3 = st.tabs(["Why this angle (rules)", "Signals scored", "Quality gates"])
    with t1:
        st.caption(f"Pain chosen: **{E.anchor_label(res['choice']['anchor'])}**. AI judges, code decides: these rules are fixed code.")
        for r in dec["log"]:
            st.markdown(f"{'🔶' if r['fired'] else '▫️'} **{r['rule']}**: {r['text']}")
    with t2:
        sc = run["stages"]["score"]["data"]["scored"]
        st.dataframe([{"id": s["id"], "signal": s["title"], "date": s.get("date"), "score": s["score"], "relevance": s["relevance"],
                       "fresh": s["freshness"], "specific": s["specificity"], "person fit": s["personFit"], "valence": s["valence"],
                       "flags": ", ".join(s["flags"]), "hook": "★" if s["id"] == res["choice"]["hookId"] else "", "source": s.get("url")}
                      for s in sc], use_container_width=True, hide_index=True, column_config={"source": st.column_config.LinkColumn("source")})
    with t3:
        for c in q["checks"]:
            st.markdown(f"{'✅' if c['pass'] else '❌'} **{c['label']}**: {c['detail']}")
        st.markdown("**Claims → signals**")
        for c in d["claims"]:
            st.markdown(f"- {c.get('text')} → `{c.get('signal_id')}`")
        if q["attempt"] > 1:
            st.caption("The first draft failed a gate, so it was rewritten once.")


def run_one(inp, fresh):
    st.session_state["run"] = None
    run = E.new_run(inp, "live", None, fresh)
    ph = st.empty()
    render_progress(run, ph)
    out = E.run_pipeline(run, lambda r: render_progress(r, ph))
    finish(out)
    return out


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎣 Hook Engine")
    st.caption("Prospect in. Researched, checked, reviewed draft out. Nothing sends without a person.")
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    st.write("Gemini key: " + ("✅ loaded" if has_key else "❌ missing (add GEMINI_API_KEY to .env)"))
    st.caption(f"Models: {E.model_default()} / {E.model_fast()}")
    st.caption(f"Seller: {E.SELLER['name']}")
    limit = int(os.environ.get("DAILY_RUN_LIMIT") or 150)
    st.caption(f"Runs today: {E.runs_today()} / {limit}")

tab_new, tab_dash = st.tabs(["New run", "Dashboard"])

# ---------------------------------------------------------------- new run
with tab_new:
    for k in ("f_name", "f_company", "f_role", "f_notes", "f_linkedin"):
        st.session_state.setdefault(k, "")

    mode_single = st.radio("Mode", ["One prospect", "A list (CSV)"], horizontal=True, label_visibility="collapsed")
    if mode_single == "One prospect":
        a, b, c = st.columns(3)
        a.text_input("Name", key="f_name", placeholder="e.g. Priya Sharma")
        b.text_input("Company (optional)", key="f_company", placeholder="e.g. Acme Logistics")
        c.text_input("Role", key="f_role", placeholder="e.g. CFO")
        st.text_input("Notes (where you met, context)", key="f_notes")
        st.text_input("LinkedIn URL (optional, never scraped; just given to the search as a hint)", key="f_linkedin")
        fresh = st.checkbox("Fresh research (skip the 24h cache)")
        if st.button("Run", type="primary", disabled=not has_key):
            inp = get_run_input()
            if not inp["name"]:
                st.error("Enter a name.")
            elif E.runs_today() >= limit:
                st.error("Daily run limit reached.")
            else:
                run_one(inp, fresh)
                st.rerun()
    else:
        st.caption("CSV columns: name, company, role, notes (company/role/notes optional). Up to 50 rows. Each is researched live; drafts land in the Dashboard.")
        up = st.file_uploader("CSV file", type="csv")
        if up and st.button("Run the list", type="primary", disabled=not has_key):
            rows = list(csv.DictReader(io.StringIO(up.getvalue().decode("utf-8-sig"))))[:50]
            rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in rows]
            rows = [r for r in rows if r.get("name")]
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
        st.divider()
        show_run(st.session_state["run"], st.session_state["run"]["id"])

# ---------------------------------------------------------------- dashboard
with tab_dash:
    runs = E.load_runs()
    if not runs:
        st.info("No runs yet. Run a prospect from the first tab.")
    else:
        counts = {}
        for r in runs:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        mc = st.columns(max(len(counts), 1))
        for col, (s, n) in zip(mc, counts.items()):
            col.metric(STATUS.get(s, (s,))[0], n)
        st.dataframe([{"when": time.strftime("%d %b %H:%M", time.localtime(r["createdAt"])), "prospect": r["input"]["name"],
                       "company": (r.get("result") or {}).get("target", {}).get("company") or r["input"].get("company") or "(none)",
                       "status": STATUS.get(r["status"], (r["status"],))[0],
                       "personalisation": ((r.get("result") or {}).get("quality") or {}).get("personalisation"),
                       "review": (r.get("review") or {}).get("decision", "")} for r in runs],
                     use_container_width=True, hide_index=True)
        labels = {r["id"]: f"{r['input']['name']} · {time.strftime('%d %b %H:%M', time.localtime(r['createdAt']))}" for r in runs}
        sel = st.selectbox("Open a run", list(labels), format_func=lambda i: labels[i])
        if sel:
            show_run(next(r for r in runs if r["id"] == sel), "dash_" + sel, readonly=True)

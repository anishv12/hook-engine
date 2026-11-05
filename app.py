"""Hook Engine (Streamlit): prospect in, researched and reviewed outreach draft out. Nothing is ever sent."""
import csv
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
ICON = {"pending": "○", "running": "⏳", "done": "✅", "warn": "⚠️", "paused": "⏸", "error": "❌", "skipped": "–"}

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
def render_progress(run, ph):
    lines = []
    for sid, label in E.STAGES:
        s = run["stages"][sid]
        ms = f" · {s['ms'] / 1000:.1f}s" if s.get("ms") else ""
        lines.append(f"{ICON.get(s['status'], '○')} **{label}**{ms}")
    ev = (run["stages"]["research"].get("data") or {}).get("events") or []
    tail = "\n".join(f"- `{e['kind']}` {e['text']}" for e in ev[-6:])
    err = next((f"\n\n:red[{s['error']}]" for s in run["stages"].values() if s.get("error")), "")
    ph.markdown("  \n".join(lines) + ("\n\n**Research log**\n" + tail if tail else "") + err)


def get_run_input():
    return {"name": st.session_state.f_name.strip(), "company": st.session_state.f_company.strip(),
            "role": st.session_state.f_role.strip(), "notes": st.session_state.f_notes.strip(),
            "linkedin": st.session_state.f_linkedin.strip()}


def fill_preset(p):
    i = p["input"]
    st.session_state.update(f_name=i["name"], f_company=i["company"], f_role=i["role"], f_notes=i.get("notes", ""), f_linkedin="", preset_id=p["id"])


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

    with st.expander("Live run view: the 8 stages", expanded=run["status"] in ("error", "awaiting_rep")):
        ph = st.empty()
        render_progress(run, ph)
        stats = run.get("researchStats")
        if stats:
            st.caption(f"{stats.get('webSearches', 0)} Google searches · {stats.get('pagesSeen', 0)} pages seen · {stats.get('dropped', 0)} unverifiable claims dropped · {stats.get('seconds', '?')}s")

    if run["status"] == "awaiting_rep" and not readonly:
        ident = run["stages"]["identity"]["data"]["identity"]
        st.warning("Several different people match this name. Which one did you mean?")
        opts = {c["id"]: c.get("label") or c["id"] for c in ident["clusters"]}
        pick = st.radio("Who is it?", list(opts), format_func=lambda i: opts[i], key=f"pick_{key}")
        if st.button("Continue with this person", type="primary", key=f"cont_{key}"):
            ph2 = st.empty()
            with st.spinner("Drafting…"):
                finish(E.resume_pipeline(run, pick, lambda r: render_progress(r, ph2)))
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


def run_one(inp, mode, preset_id, fresh):
    run = E.new_run(inp, mode, preset_id, fresh)
    ph = st.empty()
    render_progress(run, ph)
    out = E.run_pipeline(run, lambda r: render_progress(r, ph))
    finish(out)
    ph.empty()
    return out


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎣 Hook Engine")
    st.caption("Prospect in. Researched, checked, reviewed draft out. Nothing sends without a person.")
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    st.write("Gemini key: " + ("✅ loaded" if has_key else "❌ missing (add GEMINI_API_KEY to .env)"))
    st.caption(f"Models: {E.model_default()} / {E.model_fast()}")
    st.caption(f"Seller: {E.SELLER['name']} · Snapshot dated {E.CAPTURED_AT}")
    limit = int(os.environ.get("DAILY_RUN_LIMIT") or 150)
    st.caption(f"Runs today: {E.runs_today()} / {limit}")

tab_new, tab_dash = st.tabs(["New run", "Dashboard"])

# ---------------------------------------------------------------- new run
with tab_new:
    st.markdown("**Prepared prospects** (real public figures, professional facts only)")
    cols = st.columns(len(E.PRESETS))
    for col, p in zip(cols, E.PRESETS):
        with col:
            st.caption(p["tag"])
            st.button(p["input"]["name"], key="pre_" + p["id"], on_click=fill_preset, args=(p,), use_container_width=True, help=p["blurb"])

    for k in ("f_name", "f_company", "f_role", "f_notes", "f_linkedin"):
        st.session_state.setdefault(k, "")
    st.session_state.setdefault("preset_id", None)

    mode_single = st.radio("Mode", ["One prospect", "A list (CSV)"], horizontal=True, label_visibility="collapsed")
    if mode_single == "One prospect":
        a, b, c = st.columns(3)
        a.text_input("Name", key="f_name")
        b.text_input("Company (optional)", key="f_company")
        c.text_input("Role", key="f_role")
        st.text_input("Notes (where you met, context)", key="f_notes")
        st.text_input("LinkedIn URL (optional, never scraped; just given to the search as a hint)", key="f_linkedin")
        m1, m2 = st.columns([2, 3])
        research_mode = m1.radio("Research", ["Live web research", "Saved snapshot"], horizontal=True,
                                 help="Snapshot only works for the 5 prepared prospects.")
        fresh = m2.checkbox("Fresh research (skip the 24h cache)")
        if st.button("Run", type="primary", disabled=not has_key and research_mode == "Live web research"):
            inp = get_run_input()
            if not inp["name"]:
                st.error("Enter a name.")
            elif E.runs_today() >= limit:
                st.error("Daily run limit reached.")
            else:
                pid = st.session_state.preset_id
                if pid and not any(p["id"] == pid and p["input"]["name"] == inp["name"] for p in E.PRESETS):
                    pid = None
                if research_mode == "Saved snapshot" and not pid:
                    st.error("Snapshots only exist for the prepared prospects. Click one above, or use live research.")
                else:
                    run_one(inp, "live" if research_mode.startswith("Live") else "fixture", pid, fresh)
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

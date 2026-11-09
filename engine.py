"""Hook Engine: pipeline engine (Python port). Gemini + Google Search grounding.

AI judges, code decides: model ratings are inputs; scoring, rules and final status are fixed code.
"""
import datetime as dt
import json
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode

import requests

HERE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR") or HERE / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SELLER = {
    "name": "Zamp",
    "sender": "Anish",
    "pitch": "Zamp builds AI employees: autonomous agents that own a real job end to end. Best known for finance and procurement work: accounts payable and invoice processing, reconciliation and close, KYC/KYB onboarding, chargebacks and disputes, compliance checks.",
    "ask": "a short chat (15 minutes) if it looks relevant",
    # Approved wording, taken from zamp.ai. The writer may use ONE of these ideas, in its own words. Nothing else about Zamp.
    "lines": [
        "Hand the rest to an AI employee that runs it end to end.",
        "Zamp learns the job from the people doing it, your past data and plain-English feedback, then owns it, flags exceptions to the right person, and gets better with every run.",
        "No engineering effort on your side. It works through any interface: browser, desktop, API or SFTP, even legacy terminals, the way a person would.",
        "Full audit trails, end-to-end encryption, independently security reviewed, built for regulated finance teams.",
        "Zamp's experts join your team, find the highest-impact process, and stay until the agent runs in production. Your people learn to manage it.",
        "Wio Bank's CPO on Zamp (from zamp.ai): it lets them securely automate complex operations with precision, scale and full compliance.",
    ],
}

ANCHORS = [
    {"id": "ap", "label": "Invoice & AP load", "desc": "high invoice volume, manual accounts payable, growing payables team, e-invoicing"},
    {"id": "close", "label": "Slow month-end close", "desc": "reconciliation in spreadsheets, multi-bank or multi-entity close, close taking too many days, audit prep"},
    {"id": "onboard", "label": "Onboarding & KYC backlog", "desc": "customer or vendor onboarding, KYC/KYB checks, account opening at scale"},
    {"id": "disputes", "label": "Disputes & chargebacks", "desc": "chargebacks, refunds, payment disputes, claim denials, reversals"},
    {"id": "scale", "label": "Scaling ops without headcount", "desc": "rapid growth or expansion, new markets, hiring many ops/finance roles, doing more with a lean team"},
    {"id": "controls", "label": "Compliance & controls", "desc": "regulatory pressure, tax compliance, audits, internal controls, licences"},
]

GENERIC_TEMPLATES = [
    "Hi {name}, I hope this email finds you well. I wanted to reach out because we help companies like yours streamline their finance operations and save time. Would you be open to a quick call next week?",
    "Hi {name}, I came across your profile and was impressed by your work at {company}. Our AI-powered platform helps finance teams automate manual processes and reduce costs. Do you have 15 minutes this week?",
    "Hey {name}, congrats on the recent growth at {company}! Teams like yours often struggle with manual workflows. We help leading companies boost efficiency with automation. Worth a quick chat?",
    "Hi {name}, quick question: is your finance team spending too much time on repetitive tasks? We help businesses automate end-to-end workflows so your team can focus on strategic work. Open to learning more?",
    "Hi {name}, I know you're busy so I'll keep this short. We're helping finance leaders unlock efficiency, cut errors, and scale operations with AI. Would love to show you a quick demo.",
    "Hello {name}, I'm reaching out to introduce our solution that helps companies transform their back-office operations. Many of our customers see significant ROI within weeks. Can we set up a time to connect?",
    "Hi {name}, as {role} at {company}, you're probably looking for ways to improve efficiency and reduce manual work. Our platform integrates with your existing tools. Let me know if you'd be open to a short conversation.",
    "Hi {name}, I wanted to follow up on my previous note. We help innovative companies like {company} drive operational excellence through intelligent automation. Are you the right person to speak to about this?",
    "Hi {name}, saw that {company} is growing fast — congratulations! Growth often brings operational challenges. We partner with finance teams to automate processes and scale without adding headcount. Interested in a quick call?",
    "Hey {name}, hope you're having a great week. Leading finance teams are using AI to save hundreds of hours every month. I'd love to share how we could help {company} do the same. Free for 15 minutes?",
    "Hi {name}, I'll be brief. Our customers reduce manual effort by up to 80% with our AI agents for finance. I think there could be a strong fit with {company}. Would you be open to exploring this?",
    "Hi {name}, reaching out since you lead finance at {company}. We help teams streamline invoice processing, reconciliation and reporting. Happy to share a few case studies if helpful. Let me know a good time to chat.",
]

STOP = set("a an the and or but if of to in on at for with from by as is are was were be been being it its this that these those i you your we our us they their them he she his her my me so do does did not no yes can could would should will just about into over than then there here what which who whom how when where why all any some more most very also up out only own same too s t don now get got have has had let".split())

STAGES = [
    ("intake", "Intake"), ("research", "Research"),
    ("score", "Pain-space scoring"), ("rules", "Hook & guardrails"), ("draft", "Draft"),
    ("quality", "Quality gates"), ("review", "Human review"),
]

WEIGHTS = {"relevance": 0.4, "freshness": 0.2, "specificity": 0.2, "personFit": 0.2}
HOOK_THRESHOLD = 0.45
MIN_RELEVANCE = 0.35
MIN_HOOK_FRESHNESS = 0.2  # ~ under 7 months old (90-day half-life); older news can't lead an email
BUYER_FIT_MIN = 0.35
QUALITY_BAR = 0.7
CAPTURED_AT = "2026-11-02"

PRESETS = json.loads((HERE / "presets.json").read_text(encoding="utf8"))


def model_default():
    return os.environ.get("GEMINI_MODEL") or "gemini-3.6-flash"


def model_fast():
    return os.environ.get("GEMINI_FAST_MODEL") or "gemini-3.1-flash-lite"


# ---------------------------------------------------------------- Gemini client
BASE = os.environ.get("GEMINI_BASE_URL_OVERRIDE") or "https://generativelanguage.googleapis.com"


class GeminiError(Exception):
    def __init__(self, message, status=None, retryable=False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


_model_fallback = {}


def list_models():
    r = requests.get(f"{BASE}/v1beta/models?pageSize=200", headers={"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "")}, timeout=20)
    r.raise_for_status()
    return [m["name"].split("/")[-1] for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]


def _pick_model(wanted, fast):
    """The configured model id 404'd: choose the closest real one from the account's model list."""
    names = list_models()
    pool = [n for n in names if "flash" in n and not re.search(r"image|tts|live|audio|embed|thinking-exp", n)]
    if fast:
        lite = [n for n in pool if "lite" in n]
        pool = lite or pool
    else:
        pool = [n for n in pool if "lite" not in n] or pool
    pool.sort(reverse=True)  # newest version string first
    return pool[0] if pool else None


_CHAIN = ["gemini-3.6-flash", "gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash"]


def generate(model, body, fast=False, retry429=True):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiError("GEMINI_API_KEY is not set. Add it to .env (local) or app secrets (cloud).", status=401)
    model = _model_fallback.get(model, model)
    tried = {model}
    attempt = 0
    t_start = time.time()
    while True:
        try:
            res = requests.post(f"{BASE}/v1beta/models/{model}:generateContent", json=body,
                                headers={"x-goog-api-key": key}, timeout=60)
            try:
                j = res.json()
            except Exception:
                j = {}
            if not res.ok:
                msg = (j.get("error") or {}).get("message") or res.reason
                if res.status_code == 400 and not re.search("API key", msg, re.I) and "thinkingConfig" in (body.get("generationConfig") or {}):
                    body = {**body, "generationConfig": {k: v for k, v in body["generationConfig"].items() if k != "thinkingConfig"}}
                    continue
                if res.status_code == 404 and model not in _model_fallback.values():
                    alt = _pick_model(model, fast)
                    if alt and alt != model:
                        _model_fallback[model] = alt
                        model = alt
                        continue
                st = 401 if (res.status_code == 403 or (res.status_code == 400 and re.search("API key", msg, re.I))) else res.status_code
                err = GeminiError(f"Gemini API {res.status_code}: {msg}", status=st, retryable=res.status_code == 429 or res.status_code >= 500)
                err.daily = res.status_code == 429 and "PerDay" in json.dumps((j.get("error") or {}).get("details") or [])
                raise err
            cands = j.get("candidates") or []
            if not cands:
                why = (j.get("promptFeedback") or {}).get("blockReason")
                raise GeminiError(f"Gemini blocked the request ({why})." if why else "Gemini returned no answer.", retryable=not why)
            cand = cands[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought"))
            return {"text": text, "grounding": cand.get("groundingMetadata"), "model": model}
        except (requests.Timeout, requests.ConnectionError) as e:
            attempt += 1
            if attempt > 4 or time.time() - t_start > 200:
                raise GeminiError(f"Network error talking to Gemini: {e}", retryable=True)
            pool = [m for m in _CHAIN if ("lite" in m) == ("lite" in model) and m not in tried] or [m for m in _CHAIN if m not in tried]
            if pool:
                model = pool[0]
                tried.add(model)
            time.sleep(2)
        except GeminiError as e:
            attempt += 1
            if not e.retryable or attempt > 10 or time.time() - t_start > 150 or (e.status == 429 and not retry429):
                raise
            if e.status == 503 or (e.status or 0) >= 500 or getattr(e, 'daily', False):  # overloaded / day quota gone: rotate to another model
                pool = [m for m in _CHAIN if ("lite" in m) == ("lite" in model) and m not in tried] or [m for m in _CHAIN if m not in tried]
                if pool:
                    model = pool[0]
                    tried.add(model)
                time.sleep(1.5)
            else:  # 429: per-minute free-tier limit; wait it out
                time.sleep(min(20 * attempt, 45))


def extract_json(text):
    t = (text or "").strip()

    def tp(s):
        try:
            return json.loads(s)
        except Exception:
            return None
    v = tp(t)
    if v is not None:
        return v
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.I)
    if m:
        v = tp(m.group(1).strip())
        if v is not None:
            return v
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        return None
    s, e = min(starts), max(t.rfind("}"), t.rfind("]"))
    if e > s:
        return tp(t[s:e + 1])
    return None


def ai_json(prompt, tier="default", temperature=None):
    model = model_fast() if tier == "quick" else model_default()
    fast = tier == "quick"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 8192, "thinkingConfig": {"thinkingBudget": 0},
                          "seed": 7, **({} if temperature is None else {"temperature": temperature})}}
    r = generate(model, body, fast)
    v = extract_json(r["text"])
    if v is not None:
        return v
    body2 = dict(body, contents=body["contents"] + [
        {"role": "model", "parts": [{"text": r["text"] or "(empty)"}]},
        {"role": "user", "parts": [{"text": "That reply was not valid JSON. Reply again with only the JSON value."}]}])
    v = extract_json(generate(model, body2, fast)["text"])
    if v is None:
        raise GeminiError("Gemini's answer couldn't be read as JSON.")
    return v


def grounded(prompt):
    return generate(model_default(), {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                      "tools": [{"google_search": {}}],
                                      "generationConfig": {"maxOutputTokens": 8192}}, retry429=False)


def resolve_redirect(uri):
    """Grounding chunk links are Google redirect URLs; follow one hop to get the real page."""
    if "/grounding-api-redirect/" not in uri:
        return uri
    try:
        r = requests.get(uri, allow_redirects=False, timeout=6, stream=True)
        loc = r.headers.get("location")
        r.close()
        return loc if loc and re.match(r"https?://", loc) else None
    except Exception:
        return None


# ---------------------------------------------------------------- utils
def clamp01(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def r2(x):
    return round(x * 100) / 100


def tokenize(text):
    t = re.sub(r"\{[a-z]+\}", " ", str(text or "").lower())
    t = re.sub(r"[^a-z0-9%₹$ ]+", " ", t)
    return [w for w in t.split() if len(w) > 2 and w not in STOP]


def tf_vector(tokens):
    v = {}
    for t in tokens:
        v[t] = v.get(t, 0) + 1
    for a, b in zip(tokens, tokens[1:]):
        k = a + "_" + b
        v[k] = v.get(k, 0) + 1
    return v


def cosine(a, b):
    na = sum(x * x for x in a.values())
    nb = sum(y * y for y in b.values())
    dot = sum(x * b[k] for k, x in a.items() if k in b)
    return dot / math.sqrt(na * nb) if na and nb else 0.0


def genericness(draft_text):
    """TF-IDF cosine between the draft and the generic-template corpus. 0 = unlike a template, 1 = a template."""
    docs = [tf_vector(tokenize(t)) for t in GENERIC_TEMPLATES]
    draft = tf_vector(tokenize(draft_text))
    allv = docs + [draft]
    df = {}
    for d in allv:
        for k in d:
            df[k] = df.get(k, 0) + 1
    n = len(allv)

    def weight(d):
        return {k: (1 + math.log(tf)) * math.log((n + 1) / (df[k] + 0.5)) for k, tf in d.items()}
    wd = weight(draft)
    wts = [weight(d) for d in docs]
    mx, nearest = 0.0, 0
    centroid = {}
    for i, w in enumerate(wts):
        c = cosine(wd, w)
        if c > mx:
            mx, nearest = c, i
        for k, x in w.items():
            centroid[k] = centroid.get(k, 0) + x / len(wts)
    return {"max": r2(mx), "centroid": r2(cosine(wd, centroid)), "nearest": nearest}


def freshness(days_ago):
    if days_ago is None:
        return 0.3
    return r2(0.5 ** (max(0, days_ago) / 90))


def days_since(iso, now):
    if not iso:
        return None
    try:
        d = dt.datetime.fromisoformat(iso[:10]).replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return max(0, round((now - d).total_seconds() / 86400))


def iso_days_ago(days, now):
    return None if days is None else (now - dt.timedelta(days=days)).date().isoformat()


def anchor_label(i):
    return next((a["label"] for a in ANCHORS if a["id"] == i), i)


# ---------------------------------------------------------------- scoring + rules (deterministic)
def score_signals(signals, analysis):
    by_id = {str(s.get("id")): s for s in analysis.get("signals", [])}
    out = []
    for s in signals:
        a = by_id.get(str(s["id"]), {})
        pain = {an["id"]: clamp01((a.get("pain") or {}).get(an["id"], 0)) for an in ANCHORS}
        top = max(ANCHORS, key=lambda an: pain[an["id"]])["id"]
        relevance = pain[top]
        fresh = freshness(s.get("daysAgo"))
        spec, pfit = clamp01(a.get("specificity")), clamp01(a.get("person_fit"))
        try:
            valence = max(-1.0, min(1.0, float(a.get("valence") or 0)))
        except (TypeError, ValueError):
            valence = 0.0
        relates = a.get("relates_to") or "current_role"
        score = WEIGHTS["relevance"] * relevance + WEIGHTS["freshness"] * fresh + WEIGHTS["specificity"] * spec + WEIGHTS["personFit"] * pfit
        flags, eligible = [], True
        if s.get("provenance") == "model":
            score *= 0.6
            flags.append("unverified")
        if valence <= -0.35:
            eligible = False
            flags.append("negative — context only")
        if relates == "previous_role":
            eligible = False
            flags.append("previous employer")
        if relates == "other_person":
            eligible = False
            flags.append("different person")
        if relates == "unrelated":
            eligible = False
            flags.append("unrelated")
        out.append({**s, "pain": pain, "topAnchor": top, "relevance": r2(relevance), "freshness": fresh,
                    "specificity": r2(spec), "personFit": r2(pfit), "valence": r2(valence), "relatesTo": relates,
                    "why": a.get("why") or "", "score": r2(score), "eligible": eligible, "flags": flags})
    return out


def choose_hook(scored):
    elig = [s for s in scored if s["eligible"]]
    strengths = {an["id"]: sum(s["score"] * s["pain"][an["id"]] for s in elig) for an in ANCHORS}
    ranked = sorted(({"id": an["id"], "v": r2(strengths[an["id"]])} for an in ANCHORS), key=lambda x: -x["v"])
    def qualifies(s):
        return s["score"] >= HOOK_THRESHOLD and s["relevance"] >= MIN_RELEVANCE and s["freshness"] >= MIN_HOOK_FRESHNESS

    def in_anchor_of(a):
        return sorted(((s, s["score"] * s["pain"][a]) for s in elig), key=lambda x: (not qualifies(x[0]), -x[1], str(x[0]["id"])))
    # lead with the strongest anchor that has a hook good enough to use; a stale headline never beats a fresh qualifying one
    anchor = next((r["id"] for r in ranked if in_anchor_of(r["id"]) and qualifies(in_anchor_of(r["id"])[0][0])), ranked[0]["id"])
    in_anchor = in_anchor_of(anchor)
    hook = in_anchor[0][0] if in_anchor else None
    supporting = [x[0] for x in in_anchor[1:] if x[0]["pain"][anchor] >= 0.4 and x[0]["score"] >= 0.35][:2]
    top, second = ranked[0]["v"], ranked[1]["v"]
    sep = r2((top - second) / top) if top > 0 else 0
    strong = sum(1 for s in elig if s["score"] >= HOOK_THRESHOLD and s["relevance"] >= MIN_RELEVANCE and s["freshness"] >= MIN_HOOK_FRESHNESS)
    return {"anchor": anchor, "hook": hook, "supporting": supporting, "strengths": strengths, "ranked": ranked,
            "separation": sep, "strongCount": strong}


EXEC_RE = re.compile(r"\b(ceo|chief executive|founder|co-?founder|managing director|md|chairman|chairperson|owner|proprietor)\b|(?<!vice )(?<!vice-)\bpresident\b", re.I)


def is_exec_sponsor(role):
    return bool(EXEC_RE.search(role or ""))


def apply_rules(inp, scored, choice, identity, now, persona):
    log = []
    out = {"mode": "standard", "tone": "direct", "status": "draft", "holdUntil": None, "warnings": [], "target": dict(inp)}
    # R1 buyer fit
    if is_exec_sponsor(inp.get("role")):
        out["audience"] = "exec"
        log.append({"rule": "R1 · Buyer fit", "fired": True, "text": f"{inp['name']} is a {inp.get('role')}, an executive sponsor rather than the day-to-day finance owner. The email goes to them directly, stays brief and business-level, and proposes a call with them or a member of their team."})
    elif persona and isinstance(persona.get("buyer_fit"), (int, float)) and persona["buyer_fit"] < BUYER_FIT_MIN:
        out["mode"], out["status"] = "redirect", "redirect"
        out["redirectTo"] = persona.get("better_contact") or "the person who runs finance operations"
        log.append({"rule": "R1 · Buyer fit", "fired": True, "text": f"{inp['name']} looks like {persona.get('seniority') or 'a'} {persona.get('function') or ''} profile (buyer fit {persona['buyer_fit']:.2f}): {persona.get('why') or 'not a finance or operations decision-maker'}. Instead of pitching, the draft asks for an introduction to {out['redirectTo']}."})
    else:
        bf = float((persona or {}).get("buyer_fit") or 0)
        log.append({"rule": "R1 · Buyer fit", "fired": False, "text": f"Buyer fit {bf:.2f} ({persona.get('seniority') or '?'}, {persona.get('function') or '?'}). {persona.get('why') or ''}" if persona else "No persona judgment; assuming the rep picked a buyer."})
    # R2 thin signal
    hook = choice["hook"]
    if not hook or hook["score"] < HOOK_THRESHOLD or hook["relevance"] < MIN_RELEVANCE or hook["freshness"] < MIN_HOOK_FRESHNESS or choice["strongCount"] == 0:
        if out["mode"] != "redirect":
            out["mode"], out["status"] = "fallback", "no_send"
        best = f"{hook['score']:.2f}, relevance {hook['relevance']:.2f}, freshness {hook['freshness']:.2f}" if hook else "none"
        tail = "The referral note stays company-level." if out["mode"] == "redirect" else "Nothing personal is strong enough to lead with, so no email is drafted and the run is marked \"Don't send yet\"."
        log.append({"rule": "R2 · Thin signal", "fired": True, "text": f"No signal is recent (under ~7 months), scores {HOOK_THRESHOLD}+ and points at a pain the seller solves (best: {best}). {tail}"})
    else:
        log.append({"rule": "R2 · Thin signal", "fired": False, "text": f"{choice['strongCount']} signal(s) clear the {HOOK_THRESHOLD} bar and point at a pain. Leading with \"{hook['title']}\" ({hook['score']:.2f})."})
    # R3 negative context
    negs = [s for s in scored if s["valence"] <= -0.35 and s["relatesTo"] not in ("other_person", "previous_role") and (s.get("daysAgo") is None or s["daysAgo"] <= 120)]
    if negs:
        out["tone"] = "careful"
        worst = min(negs, key=lambda s: 999 if s.get("daysAgo") is None else s["daysAgo"])
        out["warnings"].append(f"Recent negative context: \"{worst['title']}\". The draft avoids congratulations and does not mention it.")
        da = worst.get("daysAgo")
        text = f"Negative context found (\"{worst['title']}\", {str(da) + ' days ago' if da is not None else 'date unknown'}). Tone switched to careful: no celebration, no reference to the event."
        if da is not None and da <= 30 and out["status"] != "redirect":
            until = now + dt.timedelta(days=30 - da)
            out["status"], out["holdUntil"] = "hold", until.date().isoformat()
            text += f" It happened within the last 30 days, so the draft is held until {until.strftime('%d %b %Y')}."
        log.append({"rule": "R3 · Negative context", "fired": True, "text": text})
    else:
        log.append({"rule": "R3 · Negative context", "fired": False, "text": "No recent negative events. Normal tone."})
    # R4 corroboration
    if out["mode"] == "fallback":
        log.append({"rule": "R4 · Corroboration", "fired": False, "text": "Not applicable: no hook is being used."})
    elif choice["supporting"]:
        log.append({"rule": "R4 · Corroboration", "fired": True, "text": f"{len(choice['supporting'])} more signal(s) point at the same pain ({anchor_label(choice['anchor'])}). Used as supporting evidence."})
    else:
        log.append({"rule": "R4 · Corroboration", "fired": False, "text": "Only one signal points at this pain. The draft asks rather than asserts."})
        out["warnings"].append("Single-source hook: the draft phrases the pain as a question.")
    if choice["separation"] < 0.15 and out["mode"] != "fallback":
        out["warnings"].append(f"Two pains are nearly tied ({anchor_label(choice['ranked'][0]['id'])} vs {anchor_label(choice['ranked'][1]['id'])}). Reviewer may want to swap the angle.")
    out["log"] = log
    return out


def groundedness(claims, allowed):
    claims = claims if isinstance(claims, list) else []
    if not claims:
        return {"ratio": 1, "ungrounded": []}
    bad = [c for c in claims if not (isinstance(c, dict) and (c.get("signal_id") == "seller" or str(c.get("signal_id")) in allowed))]
    return {"ratio": r2((len(claims) - len(bad)) / len(claims)), "ungrounded": bad}


def personalisation(generic, grounded_ratio, hook_used):
    return r2(0.5 * (1 - clamp01(generic)) + 0.25 * clamp01(grounded_ratio) + 0.25 * (1.0 if hook_used else 0.0))


_BANNED = ["hope this finds you well", "reaching out", "i wanted to", "given that", "streamline", "leverage", "unlock", "game-changer", "cutting-edge",
           "ai-powered", "map one process", "map a process", "[s1", "[s2", "[s3", "[s4", "[s5", "[s6", "[s7", "[s8", "[s9"]


_SUBJECT_BANNED = ["finance operations", "financial operations", "finance function", "streamline", "efficiency", "scaling", "opportunity", "partnership", "quick question", "maintaining", "managing", "support for"]


def hook_words(hook, target):
    """Distinctive tokens from the hook (names, numbers) that a draft built on it should mention."""
    skip = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", f"{target.get('name', '')} {target.get('company', '')}")}
    toks = re.findall(r"\d[\d.,]*\d%?|\d%?|[A-Z][A-Za-z]{3,}", f"{hook['title']} {hook['detail']}") if hook else []
    return {t.lower().rstrip(".,") for t in toks if t.lower() not in skip and t.lower() not in {"the", "this", "that", "with", "from", "have", "will", "after"}}


# ---------------------------------------------------------------- prompts
def analyze_prompt(inp, signals, profile, seller):
    sigs = [{"id": s["id"], "title": s["title"], "detail": s.get("detail"), "source": s.get("source"),
             "when": f"{s['daysAgo']} days ago" if s.get("daysAgo") is not None else "date unknown",
             "about": s.get("subject")} for s in signals]
    L = [
        "You are the analysis step of a B2B sales research pipeline. Judge each research signal about a prospect. Be strict and literal; do not invent facts. Trust the prospect details exactly as the rep entered them.",
        "", "PROSPECT AS ENTERED BY THE SALES REP:",
        json.dumps({"name": inp["name"], "company": inp.get("company") or "(not given)", "role": inp.get("role"), "rep_notes": inp.get("notes") or ""}),
        "",
        ("PROFILE FOUND BY RESEARCH (may be wrong; check against the signals): " + json.dumps(profile)) if profile else "",
        "",
        f"SELLER: {seller['name']}. Its buyers are finance and operations leaders (CFO, VP/Head of Finance, Financial Controller, Head of AP/Procurement/Shared Services, COO, Head of Ops/Compliance).",
        "", "PAIN AXES (what the seller's product solves):",
        "\n".join(f"- {a['id']}: {a['label']} — {a['desc']}" for a in ANCHORS),
        "", "SIGNALS:", json.dumps(sigs, indent=1, ensure_ascii=False), "", "TASKS",
        "1. Persona. From the person's role as entered: function (finance | operations | other), seniority (executive | senior | manager | individual), buyer_fit 0..1 (1 = owns or strongly influences finance/ops tooling decisions; an analyst, associate, engineer or salesperson is below 0.3; a CEO, founder, managing director or president is an executive sponsor and gets 0.6 or more), why (one sentence), and better_contact: the title the rep should ask for instead if buyer_fit is low (empty otherwise).",
        "2. For every signal, score:",
        "   pain: 0..1 per axis — how strongly this signal suggests that pain exists right now at the prospect's company (0 = unrelated).",
        "   valence: -1..1 — is this event bad news for the company or person (layoffs, losses, lawsuits, fraud, regulatory penalties, auditor resignations = negative; routine or planned leadership changes, appointments and reshuffles are neutral, about 0; funding, launches, promotions = positive).",
        "   person_fit: 0..1 — is this about this person's own words, work or function (1) versus general company news (low).",
        "   specificity: 0..1 — would this detail be true of almost any company (0) or is it unique to them (1).",
        '   relates_to: "current_role" (about the prospect or the company the rep entered) | "unrelated" (about a different person who happens to share the name, a different company, or a generic article).',
        "   why: at most 8 words.", "", "Give an entry for EVERY signal id. Reply with only JSON in this exact shape:",
        '{"persona":{"function":"finance","seniority":"executive","buyer_fit":0.9,"why":"","better_contact":""},"signals":[{"id":"s1","pain":{"ap":0,"close":0,"onboard":0,"disputes":0,"scale":0,"controls":0},"valence":0,"person_fit":0,"specificity":0,"relates_to":"current_role","why":""}]}',
    ]
    return "\n".join(L)


def paragraphise(body):
    """Make sure the email reads as short paragraphs: greeting, up to 2 sentences each, a closing line, sign-off."""
    b = (body or "").replace("\r", "").strip()
    sign = ""
    m = re.search(r"\n\s*(warm regards|kind regards|regards|sincerely)\s*,?\s*\n", b, re.I)
    if m:
        sign, b = b[m.start():].strip(), b[:m.start()].strip()
    paras = [x.strip() for x in re.split(r"\n\s*\n", b) if x.strip()]
    if len(paras) >= 3:
        out = paras
    else:
        flat = " ".join(b.split())
        greet = ""
        g = re.match(r"^(dear [^,\n]{1,40},)\s*", flat, re.I)
        if g:
            greet, flat = g.group(1), flat[g.end():]
        sents = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9₹$])", flat)
        out = [greet] if greet else []
        if len(sents) >= 3:
            last = sents[-1]
            rest = sents[:-1]
            half = (len(rest) + 1) // 2
            out += [" ".join(rest[:half]), " ".join(rest[half:]), last] if rest[half:] else [" ".join(rest), last]
        else:
            out += [" ".join(sents)]
    text = "\n\n".join(x for x in out if x)
    return text + ("\n\n" + re.sub(r"\s*\n\s*", "\n", sign) if sign else "")


def draft_prompt(seller, target, decision, choice, scored, critique=""):
    hook = choice["hook"]
    allowed = [s for s in scored if s["eligible"]]
    mode = decision["mode"]
    da = lambda s: f"{s['daysAgo']} days ago" if s.get("daysAgo") is not None else "date unknown"
    first = (target.get("name") or "").split()[0] if target.get("name") else ""
    if mode == "redirect":
        task = (f"THIS PERSON IS NOT THE BUYER. Write a warm, short referral ask (40-70 words): one specific line showing you know something real about {target['company']}"
                f"{' (use: ' + hook['title'] + ')' if hook else ''}, one plain line on what Zamp does, then ask who owns {decision['redirectTo']} and whether they would point you there. No pitch, no call request.")
        words = "40 to 70"
    elif mode == "fallback":
        task = ("SIGNAL QUALITY: thin. There is no strong recent hook. Write a short, honest, human note (40-70 words). You may mention one plain fact (for example their new role) but do not pretend to know more. "
                "One line on what Zamp does, then " + ("close by proposing a short call with them or a member of their team." if decision.get("audience") == "exec" else "one genuine question about how they handle finance work today."))
        words = "40 to 70"
    else:
        sup = " | ".join(f"[{s['id']}] {s['title']} — {s['detail']}" for s in choice["supporting"]) or "none"
        task = f"PAIN IT POINTS AT: {anchor_label(choice['anchor'])}.\nLEAD HOOK [{hook['id']}]: {hook['title']} — {hook['detail']} ({da(hook)})\nSUPPORTING: {sup}"
        words = "60 to 100"
    careful = decision["tone"] == "careful"
    if decision.get("audience") == "exec":
        cta_lines = [
            "- AUDIENCE: this is a CEO or founder, an executive sponsor rather than the day-to-day finance owner. Be especially respectful and brief. Speak to business outcomes they care about (control, speed, the cost of growth) rather than finance mechanics. NEVER ask them to point you to, refer you to or introduce you to someone, and never ask who owns finance or who to speak to.",
            "- CLOSE with this idea, in your own polite words: 'I would love to set up a call with you or your respective team to take this forward, at a time that suits you.' It must be a statement of intent, not a question about who to speak to. Never write 'should I speak with', 'shall I reach out to' or anything that asks them to choose or name someone. Do not ask them to do any work for you. Keep the whole body to 50 to 85 words.",
        ]
    else:
        cta_lines = ["- Close with one courteous, low-pressure question about their world, or politely offer " + seller["ask"] + " at their convenience. Never ask them to 'map a process'."]
    L = [
        "You write first-touch cold emails that busy finance leaders actually answer. A human reviews the draft before anything is sent.", "",
        f"SENDER: {seller['sender']} at {seller['name']}.", f"ABOUT ZAMP: {seller['pitch']}",
        "APPROVED ZAMP LINES (from zamp.ai). Use at most ONE idea from these, in your own plain words, whichever fits the hook best. Do not invent any other claim, number or customer about Zamp:",
        "\n".join(f"- {x}" for x in seller["lines"]), "",
        f"PROSPECT: {target['name']}, {target.get('role')} at {target.get('company') or 'their company'}. Greet them by this exact first name: {first}.", task, "",
        "OTHER FACTS YOU MAY USE:", "\n".join(f"[{s['id']}] {s['title']}" for s in allowed) or "(none)", "", "HOW TO WRITE IT",
        f"- Body {words} words. Plain text, no bullets, no exclamation marks, no emojis. Short paragraphs, 2-3 sentences at most.",
        "- LAYOUT: short paragraphs separated by a blank line (use \n\n inside the JSON body string). Greeting on its own line; paragraph 1 = what you noticed (1-2 sentences); paragraph 2 = what Zamp could take off their plate (1-2 sentences); paragraph 3 = the courteous close (1 sentence); then the sign-off. Never one block of text.",
        "- Shape: open with 'Dear {first},' on its own line. Then (1) one sentence on the specific thing you noticed about them, with the concrete detail (a number, a name, a deal); (2) one or two sentences on what Zamp could take off their plate because of it, using one approved idea in plain, professional words; (3) one courteous question as the close. Finish with 'Warm regards,' and then the sender's first name on its own line.",
        "- Voice: formal, courteous and respectful, the way a thoughtful professional writes to a senior executive they have not met. Complete sentences, no contractions, no slang, no casual shortcuts. Polite phrasing such as 'I would be grateful for', 'would you be open to', 'at your convenience', 'I appreciate you may be busy'. Still be specific and concise; never gushing, never pushy.",
        "- BANNED words and phrases: 'I hope this finds you well', 'reaching out', 'I wanted to', 'Given that', 'likely', 'significant', 'streamline', 'leverage', 'unlock', 'robust', 'journey', 'navigate', 'pressure to', 'in today's', 'AI-powered', 'cutting-edge', 'game-changer', 'map one process', and the stock line 'AI employees that execute end-to-end'.",
        "- Write numbers and currency as digits and symbols (for example ₹733 crore, 69 percent, $100 million), never spelled out in words.",
        "- Do not describe their situation back to them at length or lecture them on what their job involves. Do not say 'you must be' or 'you are probably'. Offer a possibility instead.",
        "- SUBJECT (this matters a lot): 6 to 11 words, formal and polite, sentence case, no emojis, no exclamation marks. Combine (a) one concrete detail from the lead hook (a proper noun, deal name or number) with (b) Zamp's own promotional idea, such as 'AI employee', 'hand the rest to an AI employee', 'an AI employee that runs it end to end', or 'owns the job'. Make it feel like a specific offer of relief, not a generic pitch. BANNED in the subject: 'finance operations', 'financial operations', 'finance function', 'streamline', 'efficiency', 'scaling', 'opportunity', 'partnership', 'quick question', 'maintaining', 'managing', 'support for'. Style examples with made-up names (write NEW ones for THIS prospect): 'Hand Acme's Boreal integration workload to an AI employee', 'An AI employee for the month-end close after Acme's Series C', 'Acme's new Dubai entity: let an AI employee own the invoices'. If the tone is careful, do NOT mention the bad news; use something like 'An AI employee for Acme's finance team, when the time is right'.",
        *cta_lines,
        "- Every fact about the prospect must come from the facts above. No invented metrics, quotes or customers. Never put signal ids like [s2] in the subject or body.",
        ("- TONE: careful. This company has recent bad news. No congratulations, no 'growth', 'exciting' or 'scale' framing, and never mention the bad news. Be gentle, brief and lower-pressure." if careful
         else "- TONE: courteous and professional. If the hook is their new appointment, begin with a brief, sincere congratulations."),
        "" if (choice["supporting"] or mode in ("fallback", "redirect")) else "- Only one source backs this, so phrase the point as a question, not a statement.",
        f"\nA PREVIOUS DRAFT FAILED REVIEW. Fix this: {critique}" if critique else "", "",
        'List every factual claim about the prospect or their company with the id of the signal it comes from. Claims about Zamp use signal_id "seller".',
        'Reply with only JSON: {"subject":"best subject","subject_alternatives":["a different subject","another different subject"],"body":"","claims":[{"text":"","signal_id":"s1"}]}',
    ]
    return "\n".join(l for l in L if l != "")


# ---------------------------------------------------------------- research (Gemini + Google Search)
def norm_url(u):
    try:
        p = urlparse(str(u).strip())
        if not p.hostname:
            return None
        host = re.sub(r"^(www|m)\.", "", p.hostname.lower())
        if host.endswith(".linkedin.com"):
            host = "linkedin.com"
        q = [(k, v) for k, v in parse_qsl(p.query) if not re.match(r"^utm_|^ref$|^src$|^trk", k, re.I)]
        path = p.path.rstrip("/") or "/"
        return {"full": host + path + (("?" + urlencode(q)) if q else ""), "bare": host + path, "host": host}
    except Exception:
        return None


def domain_of(u):
    n = norm_url(u)
    return n["host"] if n else ""


class Pool:
    def __init__(self):
        self.full, self.bare = {}, {}

    def add(self, url, **meta):
        n = norm_url(url)
        if not n:
            return
        v = {**self.full.get(n["full"], {"url": url}), **{k: x for k, x in meta.items() if x}}
        self.full[n["full"]] = v
        self.bare[n["bare"]] = v

    def find(self, url):
        n = norm_url(url)
        return (self.full.get(n["full"]) or self.bare.get(n["bare"])) if n else None


def verify(signals, pool, log):
    today = dt.date.today().isoformat()
    kept, dropped, seen = [], [], set()
    for s in signals or []:
        if not isinstance(s, dict) or not s.get("title"):
            continue
        hit = s.get("url") and pool.find(s["url"])
        if not hit:
            dropped.append(s)
            continue
        key = norm_url(s["url"])["bare"] + "|" + str(s["title"]).lower()[:40]
        if key in seen:
            continue
        seen.add(key)
        d = s.get("date")
        d = (d + "-01" if len(d) == 7 else d) if (d and re.match(r"^\d{4}-\d{2}(-\d{2})?$", d)) else None
        if d and d > today:
            d = None
        kept.append({"title": str(s["title"])[:140], "detail": str(s.get("detail") or "")[:400], "url": s["url"],
                     "source": s.get("source") or domain_of(s["url"]), "date": d, "kind": s.get("kind") or "press",
                     "about": s.get("about") or "company", "subject": s.get("subject") or {}})
    if dropped:
        log("warn", f"Dropped {len(dropped)} claim(s) whose link never came back from a search: " + ", ".join(f'"{str(x.get("title"))[:50]}"' for x in dropped[:3]))
    return kept[:14], dropped


def research_prompt(inp, seller):
    today = dt.date.today().isoformat()
    return f"""Today is {today}. Research this B2B sales prospect for {seller['name']} ({seller['pitch'][:300]}).

Name: {inp['name']}
Role (as the rep entered it, may be stale): {inp.get('role') or 'unknown'}
Company: {inp.get('company') or '(not given)'}
Rep notes: {inp.get('notes') or 'none'}
LinkedIn URL from the rep: {inp.get('linkedin') or 'none'}

Search Google for:
1. Who this person is NOW: current employer and title, start date, previous employer. If several different people share this name, cover each.
2. Their own words: interviews, podcasts, panels, posts or quotes.
3. Company events in the last 12 months: funding, results, acquisitions, expansion, regulation, layoffs or bad news, leadership changes. Include {today[:4]} in news searches.
4. Hiring for finance, accounting, AP, payroll, treasury, compliance or operations roles.

Write your answer as plain lines in exactly this format, nothing else:
PROFILE | current title | current company | since (YYYY-MM or unknown) | previous company | location
F1 | YYYY-MM-DD or unknown | person or company | subject: name; company; role | one factual sentence with names and numbers
F2 | ...
Give 6 to 14 F lines. Each F line must state one fact you found in the search results. Never invent facts. If you found nothing about the person, still give company facts."""


def parse_facts(text):
    profile, facts = None, []
    for l in [x.strip() for x in str(text or "").splitlines() if x.strip()]:
        parts = [p.strip() for p in l.split("|")]
        if re.match(r"^\**PROFILE", parts[0], re.I):
            g = lambda i: parts[i] if len(parts) > i else ""
            profile = {"current_title": g(1), "current_company": g(2), "since": None if re.search("unknown", g(3), re.I) else (g(3) or None),
                       "previous_company": g(4), "location": g(5)}
        elif re.match(r"^\**F\d+", parts[0], re.I) and len(parts) >= 5:
            subj = [x.strip() for x in re.sub(r"^subject:\s*", "", parts[3], flags=re.I).split(";")] + ["", "", ""]
            facts.append({"id": parts[0].replace("*", ""), "date": parts[1] if re.match(r"^\d{4}-\d{2}(-\d{2})?$", parts[1]) else None,
                          "about": "person" if re.search("person", parts[2], re.I) else "company",
                          "subject": {"name": subj[0], "company": subj[1], "role": subj[2]},
                          "fact": " | ".join(parts[4:]), "line": l, "sources": []})
    return profile, facts


# ---------------------------------------------------------------- free research: news RSS feeds (no key, no cost)
import html as _html
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, quote_plus

_UA = {"User-Agent": "Mozilla/5.0 (HookEngine research bot)"}


def _clean_html(t):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", t or ""))).strip()


def _rss(url, kind):
    try:
        r = requests.get(url, headers=_UA, timeout=15)
        root = ET.fromstring(r.content)
    except Exception:
        return []
    out = []
    for it in root.iter("item"):
        g = lambda tag: (it.findtext(tag) or "").strip()
        link = g("link")
        src = it.find("source")
        if kind == "bing":
            m = re.search(r"[?&]url=([^&]+)", link)
            if m:
                link = unquote(m.group(1))
        try:
            date = parsedate_to_datetime(g("pubDate")).date().isoformat()
        except Exception:
            date = None
        title = _clean_html(g("title"))
        publisher = (src.text.strip() if src is not None and src.text else "") or domain_of(link)
        if kind == "google" and publisher and title.endswith(" - " + publisher):
            title = title[: -len(publisher) - 3]
        out.append({"title": title, "url": link, "date": date, "publisher": publisher, "snippet": _clean_html(g("description"))[:300]})
    return out


def news_queries(inp):
    n, c, role = inp["name"], inp.get("company") or "", inp.get("role") or ""
    if c:
        return [f'"{n}" {c}', f"{c} {role or 'CFO'} appointed OR joins OR interview", f"{c} news acquisition OR funding OR results OR layoffs",
                f"{c} stake OR deal OR expansion OR restructuring",
                f"{c} hiring finance OR accounts OR operations"]
    hint = (inp.get("notes") or "")[:60]
    return [f'"{n}" {role}', f'"{n}" {role} appointed OR joins', f'"{n}" {role} {hint}'.strip()]


def research_news(inp, log, seller):
    """Free research: dated headlines from Google News + Bing News feeds, fetched in parallel.
    Each headline becomes a signal as-is (real URL, real date). The analysis step later throws out the irrelevant ones."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    qs = news_queries(inp)
    jobs = []
    for q in qs:
        log("search", f"News feeds: {q}")
        jobs.append((f"https://news.google.com/rss/search?q={quote_plus(q + ' when:1y')}&hl=en-IN&gl=IN&ceid=IN:en", "google", 12))
        jobs.append((f"https://www.bing.com/news/search?q={quote_plus(q)}&format=rss&mkt=en-IN", "bing", 8))
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(lambda j: _rss(j[0], j[1])[: j[2]], jobs))
    # keep only headlines that mention the person's surname or the company
    stop = {"limited", "technologies", "technology", "industries", "india", "private", "pvt", "ltd", "group", "services", "company"}
    toks = {w for w in re.findall(r"[a-z0-9]{4,}", (inp.get("company") or "").lower()) if w not in stop}
    last = re.findall(r"[a-z]{3,}", (inp.get("name") or "").lower())
    if last:
        toks.add(last[-1])
    items, seen = [], set()
    for got in results:
        for it in got:
            k = re.sub(r"\W+", "", it["title"].lower())[:60]
            hay = (it["title"] + " " + it["snippet"]).lower()
            if not it["title"] or k in seen or (toks and not any(t in hay for t in toks)):
                continue
            seen.add(k)
            items.append(it)
    items.sort(key=lambda x: (x["date"] or "", x["title"]), reverse=True)  # stable order whatever feed answers first
    uniq, sets = [], []
    for it in items:
        ts = {w for w in re.findall(r"[a-z0-9]{4,}", it["title"].lower())}
        if any(ts and o and len(ts & o) / len(ts | o) >= 0.5 for o in sets):
            continue
        uniq.append(it)
        sets.append(ts)
    items = uniq[:18]
    log("result", f"→ {len(items)} relevant headline(s) from {sum(len(g) for g in results)} found")
    if not items:
        log("warn", "→ No news headlines found for this prospect.")
    signals = [{"title": it["title"][:140], "detail": (it["snippet"] if it["snippet"] and it["snippet"][:40].lower() != it["title"][:40].lower() else it["title"])[:300],
                "url": it["url"], "source": it["publisher"], "date": it["date"], "kind": "press", "about": "company", "subject": {}} for it in items]
    return {"signals": signals, "profile": None,
            "stats": {"pagesSeen": len(items), "dropped": 0, "webSearches": len(qs), "seconds": round(time.time() - t0, 1)}}


# Google Search grounding needs a paid key. Free feeds by default; set RESEARCH_BACKEND=grounded to use grounding.
_grounding_blocked = {"v": os.environ.get("RESEARCH_BACKEND", "").lower() != "grounded"}


def research_live(inp, log, seller):
    t0 = time.time()
    pool = Pool()
    if _grounding_blocked["v"]:
        return research_news(inp, log, seller)
    log("info", "Gemini is searching Google…")
    try:
        g = grounded(research_prompt(inp, seller))
    except GeminiError as e:
        if e.status in (429, 400, 403, 404):
            _grounding_blocked["v"] = True
            log("warn", "Google Search grounding isn't available on this key (free tier). Using free news feeds instead.")
            return research_news(inp, log, seller)
        raise
    gm = g.get("grounding") or {}
    queries = gm.get("webSearchQueries") or []
    for q in queries:
        log("search", f"Gemini · Google: {q}")
    chunks = [c.get("web") or c.get("retrievedContext") or {} for c in gm.get("groundingChunks") or []]
    chunks = [w for w in chunks if w.get("uri")]
    log("info", f"Resolving {len(chunks)} source link(s)…")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        resolved = list(ex.map(lambda w: resolve_redirect(w["uri"]), chunks))
    sources = [{"url": resolved[i] or w["uri"], "title": w.get("title") or "",
                "domain": domain_of(resolved[i]) if resolved[i] else (w.get("title") or domain_of(w["uri"]))} for i, w in enumerate(chunks)]
    for s in sources:
        pool.add(s["url"], title=s["title"], via="google")
    if sources:
        log("result", "→ %d source(s): %s" % (len(sources), ", ".join(dict.fromkeys(s["domain"] for s in sources[:12]))))
    else:
        log("warn", "→ Google returned no grounded sources.")
    profile, facts = parse_facts(g["text"])
    for sup in gm.get("groundingSupports") or []:
        seg = ((sup.get("segment") or {}).get("text") or "").strip()
        if not seg:
            continue
        f = next((x for x in facts if seg in x["line"] or x["fact"][:60] in seg), None)
        if not f:
            continue
        for idx in sup.get("groundingChunkIndices") or []:
            if idx < len(sources) and sources[idx] not in f["sources"]:
                f["sources"].append(sources[idx])
    log("info", f"{len(facts)} fact(s) found, {sum(1 for f in facts if f['sources'])} tied to a source by Google's grounding.")
    block = "\n\n".join(f"[{f['id']}] date={f['date'] or 'unknown'} about={f['about']} subject={json.dumps(f['subject'])}\n{f['fact']}\nSOURCES: {' , '.join(x['url'] for x in f['sources']) or '(none)'}" for f in facts)
    prompt = f"""Turn this research into signals for a sales pipeline. Today is {dt.date.today().isoformat()}. Prospect: {inp['name']}, {inp.get('role')} at {inp.get('company') or '(company not given)'}.

FACTS FROM GOOGLE SEARCH (each with the URLs Google says support it):
{block or '(none)'}

Rules:
- One signal per distinct fact. Merge duplicates.
- The url of each signal MUST be copied exactly from that fact's SOURCES list. A fact with no source URL is dropped.
- Keep signals about other people with the same name; give them their own subject.
- title: a short headline. detail: one or two factual sentences. date: YYYY-MM-DD or null, never in the future.
Reply with only JSON: {{"signals":[{{"title":"","detail":"","url":"","source":"domain.com","date":null,"kind":"funding|hiring|interview|post|press|company|career|results","about":"person|company","subject":{{"name":"","company":"","role":""}}}}],"note":"one line on coverage"}}"""
    out = ai_json(prompt, tier="quick")
    kept, dropped = verify((out or {}).get("signals") if isinstance(out, dict) else [], pool, log)
    note = (out or {}).get("note") if isinstance(out, dict) else None
    log("info", f"Kept {len(kept)} verified signal(s) from {len(pool.full)} pages seen." + (f" Note: {str(note)[:200]}" if note else ""))
    return {"signals": kept, "profile": {**profile, "confidence": 0.7} if profile and (profile["current_company"] or profile["current_title"]) else None,
            "stats": {"pagesSeen": len(pool.full), "dropped": len(dropped), "webSearches": len(queries), "seconds": round(time.time() - t0, 1)}}


# 24h research cache
CACHE_FILE = DATA_DIR / "research-cache.json"


def _cache_load():
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf8"))
    except Exception:
        return {}


def research(inp, log, seller, fresh=False):
    key = "|".join((inp.get(k) or "").strip().lower() for k in ("name", "company", "role", "notes"))
    hours = float(os.environ.get("RESEARCH_CACHE_HOURS") or 24)
    cache = _cache_load()
    hit = cache.get(key)
    if hit and not fresh and time.time() - hit["at"] < hours * 3600:
        log("info", f"Reused research from {dt.datetime.fromtimestamp(hit['at']).strftime('%d %b %H:%M')} (cached). Tick “Fresh research” to search again.")
        return hit["res"]
    res = research_live(inp, log, seller)
    if len(res["signals"]) >= 3:
        cache[key] = {"at": time.time(), "res": res}
        try:
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf8")
        except Exception:
            pass
    return res


# ---------------------------------------------------------------- pipeline
def new_run(inp, mode, preset_id=None, fresh=False):
    now = time.time()
    return {"id": "run_%d" % (now * 1000), "createdAt": now, "input": dict(inp), "presetId": preset_id, "researchMode": mode,
            "fresh": fresh, "seller": dict(SELLER),
            "stages": {sid: {"status": "pending", "ms": None, "data": None} for sid, _ in STAGES}, "status": "running", "result": None}


def with_dates(signals, now):
    out = []
    for i, s in enumerate(signals):
        da = s["daysAgo"] if "daysAgo" in s else days_since(s.get("date"), now)
        out.append({**s, "id": s.get("id") or f"s{i + 1}", "daysAgo": da, "date": s.get("date") or iso_days_ago(da, now)})
    return out


def _as_analysis(a):
    """The model sometimes wraps its answer in a list or omits keys; always return a dict."""
    if isinstance(a, list):
        a = next((x for x in a if isinstance(x, dict) and ("signals" in x or "identity" in x)), next((x for x in a if isinstance(x, dict)), {}))
    if not isinstance(a, dict):
        return {"identity": {"status": "confirmed", "confidence": 0.3, "clusters": []}, "signals": []}
    if not isinstance(a.get("identity"), dict):
        a["identity"] = {"status": "confirmed", "confidence": 0.5, "clusters": []}
    if not isinstance(a.get("signals"), list):
        a["signals"] = []
    if a.get("persona") is not None and not isinstance(a["persona"], dict):
        a["persona"] = None
    return a


def describe_error(e):
    st = getattr(e, "status", None)
    if st == 401:
        return "The Gemini API key was rejected. Check GEMINI_API_KEY. " + str(e)
    if st == 429:
        return "Gemini is rate-limiting this key (free-tier limits are low). Wait a minute and run again. " + str(e)
    return str(e) or e.__class__.__name__


def run_pipeline(run, on_update):
    """Runs every stage: intake, research, scoring, rules, draft, quality gates."""
    def setst(sid, **patch):
        st = run["stages"][sid]
        if patch.get("status") == "running":
            st["_t0"] = time.time()
        elif patch.get("status") in ("done", "warn", "paused", "error", "skipped") and st.get("_t0"):
            st["ms"] = round((time.time() - st["_t0"]) * 1000)
        st.update(patch)
        on_update(run)

    fixture = run["researchMode"] == "fixture"
    now = dt.datetime.fromisoformat(CAPTURED_AT + "T12:00:00+00:00") if fixture else dt.datetime.now(dt.timezone.utc)
    run["asOf"] = now.date().isoformat()
    run["_now"] = now
    try:
        setst("intake", status="running")
        setst("intake", status="done", data={"input": run["input"]})
        setst("research", status="running", data={"events": []})
        events = run["stages"]["research"]["data"]["events"]

        def log(kind, text):
            events.append({"kind": kind, "text": text})
            on_update(run)
        run["profile"] = None
        if fixture:
            preset = next((p for p in PRESETS if p["id"] == run["presetId"]), None)
            signals = [{**s, "provenance": "fixture"} for s in (preset["signals"] if preset else [])]
            log("info", f"Loaded research snapshot: {len(signals)} signals.")
        else:
            res = research(run["input"], log, run["seller"], fresh=run.get("fresh", False))
            signals = [{**s, "provenance": "web"} for s in res["signals"]]
            run["profile"] = res.get("profile")
            run["researchStats"] = res.get("stats")
        signals = with_dates(signals, now)[:18]
        setst("research", status="done" if signals else "warn", data={"events": events, "signals": signals, "profile": run["profile"], "stats": run.get("researchStats")})
        if not signals:
            log("info", "No headlines found for this person or company.")
            return stop_without_draft(run, "no_info", "No information found about this person or company on the web, so no email was drafted. Check the spelling of the name and company, or try a better-known company.", on_update)

        setst("score", status="running")
        analysis = {"signals": []}
        if signals:
            ap = analyze_prompt(run["input"], signals, run["profile"], run["seller"])
            analysis = _as_analysis(ai_json(ap, temperature=0))
            if not analysis.get("signals"):
                analysis = _as_analysis(ai_json(ap, temperature=0))
        run["persona"] = analysis.get("persona")
        scored = score_signals(signals, analysis)
        n_un = sum(1 for x in scored if x["relatesTo"] == "unrelated")
        if n_un:
            scored = [x for x in scored if x["relatesTo"] != "unrelated"]
            log("info", f"Set aside {n_un} headline(s) that weren't about this prospect or their company.")
        run["_ctx"] = {"scored": scored}
        setst("score", status="done", data={"scored": scored})
        if not scored:
            return stop_without_draft(run, "no_info", "Every headline found was about someone or something else, so there is nothing reliable to write from. No email was drafted.", on_update)
        return finish_pipeline(run, on_update)
    except Exception as e:
        return _fail(run, e, on_update)


_WIKI_UA = {"User-Agent": "HookEngineDemo/1.0 (case-study project; contact via repo owner)"}


def wiki_key_people(company):
    """Key people from the company's Wikipedia infobox (free, no key). Returns [{name, role, url}] or []."""
    try:
        api = "https://en.wikipedia.org/w/api.php"
        q = requests.get(api, params={"action": "query", "list": "search", "srsearch": company + " company", "format": "json", "srlimit": 3}, headers=_WIKI_UA, timeout=10).json()
        toks = {w for w in re.findall(r"[a-z0-9]{4,}", company.lower()) if w not in {"limited", "technologies", "industries", "india", "private", "group"}}
        title = next((h["title"] for h in q["query"]["search"] if not toks or any(t in h["title"].lower() for t in toks)), None)
        if not title:
            return []
        r = requests.get(api, params={"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "format": "json", "titles": title}, headers=_WIKI_UA, timeout=10).json()
        page = next(iter(r["query"]["pages"].values()))
        text = page["revisions"][0]["slots"]["main"]["*"]
        m = re.search(r"\|\s*key_people\s*=\s*(.*?)(?=\n\s*\|[a-z_ ]+=|\n\}\})", text, re.S | re.I)
        if not m:
            return []
        raw = m.group(1)
        raw = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>|<!--.*?-->", "", raw, flags=re.S)
        raw = re.sub(r"\{\{(?:plainlist|flatlist|unbulleted list|ubl|hlist)\s*\|?", "", raw, flags=re.I).replace("}}", "")
        raw = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", raw)
        raw = re.sub(r"<br\s*/?>\s*(?=\()", " ", raw, flags=re.I)
        raw = re.sub(r"<br\s*/?>|\n\s*[*|]|\|", "\n", raw, flags=re.I)
        out = []
        for ln in raw.split("\n"):
            ln = re.sub(r"\{\{[^}]*\}\}|'{2,3}", "", ln).strip(" *|;,")
            mm = re.match(r"^([A-Z][A-Za-z.\-' ]{3,40}?)\s*\(([^)]{2,60})\)", ln)
            if mm:
                out.append({"name": mm.group(1).strip(), "role": mm.group(2).strip(), "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")})
        return out
    except Exception:
        return []


def _indian_names_only(people):
    """Keep only people whose names look Indian (a name-based guess by the model; it can be wrong, so it only filters suggestions)."""
    if not people:
        return people
    listing = "\n".join(f"{i}: {x['name']}" for i, x in enumerate(people))
    try:
        got = ai_json('Which of these names are typically Indian names? Reply with only JSON: {"indian":[indexes]}.\n\n' + listing, tier="quick", temperature=0)
        keep = {int(i) for i in (got.get("indian") if isinstance(got, dict) else got) or []}
    except Exception:
        return people
    return [x for i, x in enumerate(people) if i in keep]


def suggest_contacts(inp, limit=4):
    """When we stop without a draft: other senior people at the same company, taken only from news headlines (never invented).
    One search sweep for finance/leadership news, then one cheap AI call to read names and roles out of the headlines."""
    from concurrent.futures import ThreadPoolExecutor
    company = (inp.get("company") or "").strip()
    if not company:
        return []
    wiki = wiki_key_people(company)
    qs = [f'{company} CFO OR "chief financial officer" OR "head of finance" OR controller',
          f"{company} appoints OR names OR appointed vice president OR head finance OR accounts OR treasury",
          f"{company} CEO OR COO OR founder OR managing director"]
    jobs = []
    for q in qs:
        jobs.append((f"https://news.google.com/rss/search?q={quote_plus(q + ' when:1y')}&hl=en-IN&gl=IN&ceid=IN:en", "google"))
        jobs.append((f"https://www.bing.com/news/search?q={quote_plus(q)}&format=rss&mkt=en-IN", "bing"))
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda j: _rss(j[0], j[1])[:10], jobs))
    stop = {"limited", "technologies", "technology", "industries", "india", "private", "pvt", "ltd", "group", "services", "company"}
    toks = {w for w in re.findall(r"[a-z0-9]{4,}", company.lower()) if w not in stop}
    items, seen = [], set()
    for got in results:
        for it in got:
            k = re.sub(r"\W+", "", it["title"].lower())[:60]
            if not it["title"] or k in seen or (toks and not any(t in (it["title"] + " " + it["snippet"]).lower() for t in toks)):
                continue
            seen.add(k)
            items.append(it)
    items.sort(key=lambda x: (x["date"] or "", x["title"]), reverse=True)
    items = items[:16]
    me = (re.findall(r"[a-z]{3,}", (inp.get("name") or "").lower()) or [""])[-1]
    wiki_out = [{"name": w["name"], "role": w["role"], "headline": "Listed as key person on Wikipedia", "url": w["url"], "date": None, "source": "Wikipedia"}
                for w in wiki if re.findall(r"[A-Za-z]{3,}", w["name"]) and re.findall(r"[A-Za-z]{3,}", w["name"])[-1].lower() != me]
    if not items:
        return wiki_out[:limit]
    lines = "\n".join(f"[{i}] {it['title']} ({it['date'] or 'undated'}) {it['snippet'][:140]}" for i, it in enumerate(items))
    prompt = (f"These are news headlines about {company}. List up to 6 real people named in them who work AT {company} right now in a senior role (finance, operations, "
              f"accounts, treasury, or the CEO/COO/founder). Exclude {inp['name']}. Exclude former employees, people who left or joined another company, investors, analysts, "
              f"customers and anyone whose employer is not {company}. Only use names and roles written in the headline text; never guess. "
              'Reply with only JSON: [{"name":"","role":"","employer":"","id":0}] where employer is the company they work at according to the headline and id is the headline number. Return [] if there are none.\n\n' + lines)
    try:
        got = ai_json(prompt, tier="quick", temperature=0)
    except Exception:
        return wiki_out[:limit]
    got = got if isinstance(got, list) else (got.get("people") if isinstance(got, dict) else []) or []
    out, names = [], {w["name"].lower() for w in wiki_out}
    for g in got:
        try:
            name, role, i = str(g["name"]).strip(), str(g.get("role") or "").strip(), int(g["id"])
            it = items[i]
        except Exception:
            continue
        parts = re.findall(r"[A-Za-z]{3,}", name)
        hay = (it["title"] + " " + it["snippet"]).lower()
        emp = {w for w in re.findall(r"[a-z0-9]{4,}", str(g.get("employer") or "").lower()) if w not in stop}
        if re.search(r"former(ly)?|ex-|leaves?|left|resign|quits?|step(s|ped) down", hay) or (toks and not (emp & toks)):
            continue
        if not parts or parts[-1].lower() not in hay or (me and parts[-1].lower() == me) or name.lower() in names:
            continue
        names.add(name.lower())
        out.append({"name": name, "role": role, "headline": it["title"][:140], "url": it["url"], "date": it["date"], "source": it["publisher"]})
    fin = re.compile(r"financ|cfo|account|treasur|control|audit|operations|coo", re.I)
    out = wiki_out + out
    out = _indian_names_only(out)
    out.sort(key=lambda x: (0 if fin.search(x["role"]) else 1))
    return out[:limit]


def stop_without_draft(run, status, message, on_update, decision=None, slim=None):
    """Research was inadequate: skip the remaining stages and produce NO email."""
    for sid in ("score", "rules", "draft", "quality", "review"):
        st = run["stages"][sid]
        if st["status"] in ("pending", "running"):
            st["status"] = "skipped"
    run["noDraftReason"] = message
    try:
        run["alternatives"] = suggest_contacts(run["input"])
    except Exception:
        run["alternatives"] = []
    run["result"] = {"draft": None, "quality": None, "decision": decision, "choice": slim, "target": (decision or {}).get("target") or run["input"], "status": status} if decision else None
    run["status"] = status
    on_update(run)
    return run


def _fail(run, e, on_update):
    sid = next((k for k, v in run["stages"].items() if v["status"] == "running"), "review")
    run["stages"][sid]["status"] = "error"
    run["stages"][sid]["error"] = describe_error(e)
    run["status"] = "error"
    on_update(run)
    return run


def finish_pipeline(run, on_update):
    def setst(sid, **patch):
        st = run["stages"][sid]
        if patch.get("status") == "running":
            st["_t0"] = time.time()
        elif patch.get("status") in ("done", "warn", "paused", "error") and st.get("_t0"):
            st["ms"] = round((time.time() - st["_t0"]) * 1000)
        st.update(patch)
        on_update(run)
    now = run["_now"]
    scored = run["_ctx"]["scored"]
    setst("rules", status="running")
    choice = choose_hook(scored)
    decision = apply_rules(run["input"], scored, choice, None, now, run.get("persona"))
    slim = {"anchor": choice["anchor"], "hookId": choice["hook"]["id"] if choice["hook"] else None, "supportingIds": [s["id"] for s in choice["supporting"]],
            "strengths": choice["strengths"], "ranked": choice["ranked"], "separation": choice["separation"], "strongCount": choice["strongCount"]}
    setst("rules", status="warn" if decision["mode"] == "fallback" or decision["status"] == "hold" else "done", data={"choice": slim, "decision": decision})

    if decision["mode"] == "fallback":
        return stop_without_draft(run, decision["status"] if decision["status"] in ("hold", "no_send") else "no_send",
                                  "The research did not turn up anything recent and relevant enough to personalise an email, so no draft was written. A generic email would do more harm than good.", on_update, decision, slim)

    allowed = {str(s["id"]) for s in scored if s["eligible"]}
    hook = choice["hook"]
    mode = decision["mode"]
    MAX_ATTEMPTS = 4
    attempt, critique, draft, quality, best = 0, "", None, None, None
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        setst("draft", status="running", data={"attempt": attempt})
        d = ai_json(draft_prompt(run["seller"], decision["target"], decision, choice, scored, critique), temperature=0.2 if attempt == 1 else 0.3)
        draft = {"subject": str(d.get("subject") or ""), "body": str(d.get("body") or ""), "claims": d.get("claims") if isinstance(d.get("claims"), list) else [],
                 "subjectAlternatives": [x.strip() for x in (d.get("subject_alternatives") or []) if isinstance(x, str) and x.strip() and not any(b in x.lower() for b in _SUBJECT_BANNED)][:2]}
        draft["body"] = re.sub(r"(regards,)[ \t]*\n\s*\n+", r"\1\n", draft["body"], flags=re.I)
        sender = run["seller"]["sender"]
        if draft["body"].strip() and not draft["body"].rstrip().endswith(sender):
            draft["body"] = draft["body"].rstrip() + "\n\nWarm regards,\n" + sender
        draft["body"] = paragraphise(draft["body"])
        setst("draft", status="done", data={"attempt": attempt, "draft": draft})
        setst("quality", status="running")
        text = draft["subject"] + " " + draft["body"]
        gen = genericness(text)
        ground = groundedness(draft["claims"], allowed)
        words = len(draft["body"].split())
        used = hook_words(hook, decision["target"]) if (hook and mode not in ("fallback", "redirect")) else set()
        hit = sorted(w for w in used if w in text.lower())
        hook_ok = True if not used else bool(hit)
        banned = [b for b in _BANNED if b in text.lower()]
        if decision.get("audience") == "exec":
            banned += ["exec: " + b for b in ("point me", "pointing me", "refer me", "introduce me", "who owns", "who handles", "should i speak", "should i reach", "shall i") if b in text.lower()]
            if not re.search(r"\bcall\b|\bconversation\b|\bchat\b", draft["body"].lower()) or not re.search(r"\bteams?\b|colleague", draft["body"].lower()):
                banned.append("exec: no call offer with them or their team")
        banned += ["subject: " + b for b in _SUBJECT_BANNED if b in draft["subject"].lower()]
        if not re.search(r"\d|[A-Z][a-z]{3,}", draft["subject"][1:]) :
            banned.append("subject has no concrete detail")
        if not re.search(r"ai employee|owns? the job|end to end|digital worker|digital employee|zamp", draft["subject"].lower()):
            banned.append("subject has no Zamp line")
        p = personalisation(gen["max"], ground["ratio"], hook_ok)
        lo, hi = (30, 90) if mode in ("fallback", "redirect") else (45, 110)
        checks = [
            {"id": "hook", "label": "Uses the hook", "value": 1 if hook_ok else 0, "pass": hook_ok, "detail": ("Mentions " + ", ".join(hit[:4]) + " from the lead signal.") if hit else ("No lead hook in this mode." if not used else "Never mentions the specific detail from the lead signal.")},
            {"id": "generic", "label": "Distance from templates", "value": r2(1 - gen["max"]), "pass": gen["max"] < 0.3, "detail": f"Closest generic template similarity {gen['max']:.2f} (bar: under 0.30)."},
            {"id": "ground", "label": "Grounded claims", "value": ground["ratio"], "pass": ground["ratio"] == 1, "detail": f"{len(ground['ungrounded'])} claim(s) not traceable to a signal." if ground["ungrounded"] else f"{len(draft['claims'])} claim(s), all traced to signals.", "ungrounded": ground["ungrounded"]},
            {"id": "voice", "label": "Plain voice", "value": 0 if banned else 1, "pass": not banned, "detail": ("Cliché or leaked tag: " + ", ".join(banned)) if banned else "No banned phrases."},
            {"id": "length", "label": "Length", "value": words, "pass": lo <= words <= hi, "detail": f"{words} words (target {lo}–{hi})."},
        ]
        passed = p >= QUALITY_BAR and all(c["pass"] for c in checks if c["id"] != "length")
        clean = passed and all(c["pass"] for c in checks)
        quality = {"personalisation": p, "checks": checks, "passed": passed, "attempt": attempt}
        n_fail = sum(1 for c in checks if not c["pass"]) + (0 if p >= QUALITY_BAR else 1)
        if best is None or n_fail < best[0] or (n_fail == best[0] and p > best[1]["quality"]["personalisation"]):
            best = (n_fail, {"draft": draft, "quality": quality})
        if clean:
            setst("quality", status="done", data=quality)
            break
        setst("quality", status="warn", data=quality)
        if attempt >= MAX_ATTEMPTS:
            draft, quality = best[1]["draft"], best[1]["quality"]
            break
        setst("draft", status="running", data={"attempt": attempt + 1, "rewriting": True})
        critique = " ".join(c["label"] + ": " + c["detail"] + ((" Remove: " + "; ".join(str(u.get("text")) for u in c["ungrounded"])) if c.get("ungrounded") else "") for c in checks if not c["pass"]) + (f" Personalisation {p:.2f} is below {QUALITY_BAR}: use the specific detail from the lead signal." if p < QUALITY_BAR else "") + " Fix only those problems; keep it short, human and specific."

    status = {"no_send": "no_send", "redirect": "redirect", "hold": "hold"}.get(decision["status"]) or ("ready" if quality["passed"] else "needs_edit")
    run["result"] = {"draft": draft, "quality": quality, "decision": decision, "choice": slim, "target": decision["target"], "status": status}
    run["status"] = status
    setst("review", status="paused", data={"status": status})
    return run


# ---------------------------------------------------------------- history store
RUNS_FILE = DATA_DIR / "runs.json"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [_clean(x) for x in o]
    return o


def load_runs():
    try:
        return json.loads(RUNS_FILE.read_text(encoding="utf8"))
    except Exception:
        return []


SEED_FILE = HERE / "seed_runs.json"


def load_all_runs():
    """Your own runs first, then the saved sample runs shipped with the app (so the dashboard is never empty)."""
    own = load_runs()
    try:
        seeds = json.loads(SEED_FILE.read_text(encoding="utf8"))
    except Exception:
        seeds = []
    have = {r["id"] for r in own}
    return own + [{**r, "sample": True} for r in seeds if r["id"] not in have]


def save_run(run):
    runs = [r for r in load_runs() if r["id"] != run["id"]]
    runs.insert(0, _clean(run))
    try:
        RUNS_FILE.write_text(json.dumps(runs[:200], ensure_ascii=False, default=str), encoding="utf8")
    except Exception:
        pass


def runs_today():
    start = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return sum(1 for r in load_runs() if r.get("createdAt", 0) >= start)

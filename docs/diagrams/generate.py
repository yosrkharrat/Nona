"""Generate Nona diagrams as both .excalidraw (editable) and .svg (README) from one description."""
import json, random, sys, html

OUT = sys.argv[1]
FONT = 3            # Excalidraw monospace
INK, GRAY, RED = "#1e1e1e", "#868e96", "#e03131"
BLUE, GREEN, YELLOW, LIGHT = "#e7f5ff", "#ebfbee", "#fff3bf", "#f8f9fa"
CW = 0.6            # approx char width factor for monospace
rnd = random.Random(7)


class Scene:
    def __init__(self):
        self.els, self.svg = [], []
        self.byid = {}

    def _base(self, id, type, x, y, w, h, stroke=INK, bg="transparent", dash=False, sw=2):
        e = {"id": id, "type": type, "x": x, "y": y, "width": w, "height": h, "angle": 0,
             "strokeColor": stroke, "backgroundColor": bg, "fillStyle": "solid", "strokeWidth": sw,
             "strokeStyle": "dashed" if dash else "solid", "roughness": 0, "opacity": 100,
             "groupIds": [], "frameId": None, "roundness": None, "seed": rnd.randint(1, 2**31),
             "version": 1, "versionNonce": rnd.randint(1, 2**31), "isDeleted": False,
             "boundElements": None, "updated": 1794700000000, "link": None, "locked": False}
        self.els.append(e); self.byid[id] = e
        return e

    def _text_el(self, id, x, y, text, size, color, align, container=None, w=None, h=None):
        lines = text.split("\n")
        tw = w or max(len(l) for l in lines) * size * CW
        th = h or len(lines) * size * 1.25
        e = self._base(id, "text", x, y, tw, th, stroke=color)
        e.update({"text": text, "originalText": text, "fontSize": size, "fontFamily": FONT,
                  "textAlign": align, "verticalAlign": "middle" if container else "top",
                  "containerId": container, "autoResize": True, "lineHeight": 1.25})
        return e

    def _svg_text(self, x, y, text, size, color, anchor, weight="normal"):
        lines = text.split("\n")
        out = [f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">']
        for i, l in enumerate(lines):
            out.append(f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.25}">{html.escape(l) or " "}</tspan>')
        out.append("</text>")
        self.svg.append("".join(out))

    def box(self, id, x, y, w, h, label="", size=15, stroke=INK, bg="transparent", dash=False,
            label_top=False, color=INK, bold=False):
        e = self._base(id, "rectangle", x, y, w, h, stroke=stroke, bg=bg, dash=dash)
        e["roundness"] = {"type": 3}
        fill = "none" if bg == "transparent" else bg
        da = ' stroke-dasharray="8 6"' if dash else ""
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"{da}/>')
        if label:
            if label_top:   # free text pinned to the top-left of the box (box acts as a group frame)
                self.text(id + "_t", x + 12, y + 10, label, size, color, bold=bold)
            else:
                n = len(label.split("\n"))
                th = n * size * 1.25
                t = self._text_el(id + "_t", x + 6, y + (h - th) / 2, label, size, color, "center",
                                  container=id, w=w - 12, h=th)
                e["boundElements"] = [{"type": "text", "id": t["id"]}]
                self._svg_text(x + w / 2, y + (h - th) / 2 + size * 0.95, label, size, color, "middle",
                               "bold" if bold else "normal")
        return e

    def text(self, id, x, y, text, size=14, color=INK, bold=False):
        self._text_el(id, x, y, text, size, color, "left")
        self._svg_text(x, y + size * 0.95, text, size, color, "start", "bold" if bold else "normal")

    def arrow(self, id, pts, start=None, end=None, color=INK, dash=False, both=False):
        x0, y0 = pts[0]
        rel = [[px - x0, py - y0] for px, py in pts]
        xs, ys = [p[0] for p in rel], [p[1] for p in rel]
        e = self._base(id, "arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys), stroke=color, dash=dash)
        e.update({"points": rel, "lastCommittedPoint": None, "elbowed": False,
                  "startArrowhead": "arrow" if both else None, "endArrowhead": "arrow",
                  "startBinding": {"elementId": start, "focus": 0, "gap": 4} if start else None,
                  "endBinding": {"elementId": end, "focus": 0, "gap": 4} if end else None})
        for ref in (start, end):
            if ref:
                tgt = self.byid[ref]
                tgt["boundElements"] = (tgt["boundElements"] or []) + [{"type": "arrow", "id": id}]
        mk = "red" if color == RED else ("gray" if color == GRAY else "ink")
        da = ' stroke-dasharray="7 5"' if dash else ""
        ms = f' marker-start="url(#s-{mk})"' if both else ""
        p = " ".join(f"{a},{b}" for a, b in pts)
        self.svg.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2"{da}{ms} marker-end="url(#e-{mk})"/>')

    def save(self, name, W, H):
        for i, e in enumerate(self.els):
            e["index"] = f"a{i:04d}"
        scene = {"type": "excalidraw", "version": 2, "source": "nona/gen_diagrams.py",
                 "elements": self.els, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}
        json.dump(scene, open(f"{OUT}/{name}.excalidraw", "w"), indent=1, ensure_ascii=False)
        markers = "".join(
            f'<marker id="e-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            f'<marker id="s-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for k, c in (("ink", INK), ("red", RED), ("gray", GRAY)))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
               f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">'
               f'<defs>{markers}</defs><rect width="{W}" height="{H}" fill="#ffffff"/>'
               + "".join(self.svg) + "</svg>")
        open(f"{OUT}/{name}.svg", "w").write(svg)


# ───────────────────────── 1. architecture (container view) ─────────────────────────
s = Scene()
s.text("title", 40, 28, "Nona: container view (v0.1)", 28, bold=True)
s.text("sub", 40, 70, "one local process · anonymize on the way out · restore on the way back · nothing stored after the response", 15, GRAY)

s.box("trust", 40, 120, 1300, 760, "TRUSTED ZONE: your machine / VPC (real data allowed here)", 14, GRAY, dash=True, label_top=True, color=GRAY)

# client
s.box("client", 70, 400, 210, 110, "client app\n(OpenAI SDK)\nbase_url → :8787", 15, bg=LIGHT)
s.text("client_n", 70, 525, "the only code change.\ngets real names back.", 13, GRAY)

# nona process
s.box("proc", 310, 160, 1000, 700, "nona serve: one Python process (FastAPI + uvicorn)", 15, INK, label_top=True, bold=True)
s.box("api", 330, 230, 140, 470, "API layer\n\n/v1/chat/\ncompletions\n\n/v1/models\n/healthz\n/metrics\n\n/v1/nona/*\n(inspect,\n localhost)", 13, bg=LIGHT)

# request lane
s.text("lane1", 490, 205, "① request path: anonymize", 14, "#1c7ed6", bold=True)
xs = [490, 622, 754, 886, 1018]
labels = ["walk\npayload", "normalize\nArabizi 3→ع\noffset map", "detect\n(union,\nrecall first)", "resolve\n+ link\nmentions", "pseudonymize\nTYPE_n by\n1st appear."]
ids = ["walk", "norm", "detect", "link", "pseudo"]
for x, lab, i in zip(xs, labels, ids):
    s.box(i, x, 235, 112, 80, lab, 13, bg=BLUE)
s.arrow("a_api_walk", [(470, 275), (490, 275)], "api", "walk")
for a, b, x in zip(ids, ids[1:], xs):
    s.arrow(f"a_{a}_{b}", [(x + 112, 275), (x + 132, 275)], a, b)

# detectors group
s.box("dets", 490, 350, 370, 190, "detectors (in-process, offline)", 13, GRAY, label_top=True, color=GRAY)
s.box("d_regex", 502, 380, 168, 68, "regex + checksum\nemail · phone\nIBAN · CIN(ctx)", 12, bg=LIGHT)
s.box("d_gliner", 680, 380, 168, 68, "GLiNER multi\nCPU · weights\npulled once", 12, bg=LIGHT)
s.box("d_gaz", 502, 458, 168, 68, "gazetteers +\ncontext cues\nm3a · khalti · M.", 12, bg=LIGHT)
s.box("d_cache", 680, 458, 168, 68, "detection cache\nLRU+TTL, RAM\nspans only", 12, bg=LIGHT)
s.arrow("a_detect_dets", [(810, 315), (810, 350)], "detect", "dets", both=True)

# mapping (the sensitive state)
s.box("map", 886, 365, 244, 150, "MAPPING (per request)\nPERSON_1 ↔ Ahmed Ben Salem\nPHONE_1  ↔ +216 22 123 456\n\nRAM only · dropped after\nthe response (ADR-0002)", 12, "#f08c00", bg=YELLOW)
s.arrow("a_pseudo_map", [(1074, 315), (1074, 365)], "pseudo", "map")
s.text("t_write", 1080, 330, "write", 12, GRAY)

# upstream client
s.box("up", 1160, 230, 130, 470, "upstream\nclient\n\nhttpx async\npooled\nSSE stream", 13, bg=LIGHT)
s.arrow("a_pseudo_up", [(1130, 275), (1160, 275)], "pseudo", "up")

# response lane
s.text("lane2", 490, 590, "② response path: restore", 14, "#2f9e44", bold=True)
s.box("restore", 490, 620, 640, 80,
      "restore: streaming state machine\nholds back ≤16 chars that could start a placeholder\nswaps only placeholders issued in this request · JSON-escapes tool_call args", 12, bg=GREEN)
s.arrow("a_up_restore", [(1160, 660), (1130, 660)], "up", "restore")
s.arrow("a_restore_api", [(490, 660), (470, 660)], "restore", "api")
s.arrow("a_map_restore", [(1010, 515), (1010, 620)], "map", "restore")
s.text("t_read", 1016, 560, "read", 12, GRAY)

# client <-> api
s.arrow("a_client_api", [(280, 430), (330, 430)], "client", "api")
s.arrow("a_api_client", [(330, 480), (280, 480)], "api", "client")
s.text("t_real", 70, 360, "→ real data in\n← restored answer out", 13)

# side notes inside process
s.box("fail", 330, 725, 500, 115, "FAIL CLOSED (ADR-0003)\ndetector error or timeout → HTTP 503\nnona_detector_unavailable\nnothing is forwarded upstream", 13, RED, color=RED)
s.box("side", 850, 725, 440, 115, "side channels\nconfig: nona.yaml (policies, allow/deny)\nlogs + /metrics: counts & timings only,\nnever bodies (FR-19 CI grep test)", 13, GRAY, color=INK)

# outside: upstream LLM
s.box("llm", 1480, 400, 220, 110, "upstream LLM\nOpenAI · OpenRouter\nOllama", 15, bg=LIGHT)
s.text("llm_n", 1480, 300, "UNTRUSTED\nhonest but curious:\nlogs, training,\nsubprocessors", 13, GRAY)
s.arrow("a_up_llm", [(1290, 430), (1480, 430)], "up", "llm")
s.arrow("a_llm_up", [(1480, 480), (1290, 480)], "llm", "up")
s.text("t_out", 1352, 398, "placeholders\nonly", 12)
s.text("t_in", 1352, 488, "PERSON_1\nin deltas", 12)
s.text("leak", 1360, 545, "◆ LEAK RATE is measured on\n  this outbound wire: what\n  actually leaves, not what\n  the detector thinks it caught", 13, RED, bold=True)
s.save("nona-architecture", 1740, 910)

# ───────────────────────── 2. benchmark harness ─────────────────────────
b = Scene()
b.text("title", 40, 28, "Nona: benchmark harness (nona bench)", 28, bold=True)
b.text("sub", 40, 70, "every system gets the same messages · we score what reaches the upstream, not what the detector reports", 15, GRAY)

b.box("data", 40, 170, 270, 170, "synthetic dataset\nFR · EN · AR · Darija\nArabizi · mixed\n\ndev: tune thresholds\ntest: FROZEN, hash in repo", 13, bg=YELLOW, color=INK)
b.text("data_n", 40, 350, "templates + generators →\ngold labels by construction\n+ ~50 hand-written hard cases\nNO real or client data", 12, GRAY)

b.box("sys", 380, 130, 280, 290, "systems under test", 13, GRAY, label_top=True, color=GRAY)
b.box("s_nona", 400, 165, 240, 60, "Nona\n(release config)", 13, bg=BLUE)
b.box("s_pres", 400, 240, 240, 75, "Presidio\n① default (EN)\n② FR spaCy + patterns", 13, bg=LIGHT)
b.box("s_llmg", 400, 330, 240, 60, "LLM Guard\nAnonymize, default", 13, bg=LIGHT)
b.arrow("a_data_sys", [(310, 255), (380, 255)], "data", "sys")

b.box("cap", 740, 170, 250, 120, "capturing mock upstream\nrecords the exact payload\nthat would leave the\nmachine (the wire)", 13, RED, color=INK)
b.arrow("a_sys_cap", [(660, 230), (740, 230)], "sys", "cap")

b.box("util", 740, 340, 250, 120, "utility run\nsame tasks, raw vs\nthrough each system\nOllama in CI · hosted\nmodel for README", 13, bg=GREEN)
b.arrow("a_sys_util", [(660, 380), (740, 380)], "sys", "util")

b.box("metrics", 1070, 130, 300, 330,
      "metrics\n\nLEAK RATE (headline)\npartial leak rate\nover-masking rate\nrestoration failure\nutility loss ± 95% CI\n(extract · reply · sum.)\noverhead p50 / p95\n\nper language × type", 13, bg=LIGHT)
b.arrow("a_cap_m", [(990, 230), (1070, 230)], "cap", "metrics")
b.arrow("a_util_m", [(990, 400), (1070, 400)], "util", "metrics")
b.arrow("a_gold", [(175, 420), (175, 520), (1220, 520), (1220, 460)], "data", "metrics", GRAY, dash=True)
b.text("t_gold", 480, 527, "gold spans (exact, from the generator)", 12, GRAY)

b.box("report", 1450, 200, 230, 190, "report.py\n\nresults table →\nREADME\n\nfooter: model revs,\ndataset hash,\nknown failures", 13, bg=YELLOW, color=INK)
b.arrow("a_m_rep", [(1370, 295), (1450, 295)], "metrics", "report")
b.text("rule", 1070, 560, "rule: no headline number in the README before\none clean run on the frozen test split (NFR-11)", 13, RED, bold=True)
b.save("nona-benchmark", 1720, 620)
print("ok")

"""Original cartoon rapper avatars, built from parts.

Every look is a combination of interchangeable parts (hair, eyewear, mouth/grill,
outfit, chain, colours) chosen from a seed, so characters are original by
construction. They are not drawings of any real person. Users re-roll a seed
or pick parts in the UI; the chosen parts are stored on the character.

``render_svg(look)`` returns a standalone SVG (thick outlines, flat colours) that
the editor can animate (the ``.sl-head`` group bobs to the beat).
"""
from __future__ import annotations

import random

SKIN = ["#f6d1b1", "#e8b48f", "#c98e63", "#a8714b", "#8a5634", "#6b3f24", "#4f2d19"]
HAIR_COLORS = {"black": "#17151a", "brown": "#4a2f1f", "platinum": "#efe7cf", "pink": "#ff6fb5", "green": "#5df27a",
               "blue": "#4f8cff", "red": "#ff4747", "purple": "#a26bff", "orange": "#ff9a3c"}
HAIR = ["twists", "dreads", "spikes", "buzz", "curls", "puffs", "mohawk", "fringe", "beanie", "bucket", "durag", "mask"]
EYES = ["round", "sleepy", "shades", "visor", "star_shades", "goggles"]
MOUTH = ["smirk", "grin_silver", "grin_gold", "grin_iced", "flat", "open"]
OUTFIT = ["hoodie", "puffer", "tee", "jersey", "trench"]
CHAIN = ["none", "gold", "silver", "iced", "beads"]
EXTRAS = ["none", "earrings", "bandaid", "headphones", "piercing", "stars"]
OUTFIT_COLORS = ["#111318", "#f2f2f2", "#ff3b6b", "#4f8cff", "#5df27a", "#ffd23f", "#a26bff", "#ff8a3d", "#2a2d36", "#c7a8ff"]
BG = [("#ff5d73", "#ffb86b"), ("#7275ff", "#3ee0c5"), ("#1d1f2b", "#5a3df0"), ("#c8ff3d", "#3dff9e"), ("#ff6fb5", "#8a5cff"),
      ("#0d1b2a", "#1b998b"), ("#2b2d42", "#ef233c"), ("#f7f7f7", "#c9c9c9"), ("#14213d", "#fca311")]

INK = "#111111"
SW = 5  # outline width


def random_look(seed: int | str, mood: str | None = None) -> dict:
    rng = random.Random(str(seed))
    dark = mood == "dark"
    look = {
        "seed": str(seed),
        "skin": rng.choice(SKIN),
        "hair": rng.choice(HAIR),
        "hair_color": rng.choice(list(HAIR_COLORS)),
        "eyes": rng.choice(EYES),
        "mouth": rng.choice(MOUTH),
        "outfit": rng.choice(OUTFIT),
        "outfit_color": rng.choice(OUTFIT_COLORS[:2] + OUTFIT_COLORS[8:] if dark else OUTFIT_COLORS),
        "chain": rng.choice(CHAIN),
        "extra": rng.choice(EXTRAS),
        "bg": rng.randrange(len(BG)) if not dark else rng.choice([2, 5, 6, 8]),
    }
    return look


def validate_look(look: dict) -> dict:
    allowed = {"skin": None, "hair": HAIR, "hair_color": list(HAIR_COLORS), "eyes": EYES, "mouth": MOUTH, "outfit": OUTFIT,
               "chain": CHAIN, "extra": EXTRAS}
    out = random_look(look.get("seed", "x"))
    for k, opts in allowed.items():
        if k in look:
            if k == "skin":
                if look[k] not in SKIN:
                    raise ValueError("unknown skin tone")
            elif look[k] not in opts:
                raise ValueError(f"unknown {k}: {look[k]}")
            out[k] = look[k]
    if "outfit_color" in look:
        if look["outfit_color"] not in OUTFIT_COLORS:
            raise ValueError("unknown outfit colour")
        out["outfit_color"] = look["outfit_color"]
    if "bg" in look:
        out["bg"] = int(look["bg"]) % len(BG)
    return out


def options() -> dict:
    return {"skin": SKIN, "hair": HAIR, "hair_color": list(HAIR_COLORS), "eyes": EYES, "mouth": MOUTH, "outfit": OUTFIT,
            "outfit_color": OUTFIT_COLORS, "chain": CHAIN, "extra": EXTRAS, "bg": len(BG)}


def _shade(hex_color: str, f: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, int(c * f))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hair(style: str, c: str, skin: str) -> tuple[str, str]:
    """Returns (behind_head, over_head) SVG fragments. Head centre (128,112), radius ~62."""
    o = f'stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round"'
    if style == "twists":
        tw = "".join(f'<path d="M{x} 70 q-6 -26 4 -40 q10 14 4 40 z" fill="{c}" {o}/>' for x in range(78, 184, 15))
        return "", f'<path d="M66 100 Q70 48 128 46 Q186 48 190 100 Q160 74 128 74 Q96 74 66 100Z" fill="{c}" {o}/>' + tw
    if style == "dreads":
        back = "".join(f'<path d="M{x} 80 Q{x - 4} 150 {x + 2} {176 + (x % 3) * 8}" stroke="{c}" stroke-width="13" stroke-linecap="round" fill="none"/>'
                       f'<path d="M{x} 80 Q{x - 4} 150 {x + 2} {176 + (x % 3) * 8}" stroke="{INK}" stroke-width="2" fill="none" opacity=".35"/>'
                       for x in (64, 78, 178, 192))
        top = f'<path d="M64 108 Q60 44 128 40 Q196 44 192 108 Q170 70 128 70 Q86 70 64 108Z" fill="{c}" {o}/>'
        top += "".join(f'<path d="M{x} 52 q4 30 -2 52" stroke="{INK}" stroke-width="2.5" fill="none" opacity=".45"/>' for x in range(86, 176, 14))
        return back, top
    if style == "spikes":
        pts = " ".join(f"{x},{y}" for x, y in [(62, 104), (66, 60), (84, 74), (92, 28), (110, 62), (128, 18), (146, 62), (164, 28), (172, 74), (190, 60), (194, 104), (166, 80), (90, 80)])
        return "", f'<polygon points="{pts}" fill="{c}" {o}/>'
    if style == "buzz":
        return "", f'<path d="M68 96 Q72 52 128 50 Q184 52 188 96 Q160 74 128 72 Q96 74 68 96Z" fill="{c}" opacity=".9" {o}/>'
    if style == "curls":
        cs = "".join(f'<circle cx="{x}" cy="{y}" r="15" fill="{c}" {o}/>' for x, y in
                     [(76, 80), (92, 56), (114, 44), (140, 44), (164, 56), (180, 80), (102, 66), (128, 58), (154, 66)])
        return "", cs
    if style == "puffs":
        return "", (f'<circle cx="74" cy="56" r="26" fill="{c}" {o}/><circle cx="182" cy="56" r="26" fill="{c}" {o}/>'
                    f'<path d="M68 98 Q72 54 128 52 Q184 54 188 98 Q160 76 128 76 Q96 76 68 98Z" fill="{c}" {o}/>')
    if style == "mohawk":
        return "", (f'<path d="M110 74 L104 14 L120 30 L128 4 L136 30 L152 14 L146 74 Z" fill="{c}" {o}/>'
                    f'<path d="M70 96 Q74 60 128 58 Q182 60 186 96" fill="none" stroke="{_shade(skin, .8)}" stroke-width="3" stroke-dasharray="2 5"/>')
    if style == "fringe":
        return (f'<path d="M62 100 Q56 150 70 168 L84 150 Z" fill="{c}" {o}/><path d="M194 100 Q200 150 186 168 L172 150 Z" fill="{c}" {o}/>',
                f'<path d="M60 112 Q58 42 128 40 Q198 42 196 112 L178 84 L160 104 L150 80 L128 108 L112 80 L96 104 L80 86 Z" fill="{c}" {o}/>')
    if style == "beanie":
        return "", (f'<path d="M62 92 Q62 30 128 28 Q194 30 194 92 Z" fill="{c}" {o}/>'
                    f'<rect x="58" y="80" width="140" height="24" rx="10" fill="{_shade(c, .8)}" {o}/>'
                    f'<circle cx="128" cy="24" r="10" fill="{_shade(c, 1.2)}" {o}/>')
    if style == "bucket":
        return "", (f'<path d="M78 82 Q80 34 128 32 Q176 34 178 82 Z" fill="{c}" {o}/>'
                    f'<path d="M44 92 Q128 66 212 92 L200 104 Q128 82 56 104 Z" fill="{_shade(c, .85)}" {o}/>')
    if style == "durag":
        return (f'<path d="M180 98 Q214 126 206 176 L190 172 Q196 132 172 108 Z" fill="{c}" {o}/>',
                f'<path d="M64 104 Q60 40 128 38 Q196 40 192 104 Q170 84 128 84 Q86 84 64 104Z" fill="{c}" {o}/>'
                f'<path d="M78 66 Q128 50 178 66" fill="none" stroke="{_shade(c, 1.3)}" stroke-width="4" opacity=".6"/>')
    if style == "mask":  # ski mask
        return "", (f'<path d="M62 120 Q58 38 128 36 Q198 38 194 120 Q196 168 128 176 Q60 168 62 120Z" fill="{c}" {o}/>'
                    f'<rect x="80" y="92" width="96" height="34" rx="16" fill="{_shade("#f6d1b1", 1)}" opacity="0"/>')
    return "", ""


def _eyes(kind: str, skin: str) -> str:
    o = f'stroke="{INK}" stroke-width="{SW}"'
    if kind == "round":
        return (f'<circle cx="104" cy="112" r="14" fill="#fff" {o}/><circle cx="152" cy="112" r="14" fill="#fff" {o}/>'
                f'<circle cx="107" cy="114" r="6" fill="{INK}"/><circle cx="155" cy="114" r="6" fill="{INK}"/>'
                f'<circle cx="109" cy="111" r="2" fill="#fff"/><circle cx="157" cy="111" r="2" fill="#fff"/>')
    if kind == "sleepy":
        return (f'<path d="M90 112 Q104 104 118 112" fill="none" {o} stroke-linecap="round"/>'
                f'<path d="M138 112 Q152 104 166 112" fill="none" {o} stroke-linecap="round"/>'
                f'<path d="M92 114 Q104 124 116 114" fill="#fff" {o}/><path d="M140 114 Q152 124 164 114" fill="#fff" {o}/>')
    if kind == "shades":
        return (f'<rect x="80" y="98" width="44" height="28" rx="8" fill="{INK}"/><rect x="132" y="98" width="44" height="28" rx="8" fill="{INK}"/>'
                f'<path d="M124 108 h8" {o}/><path d="M88 104 l10 0" stroke="#fff" stroke-width="3" opacity=".6"/>')
    if kind == "visor":
        return (f'<path d="M74 100 Q128 88 182 100 L178 124 Q128 116 78 124 Z" fill="url(#visor)" {o}/>'
                f'<path d="M86 104 Q128 96 170 104" stroke="#fff" stroke-width="3" opacity=".5" fill="none"/>')
    if kind == "star_shades":
        def star(cx, cy):
            import math
            pts = []
            for i in range(10):
                r = 18 if i % 2 == 0 else 8
                a = -math.pi / 2 + i * math.pi / 5
                pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
            return f'<polygon points="{" ".join(pts)}" fill="#ffd23f" {o}/>'
        return star(104, 112) + star(152, 112) + f'<path d="M120 110 h16" {o}/>'
    if kind == "goggles":
        return (f'<rect x="66" y="96" width="124" height="34" rx="16" fill="#2a2d36" {o}/>'
                f'<circle cx="104" cy="113" r="12" fill="#3ee0c5" {o}/><circle cx="152" cy="113" r="12" fill="#3ee0c5" {o}/>')
    return ""


def _mouth(kind: str) -> str:
    o = f'stroke="{INK}" stroke-width="{SW}" stroke-linecap="round"'
    grill = {"grin_silver": "#d9dde3", "grin_gold": "#ffcc33", "grin_iced": "#e6f7ff"}
    if kind in grill:
        teeth = "".join(f'<line x1="{x}" y1="148" x2="{x}" y2="160" stroke="{INK}" stroke-width="2"/>' for x in range(116, 142, 7))
        sparkle = '<path d="M142 144 l3 -6 l3 6 l-3 6 z" fill="#fff"/>' if kind == "grin_iced" else ""
        return f'<path d="M106 146 Q128 140 150 146 Q146 166 128 166 Q110 166 106 146Z" fill="{grill[kind]}" {o}/>{teeth}{sparkle}'
    if kind == "smirk":
        return f'<path d="M112 152 Q132 160 150 146" fill="none" {o}/>'
    if kind == "flat":
        return f'<path d="M114 154 L142 154" {o}/>'
    if kind == "open":
        return f'<path d="M112 146 Q128 172 144 146 Z" fill="#5a1d2b" {o}/>'
    return ""


def _outfit(kind: str, c: str) -> str:
    o = f'stroke="{INK}" stroke-width="{SW}" stroke-linejoin="round"'
    dark = _shade(c, .78)
    if kind == "hoodie":
        return (f'<path d="M26 256 Q30 192 82 178 L174 178 Q226 192 230 256 Z" fill="{c}" {o}/>'
                f'<path d="M82 178 Q128 214 174 178" fill="{dark}" {o}/>'
                f'<path d="M112 200 L110 232 M144 200 L146 232" stroke="{INK}" stroke-width="3"/>')
    if kind == "puffer":
        bands = "".join(f'<path d="M30 {y} Q128 {y - 10} 226 {y}" fill="none" stroke="{INK}" stroke-width="3" opacity=".5"/>' for y in (204, 226, 246))
        return (f'<path d="M20 256 Q22 186 80 172 L176 172 Q234 186 236 256 Z" fill="{c}" {o}/>{bands}'
                f'<path d="M84 172 Q128 196 172 172 L168 160 Q128 178 88 160 Z" fill="{dark}" {o}/>'
                f'<path d="M128 186 L128 256" stroke="{INK}" stroke-width="3"/>')
    if kind == "tee":
        return f'<path d="M34 256 Q38 196 90 182 Q128 200 166 182 Q218 196 222 256 Z" fill="{c}" {o}/>'
    if kind == "jersey":
        return (f'<path d="M34 256 Q38 196 90 182 Q128 210 166 182 Q218 196 222 256 Z" fill="{c}" {o}/>'
                f'<path d="M90 182 Q128 214 166 182" fill="none" stroke="#fff" stroke-width="6"/>'
                f'<text x="128" y="246" text-anchor="middle" font-family="Arial Black,Arial" font-weight="900" font-size="30" fill="#fff" stroke="{INK}" stroke-width="2">{(sum(ord(ch) for ch in c) % 90) + 1}</text>')
    if kind == "trench":
        return (f'<path d="M22 256 Q24 192 82 176 L174 176 Q232 192 234 256 Z" fill="{c}" {o}/>'
                f'<path d="M82 176 L118 256 M174 176 L138 256" stroke="{INK}" stroke-width="4"/>'
                f'<path d="M82 176 L104 206 L128 186 L152 206 L174 176" fill="{dark}" {o}/>')
    return ""


def _chain(kind: str) -> str:
    if kind == "none":
        return ""
    col = {"gold": "#ffcc33", "silver": "#d9dde3", "iced": "#e6f7ff", "beads": "#ff6fb5"}[kind]
    links = "".join(f'<circle cx="{128 + 46 * __import__("math").cos(a / 10)}" cy="{186 + 34 * __import__("math").sin(a / 10)}" r="{6 if kind == "beads" else 4.5}" fill="{col}" stroke="{INK}" stroke-width="2"/>'
                    for a in range(2, 30, 2))
    pendant = (f'<path d="M128 214 l14 14 l-14 14 l-14 -14 z" fill="{col}" stroke="{INK}" stroke-width="3"/>'
               + ('<path d="M124 224 l4 -6 l4 6 l-4 6 z" fill="#fff"/>' if kind == "iced" else "")) if kind != "beads" else ""
    return links + pendant


def _extra(kind: str) -> str:
    o = f'stroke="{INK}" stroke-width="3"'
    if kind == "earrings":
        return f'<circle cx="66" cy="136" r="5" fill="#e6f7ff" {o}/><circle cx="190" cy="136" r="5" fill="#e6f7ff" {o}/>'
    if kind == "bandaid":
        return f'<rect x="150" y="128" width="26" height="11" rx="4" transform="rotate(-20 163 133)" fill="#f2c094" {o}/>'
    if kind == "headphones":
        return (f'<path d="M58 112 Q58 34 128 34 Q198 34 198 112" fill="none" stroke="{INK}" stroke-width="10"/>'
                f'<rect x="46" y="98" width="24" height="40" rx="10" fill="#7275ff" stroke="{INK}" stroke-width="{SW}"/>'
                f'<rect x="186" y="98" width="24" height="40" rx="10" fill="#7275ff" stroke="{INK}" stroke-width="{SW}"/>')
    if kind == "piercing":
        return f'<circle cx="160" cy="92" r="3.5" fill="#d9dde3" {o}/><circle cx="114" cy="140" r="3" fill="#d9dde3" {o}/>'
    if kind == "stars":
        return f'<path d="M88 136 l3 -7 l3 7 l7 3 l-7 3 l-3 7 l-3 -7 l-7 -3 z" fill="{INK}"/>'
    return ""


def render_svg(look: dict, size: int = 256, title: str = "") -> str:
    look = validate_look(look)
    skin = look["skin"]
    hc = HAIR_COLORS[look["hair_color"]]
    bg1, bg2 = BG[look["bg"]]
    uid = f'a{abs(hash(str(sorted(look.items())))) % 10**8}'
    behind, over = _hair(look["hair"], hc, skin)
    o = f'stroke="{INK}" stroke-width="{SW}"'
    face_hidden = look["hair"] == "mask"
    head = (f'<circle cx="64" cy="122" r="14" fill="{skin}" {o}/><circle cx="192" cy="122" r="14" fill="{skin}" {o}/>'
            f'<path d="M66 116 Q64 52 128 50 Q192 52 190 116 Q190 176 128 180 Q66 176 66 116Z" fill="{skin}" {o}/>')
    if face_hidden:  # ski mask: eye and mouth holes over the mask
        face = (f'<rect x="82" y="96" width="92" height="34" rx="16" fill="{skin}" {o}/>{_eyes(look["eyes"] if look["eyes"] in ("round", "sleepy") else "round", skin)}'
                f'<rect x="108" y="140" width="40" height="24" rx="11" fill="{skin}" {o}/>{_mouth("flat")}')
    else:
        face = (f'<path d="M92 92 Q104 86 116 92 M140 92 Q152 86 164 92" fill="none" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
                f'{_eyes(look["eyes"], skin)}<path d="M126 124 Q122 136 130 138" fill="none" stroke="{_shade(skin, .7)}" stroke-width="4" stroke-linecap="round"/>'
                f'{_mouth(look["mouth"])}')
    neck = f'<rect x="108" y="164" width="40" height="26" fill="{_shade(skin, .9)}" {o}/>'
    label = f"<title>{title}</title>" if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="{size}" height="{size}" role="img">{label}'
            f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{bg1}"/><stop offset="1" stop-color="{bg2}"/></linearGradient>'
            f'<linearGradient id="visor" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ff3b6b"/><stop offset=".5" stop-color="#7275ff"/><stop offset="1" stop-color="#3ee0c5"/></linearGradient></defs>'
            f'<rect width="256" height="256" rx="40" fill="url(#{uid})"/>'
            f'<g class="sl-body">{_outfit(look["outfit"], look["outfit_color"])}{_chain(look["chain"])}</g>'
            f'<g class="sl-head">{behind}{neck if False else ""}{head}{over if look["hair"] != "mask" else ""}'
            f'{over if look["hair"] == "mask" else ""}{face}{_extra(look["extra"]) if not face_hidden else ""}</g></svg>')

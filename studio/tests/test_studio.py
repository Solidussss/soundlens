"""SoundLens Studio tests. Run: python -m pytest studio/tests -q

They render real (short) audio, so they take ~1 minute on a laptop CPU.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from studio import SAMPLE_RATE, analysis, characters, composer, dsp, synth  # noqa: E402
from studio.actions import ActionError, apply_actions  # noqa: E402
from studio.orchestrator import interpret  # noqa: E402
from studio.project import section_beats, track_by_ref, track_notes_absolute, validate_project  # noqa: E402
from studio.render import Renderer  # noqa: E402

SR = SAMPLE_RATE


@pytest.fixture(scope="module")
def vanta():
    return characters.get("vanta")


@pytest.fixture(scope="module")
def project(vanta):
    return composer.generate_project(vanta, {"duration_s": 16}, seed=5)


@pytest.fixture(scope="module")
def renderer(tmp_path_factory):
    return Renderer(tmp_path_factory.mktemp("cache"))


# -- DSP ------------------------------------------------------------------------

def _sine(f=110.0, s=1.0, amp=0.5):
    t = np.arange(int(SR * s)) / SR
    return dsp.as_stereo(amp * np.sin(2 * np.pi * f * t))


def test_limiter_respects_ceiling():
    y = dsp.limiter(_sine(amp=3.0), SR, ceiling_db=-1.0)
    assert np.max(np.abs(y)) <= dsp.db_to_lin(-1.0) + 1e-9


def test_distortion_adds_harmonics_and_keeps_sub():
    x = _sine(55.0, 2.0, 0.6)
    clean = analysis.distortion_profile(x)
    dirty = analysis.distortion_profile(dsp.distortion(x, SR, drive_db=24, mode="hard"))
    assert dirty["thd_pct"] > clean["thd_pct"] + 10
    kept = dsp.distortion(x, SR, drive_db=24, mode="hard", keep_sub_hz=80)
    spec = np.abs(np.fft.rfft(kept[0]))
    f = np.fft.rfftfreq(kept.shape[1], 1 / SR)
    assert spec[np.argmin(np.abs(f - 55))] > 0.3 * np.abs(np.fft.rfft(x[0]))[np.argmin(np.abs(f - 55))]


def test_eq_boosts_band():
    x = _sine(1000.0, 1.0, 0.1)
    y = dsp.eq(x, SR, [{"type": "peak", "freq": 1000, "gain_db": 6, "q": 1}])
    assert 20 * np.log10(np.std(y[:, SR // 2:]) / np.std(x[:, SR // 2:])) == pytest.approx(6.0, abs=0.3)


def test_effect_validation_rejects_bad_params():
    with pytest.raises(ValueError):
        dsp.validate_effect({"type": "distortion", "params": {"drive_db": 999}})
    with pytest.raises(ValueError):
        dsp.validate_effect({"type": "nope"})


def test_every_preset_renders_finite_and_deterministic():
    for name in synth.PRESETS:
        inst = synth.preset(name)
        notes = [{"t": 0, "d": 1, "p": 60 if inst["engine"] not in ("808",) else 33, "v": 0.8}]
        a = synth.render_notes(inst, notes, 120, SR, SR, seed=3)
        b = synth.render_notes(inst, notes, 120, SR, SR, seed=3)
        assert np.isfinite(a).all() and np.array_equal(a, b), name


def test_note_sound_independent_of_other_notes():
    inst = synth.preset("supersaw_lead")
    n1 = {"t": 0.0, "d": 1.0, "p": 60, "v": 0.8}
    alone = synth.render_notes(inst, [n1], 120, SR * 2, SR, seed=1)
    with_other = synth.render_notes(inst, [{"t": 0.0, "d": 0.5, "p": 40, "v": 0.8}, n1], 120, SR * 2, SR, seed=1)
    other = synth.render_notes(inst, [{"t": 0.0, "d": 0.5, "p": 40, "v": 0.8}], 120, SR * 2, SR, seed=1)
    assert np.allclose(with_other - other, alone, atol=1e-12)


# -- characters + composer ------------------------------------------------------------

def test_characters_valid_and_original():
    chars = characters.load_all()
    assert len(chars) >= 5
    for c in chars.values():
        assert "<svg" in characters.public_view(c)["avatar_svg"]


def test_generated_project_is_valid(project):
    validate_project(project)
    roles = {t["role"] for t in project["tracks"]}
    assert {"kick", "808", "melody", "hat"} <= roles
    for t in project["tracks"]:
        assert track_notes_absolute(t), f"{t['name']} has no notes"  # no empty placeholder tracks
    assert project["params"]["scale"] in ("minor", "phrygian", "harmonic_minor")


def test_808_follows_kick(project):
    kicks = {round(n["t"], 4) for n in track_notes_absolute(track_by_ref(project, "kick"))}
    bass = [n for n in track_notes_absolute(track_by_ref(project, "808")) if not n.get("slide")]
    assert bass and all(round(n["t"], 4) in kicks for n in bass)


def test_generation_is_reproducible(vanta):
    a = composer.generate_project(vanta, {"duration_s": 16}, seed=9)
    b = composer.generate_project(vanta, {"duration_s": 16}, seed=9)
    assert [track_notes_absolute(t) for t in a["tracks"]] == [track_notes_absolute(t) for t in b["tracks"]]


def test_requested_params_are_honoured(vanta):
    p = composer.generate_project(vanta, {"duration_s": 20, "bpm": 150, "key": "D", "scale": "dorian"}, seed=1)
    assert (p["params"]["bpm"], p["params"]["key"], p["params"]["scale"]) == (150, "D", "dorian")


# -- edits preserve other tracks ------------------------------------------------------------

def test_instrument_swap_changes_only_that_track(project, vanta, renderer):
    new, info = apply_actions(project, [{"op": "set_instrument", "track": "melody", "preset": "bell_bright"}], vanta)
    assert [c["name"] for c in info["diff"]["sound_changed"]] == ["Melody"]
    for t in project["tracks"]:
        if t["role"] != "melody":
            assert np.array_equal(renderer.stem(project, t), renderer.stem(new, track_by_ref(new, t["id"])))
    assert track_notes_absolute(track_by_ref(new, "melody")) == track_notes_absolute(track_by_ref(project, "melody"))


def test_section_regeneration_keeps_rest_of_track(project, vanta):
    sec = next(s for s in project["sections"] if s["name"] == "verse")
    lo, hi = section_beats(project, sec)
    new, _ = apply_actions(project, [{"op": "regenerate_part", "track": "melody", "sections": [sec["id"]]}], vanta)
    outside = lambda p: [n for n in track_notes_absolute(track_by_ref(p, "melody")) if not lo <= n["t"] < hi]
    assert outside(new) == outside(project)


def test_failed_batch_changes_nothing(project, vanta):
    before = json.dumps(project, sort_keys=True)
    with pytest.raises(ActionError):
        apply_actions(project, [{"op": "set_mixer", "track": "808", "delta_db": 3}, {"op": "set_instrument", "track": "808", "preset": "nope"}], vanta)
    assert json.dumps(project, sort_keys=True) == before


def test_clear_section_and_duplicate_section(project, vanta):
    hook = next(s for s in project["sections"] if s["name"] == "hook")
    lo, hi = section_beats(project, hook)
    cleared, _ = apply_actions(project, [{"op": "clear_section", "track": "drums", "sections": [hook["id"]]}], vanta)
    for role in ("kick", "hat"):
        assert not [n for n in track_notes_absolute(track_by_ref(cleared, role)) if lo <= n["t"] < hi]
    dup, _ = apply_actions(project, [{"op": "duplicate_section", "section": hook["id"]}], vanta)
    assert len(dup["sections"]) == len(project["sections"]) + 1
    validate_project(dup)


def test_tempo_change_keeps_notes(project, vanta):
    new, _ = apply_actions(project, [{"op": "set_tempo", "bpm": 120}], vanta)
    assert track_notes_absolute(track_by_ref(new, "melody")) == track_notes_absolute(track_by_ref(project, "melody"))


# -- orchestrator ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,op", [
    ("Make the 808 more distorted.", "update_effect"),
    ("Make the hi-hats faster.", "regenerate_part"),
    ("Add a counter melody.", "add_part"),
    ("Regenerate the second verse.", "regenerate_part"),
    ("Make the entire song faster", "set_tempo"),
    ("Change the melody to a bell", "set_instrument"),
    ("make the bass louder", "set_mixer"),
    ("take the drums out of verse 2", "clear_section"),
])
def test_orchestrator_examples(project, vanta, text, op):
    r = interpret(project, text, {}, character=vanta, allow_llm=False)
    assert r["actions"] and r["actions"][0]["op"] == op, r


def test_orchestrator_refuses_what_it_cannot_do(project, vanta):
    for text in ("Make the vocals more aggressive", "replace the melody with a guitar", "replace the piano with a pad"):
        r = interpret(project, text, {}, character=vanta, allow_llm=False)
        assert not r["actions"] and r["problems"], text


def test_compound_command(project, vanta):
    r = interpret(project, "make the 808 louder and more distorted", {}, character=vanta, allow_llm=False)
    assert [a["op"] for a in r["actions"]][:2] == ["set_mixer", "update_effect"]


# -- render / export / QA ----------------------------------------------------------------------

def test_mix_export_and_qa(project, renderer, tmp_path):
    r = renderer
    calib = r.calibrate_master(project)
    assert abs(calib["lufs"] - calib["target"]) < 1.5
    path = r.export_wav(project, tmp_path / "mix.wav")
    from studio.render import song_samples
    integ = analysis.export_integrity(path, SR, 2, song_samples(project) / SR)
    assert integ["ok"]
    zpath = r.export_stems_zip(project, tmp_path / "stems.zip")
    names = zipfile.ZipFile(zpath).namelist()
    assert sum(n.endswith(".wav") for n in names) == len(project["tracks"]) and any(n.endswith(".mid") for n in names)
    kick = track_by_ref(project, "kick")
    ta = analysis.timing_alignment(r.stem(project, kick), [n["t"] for n in track_notes_absolute(kick)], project["params"]["bpm"])
    assert ta["matched_pct"] >= 95 and ta["median_abs_error_ms"] < 10


def test_midi_roundtrip(project):
    from studio.midi_io import midi_bytes_to_notes, track_to_midi_bytes
    t = track_by_ref(project, "melody")
    notes, bpm = midi_bytes_to_notes(track_to_midi_bytes(project, t))
    assert bpm == pytest.approx(project["params"]["bpm"], rel=1e-3)
    assert [n["p"] for n in notes] == [n["p"] for n in track_notes_absolute(t)]


# -- HTTP API (dev auth) ------------------------------------------------------------------------

def test_api_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDLENS_STUDIO_DEV_AUTH", "1")
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from studio.api import create_router

    def no_users(authorization):
        raise HTTPException(401, "login")

    app = FastAPI()
    app.include_router(create_router(no_users, tmp_path))
    c = TestClient(app)
    assert c.get("/studio-api/characters").json()["characters"]
    job = c.post("/studio-api/projects", json={"character_id": "lowtide", "duration_s": 12, "seed": 3}).json()
    import time
    for _ in range(300):
        j = c.get(f"/studio-api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    pid = j["result"]["project_id"]
    view = c.get(f"/studio-api/projects/{pid}").json()
    tid = view["project"]["tracks"][0]["id"]
    wav = c.get(f"/studio-api/projects/{pid}/stems/{tid}")
    assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
    r = c.post(f"/studio-api/projects/{pid}/command", json={"text": "make the 808 more distorted"}).json()
    assert r["diff"]["sound_changed"][0]["name"] == "808"
    assert c.post(f"/studio-api/projects/{pid}/command", json={"text": "make the vocals louder"}).json()["interpretation"]["problems"]
    assert c.post(f"/studio-api/projects/{pid}/undo").json()["can_redo"] is True
    assert c.post(f"/studio-api/projects/{pid}/edit", json={"actions": [{"op": "set_mixer", "track": "808", "pan": 9}]}).status_code == 400
    ex = c.post(f"/studio-api/projects/{pid}/export", json={"format": "wav"}).json()
    for _ in range(300):
        j = c.get(f"/studio-api/jobs/{ex['id']}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    assert c.get(f"/studio-api/projects/{pid}/exports/{j['result']['file']}").status_code == 200
    assert c.get(f"/studio-api/projects/{pid}/exports/..%2F..%2Fproject.json").status_code == 404
    assert c.get("/studio-api/projects/../../etc").status_code in (404, 400)


# -- reference collections ----------------------------------------------------------------------

def test_reference_collection_permissions(tmp_path):
    import soundfile as sf
    from studio import references
    f = tmp_path / "ref.wav"
    t = np.arange(SR * 4) / SR
    sf.write(str(f), np.stack([0.3 * np.sin(2 * np.pi * 55 * t)] * 2, axis=1), SR)
    col = references.create("refs", "Test refs", "SoundLens (own)", "owned", ["analysis"], [f], root=tmp_path / "r")
    assert col["items"][0]["sha256"]
    prof = references.analyze("refs", root=tmp_path / "r")
    assert prof["n"] == 1 and prof["lufs"] is not None
    with pytest.raises(PermissionError):
        references.require_use(references.load("refs", root=tmp_path / "r"), "training")
    with pytest.raises(ValueError):
        references.create("x", "x", "", "", ["analysis"], [f], root=tmp_path / "r")


def test_cache_prune_evicts_and_rerenders(tmp_path, project):
    from studio.store import ProjectStore
    st = ProjectStore(tmp_path)
    r = Renderer(st.cache)
    t = track_by_ref(project, "kick")
    first = r.stem(project, t)
    assert st.prune_cache(max_bytes=0)["removed"] >= 1
    assert np.array_equal(Renderer(st.cache).stem(project, t), first)


# -- learning characters from songs ---------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_song(tmp_path_factory):
    """A rendered beat with known tempo/feel stands in for an uploaded song."""
    c = characters.get("mercury_halo")
    p = composer.generate_project(c, {"duration_s": 40}, seed=4)
    import soundfile as sf
    r = Renderer(tmp_path_factory.mktemp("refcache"))
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    sf.write(str(path), np.clip(r.mix(p), -1, 1).T, SR)
    return path, p


def test_learner_recovers_tempo_and_feel(reference_song):
    from studio import learn
    path, p = reference_song
    t = learn.analyze_track(path)
    assert abs(t["bpm"] - p["params"]["bpm"]) / p["params"]["bpm"] < 0.01
    assert t["feel"] == "half"
    assert t["low_bars"] and t["mix"]["lufs"] < 0


def test_learned_character_generates(reference_song, tmp_path):
    from studio import learn
    path, _ = reference_song
    style = learn.aggregate([learn.analyze_track(path)])
    c = learn.character_from_style(style, "c_test", "Test Kid", {"look": {"seed": "t"}})
    assert c["drums"]["kick"]["pattern"] == "learned" and c["drums"]["kick"]["patterns"]
    p = composer.generate_project(c, {"duration_s": 16}, seed=1)
    validate_project(p)
    assert p["params"]["bpm"] == pytest.approx(style["bpm"]["median"], abs=1)
    assert "<svg" in characters.avatar_svg(c)


def test_pattern_library_clusters_noise():
    from studio import learn
    bars = ["x.......x.x....."] * 8 + ["x.......x.x...x."] * 2 + ["x...x...x.x....."]
    lib = learn._pattern_library(bars)
    assert lib[0]["pattern"] == "x.......x.x....." and lib[0]["share"] > 0.9


def test_avatars_are_valid_svg_and_validated():
    from studio import avatars
    import xml.dom.minidom
    for i in range(12):
        xml.dom.minidom.parseString(avatars.render_svg(avatars.random_look(i)))
    with pytest.raises(ValueError):
        avatars.validate_look({"hair": "not-a-hair"})


def test_api_learn_character_flow(tmp_path, monkeypatch, reference_song):
    monkeypatch.setenv("SOUNDLENS_STUDIO_DEV_AUTH", "1")
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from studio.api import create_router
    import time

    def no_users(authorization):
        raise HTTPException(401, "login")

    app = FastAPI()
    app.include_router(create_router(no_users, tmp_path))
    c = TestClient(app)
    path, _ = reference_song
    with open(path, "rb") as fh:
        r = c.post("/studio-api/characters/learn", data={"name": "Test Kid", "rights": ""}, files=[("files", ("a.wav", fh, "audio/wav"))])
    assert r.status_code == 400  # rights must be confirmed
    with open(path, "rb") as fh:
        job = c.post("/studio-api/characters/learn", data={"name": "Test Kid", "rights": "yes"}, files=[("files", ("a.wav", fh, "audio/wav"))]).json()
    for _ in range(900):
        j = c.get(f"/studio-api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert j["status"] == "done", j
    cid = j["result"]["character_id"]
    assert not list((tmp_path / "studio" / "learning").glob("*/*.wav")), "uploaded songs must be deleted"
    listed = {x["id"]: x for x in c.get("/studio-api/characters").json()["characters"]}
    assert cid in listed and listed[cid]["custom"] and listed[cid]["learned_summary"]["songs"] == 1
    gen = c.post("/studio-api/projects", json={"character_id": cid, "duration_s": 12, "seed": 2}).json()
    for _ in range(300):
        g = c.get(f"/studio-api/jobs/{gen['id']}").json()
        if g["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert g["status"] == "done", g
    assert c.post(f"/studio-api/characters/{cid}", json={"name": "Renamed"}).json()["name"] == "Renamed"
    assert c.delete(f"/studio-api/characters/{cid}").json()["ok"]

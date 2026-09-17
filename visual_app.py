from __future__ import annotations

from pathlib import Path

from fastapi.responses import HTMLResponse

import production_app as production
import soundlens_3d as event_model

app = production.app
_INDEX_PATH = Path(__file__).with_name("index.html")


# ---------------------------------------------------------------------------
# Musical-event semantics v3.1
# ---------------------------------------------------------------------------
# The v3 detector already fuses aligned measurements. This light semantic pass
# keeps its confidence/importance math intact while making event names describe
# the measured direction of change instead of falling back to generic labels.

_original_fuse_candidates = event_model._fuse_candidates


def _directional_fuse_candidates(candidates, times, sections, duration):
    # Give structural regions honest positional names without pretending we can
    # identify verse/hook/chorus from acoustic novelty alone.
    section_count = len(sections)
    for i, section in enumerate(sections):
        if section_count <= 1:
            label = "Full track"
        elif i == 0:
            label = "Opening section"
        elif i == section_count - 1:
            label = "Closing section"
        else:
            label = f"Middle section {i}"
        section["label"] = label

    events = _original_fuse_candidates(candidates, times, sections, duration)

    for event in events:
        evidence = event.get("evidence") or []
        summaries = " | ".join(str(item.get("summary") or "").lower() for item in evidence)
        kinds = set(event.get("evidence_types") or [])
        direction = None

        energy_up = "energy rises" in summaries
        energy_down = "energy drops" in summaries
        bass_up = "low end enters" in summaries
        bass_down = "low end pulls back" in summaries
        stereo_up = "stereo opens" in summaries
        stereo_down = "stereo narrows" in summaries
        brighter = "tone brighter" in summaries
        darker = "tone darker" in summaries

        if "clip" in kinds:
            event["title"] = "Digital ceiling hit"
            direction = "ceiling_hit"
        elif "section" in kinds:
            if bass_up and (energy_up or "transient" in kinds):
                event["title"] = "Section lift + low-end entrance"
                direction = "lift"
            elif energy_up:
                event["title"] = "Section lift"
                direction = "lift"
            elif bass_up:
                event["title"] = "Low-end entrance at section change"
                direction = "lift"
            elif energy_down or bass_down:
                event["title"] = "Section drop"
                direction = "drop"
            elif stereo_up:
                event["title"] = "Section opens wider"
                direction = "widen"
            elif stereo_down:
                event["title"] = "Section narrows"
                direction = "narrow"
            elif brighter:
                event["title"] = "Section brightens"
                direction = "brighten"
            elif darker:
                event["title"] = "Section darkens"
                direction = "darken"
            else:
                event["title"] = "Structural transition"
                direction = "transition"
        elif "bass" in kinds and "energy" in kinds:
            if bass_up and energy_up:
                event["title"] = "Low-end + energy lift"
                direction = "lift"
            elif bass_down and energy_down:
                event["title"] = "Low-end + energy pullback"
                direction = "drop"
            elif bass_up:
                event["title"] = "Low-end entrance"
                direction = "lift"
            elif bass_down:
                event["title"] = "Low-end pullback"
                direction = "drop"
        elif "bass" in kinds:
            if bass_up:
                event["title"] = "Low-end entrance"
                direction = "lift"
            elif bass_down:
                event["title"] = "Low-end pullback"
                direction = "drop"
        elif "energy" in kinds:
            if energy_up:
                event["title"] = "Energy lift"
                direction = "lift"
            elif energy_down:
                event["title"] = "Energy drop"
                direction = "drop"
        elif "stereo" in kinds:
            if stereo_up:
                event["title"] = "Stereo field opens"
                direction = "widen"
            elif stereo_down:
                event["title"] = "Stereo field narrows"
                direction = "narrow"
        elif "brightness" in kinds:
            if brighter:
                event["title"] = "Tonal brightening"
                direction = "brighten"
            elif darker:
                event["title"] = "Tonal darkening"
                direction = "darken"

        if direction:
            event["direction"] = direction

    return events


event_model._fuse_candidates = _directional_fuse_candidates


def _patch_visual_hierarchy(html: str) -> str:
    old_norm = """function bandNorm(key){const vals=visual.slices.map(s=>valAt(s,key));const max=Math.max(...vals,.0001);return vals.map(v=>Math.min(1,v/max));}"""
    new_norm = """function bandNorm(key){const vals=visual.slices.map(s=>Math.max(0,valAt(s,key)));if(!vals.length)return[];const sorted=[...vals].sort((a,b)=>a-b);const p95=sorted[Math.min(sorted.length-1,Math.floor((sorted.length-1)*.95))]||.0001;const p20=sorted[Math.min(sorted.length-1,Math.floor((sorted.length-1)*.20))]||0;const span=Math.max(.0001,p95-p20);return vals.map(v=>Math.max(0,Math.min(1,(v-p20)/span)));}"""

    old_sculpture = """function buildSculpture(){clearGroup(sculpture);clearGroup(pinGroup);pinMeshes=[];const slices=visual.slices;if(!slices?.length)return;trackLength=Math.min(70,Math.max(38,visual.duration/3.7));
 const bandKeys=['sub','bass','low_mid','mid','high','air'];const norms=Object.fromEntries(bandKeys.map(k=>[k,bandNorm(k)]));
 bandKeys.forEach((band,bi)=>{const points=[];const mirror=[];for(let i=0;i<slices.length;i++){const s=slices[i],x=(i/(slices.length-1)-.5)*trackLength;const e=norms[band][i];const energy=Number(s.energy||0),stereo=Number(s.stereo||0),trans=Number(s.transient||0);const baseY=(bi-2.5)*1.35;const lift=e*3.1+energy*.7;const z=(bi-2.5)*1.2 + stereo*(bi%2?2.1:-2.1);points.push(new THREE.Vector3(x,baseY+lift,z));mirror.push(new THREE.Vector3(x,baseY-lift*.66,-z));}
  [points,mirror].forEach((pts,mi)=>{const curve=new THREE.CatmullRomCurve3(pts,false,'catmullrom',.12);const radius=.055+bi*.008;const geo=new THREE.TubeGeometry(curve,Math.min(500,slices.length*2),radius,6,false);const mat=new THREE.MeshPhysicalMaterial({color:palette[bi],emissive:palette[bi],emissiveIntensity:.36,transparent:true,opacity:mi?.28:.64,roughness:.22,metalness:.05,transmission:.42,thickness:.8,depthWrite:false});const mesh=new THREE.Mesh(geo,mat);mesh.userData={band};sculpture.add(mesh);});
 });
 // translucent skin joining overall energy / stereo width
 const positions=[];const indices=[];for(let i=0;i<slices.length;i++){const s=slices[i],x=(i/(slices.length-1)-.5)*trackLength,y=(Number(s.energy||0)-.32)*5.4,w=1.1+Number(s.stereo||0)*5.4;positions.push(x,y,w,x,-y*.72,-w);if(i<slices.length-1){const a=i*2,b=a+1,c=a+2,d=a+3;indices.push(a,c,b,b,c,d)}}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setIndex(indices);g.computeVertexNormals();const skin=new THREE.Mesh(g,new THREE.MeshPhysicalMaterial({color:0x8f91ff,transparent:true,opacity:.075,side:THREE.DoubleSide,roughness:.12,metalness:.1,transmission:.78,depthWrite:false}));skin.userData={band:'skin'};sculpture.add(skin);
 // playhead
 const pg=new THREE.PlaneGeometry(.08,19);const pm=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.22,side:THREE.DoubleSide});playhead=new THREE.Mesh(pg,pm);playhead.rotation.y=Math.PI/2;playhead.position.x=-trackLength/2;sculpture.add(playhead);
 buildPins();
}"""

    new_sculpture = """function buildSculpture(){clearGroup(sculpture);clearGroup(pinGroup);pinMeshes=[];const slices=visual.slices;if(!slices?.length)return;trackLength=Math.min(70,Math.max(38,visual.duration/3.7));
 const bandKeys=['sub','bass','low_mid','mid','high','air'];const norms=Object.fromEntries(bandKeys.map(k=>[k,bandNorm(k)]));
 const smooth=(arr,r=2)=>arr.map((_,i)=>{let sum=0,w=0;for(let j=Math.max(0,i-r);j<=Math.min(arr.length-1,i+r);j++){const ww=r+1-Math.abs(i-j);sum+=arr[j]*ww;w+=ww}return w?sum/w:arr[i]});
 const energy=smooth(slices.map(s=>Math.max(0,Math.min(1,Number(s.energy||0)))),2);const stereo=smooth(slices.map(s=>Math.max(0,Math.min(1,Number(s.stereo||0)))),2);const transient=smooth(slices.map(s=>Math.max(0,Math.min(1,Number(s.transient||0)))),1);
 bandKeys.forEach((band,bi)=>{const points=[];const mirror=[];const bandVals=smooth(norms[band],2);for(let i=0;i<slices.length;i++){const x=(i/(slices.length-1)-.5)*trackLength;const e=bandVals[i];const overall=energy[i];const width=stereo[i];const hit=transient[i];const baseY=(bi-2.5)*1.62;const bandWeight=bi<2?1.16:(bi>3?.92:1);const lift=(e*2.75+overall*.92)*bandWeight;const depthBase=(bi-2.5)*1.33;const depthSpread=(.45+width*2.9)*(bi<3?-1:1);const micro=hit*.22*(bi%2?-1:1);const z=depthBase+depthSpread+micro;points.push(new THREE.Vector3(x,baseY+lift,z));mirror.push(new THREE.Vector3(x,baseY-lift*.56,-z));}
  [points,mirror].forEach((pts,mi)=>{const curve=new THREE.CatmullRomCurve3(pts,false,'catmullrom',.18);const avgTransient=transient.reduce((a,b)=>a+b,0)/Math.max(1,transient.length);const radius=(.047+bi*.0065)*(1+avgTransient*.48);const geo=new THREE.TubeGeometry(curve,Math.min(560,slices.length*2),radius,7,false);const mat=new THREE.MeshPhysicalMaterial({color:palette[bi],emissive:palette[bi],emissiveIntensity:.28+avgTransient*.46,transparent:true,opacity:mi?.20:.68,roughness:.26,metalness:.04,transmission:.34,thickness:.7,depthWrite:false});const mesh=new THREE.Mesh(geo,mat);mesh.userData={band};sculpture.add(mesh);});
 });
 // overall body: height follows smoothed energy, depth follows stereo width
 const positions=[];const indices=[];for(let i=0;i<slices.length;i++){const x=(i/(slices.length-1)-.5)*trackLength;const y=(energy[i]-.28)*5.15;const w=.8+stereo[i]*4.9+energy[i]*.55;positions.push(x,y,w,x,-y*.66,-w);if(i<slices.length-1){const a=i*2,b=a+1,c=a+2,d=a+3;indices.push(a,c,b,b,c,d)}}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setIndex(indices);g.computeVertexNormals();const skin=new THREE.Mesh(g,new THREE.MeshPhysicalMaterial({color:0x8f91ff,transparent:true,opacity:.065,side:THREE.DoubleSide,roughness:.16,metalness:.06,transmission:.82,depthWrite:false}));skin.userData={band:'skin'};sculpture.add(skin);
 // transient accents: sparse vertical sparks at only the strongest attacks
 const hitThreshold=[...transient].sort((a,b)=>a-b)[Math.max(0,Math.floor((transient.length-1)*.92))]||1;for(let i=1;i<slices.length-1;i++){if(transient[i]<Math.max(.58,hitThreshold)||transient[i]<transient[i-1]||transient[i]<transient[i+1])continue;const x=(i/(slices.length-1)-.5)*trackLength;const h=.5+transient[i]*1.55;const geo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x,-.2,0),new THREE.Vector3(x,h,0)]);const line=new THREE.Line(geo,new THREE.LineBasicMaterial({color:0xffd58a,transparent:true,opacity:.14+transient[i]*.32}));line.userData={band:'transient_accent'};sculpture.add(line)}
 const pg=new THREE.PlaneGeometry(.08,19);const pm=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.22,side:THREE.DoubleSide});playhead=new THREE.Mesh(pg,pm);playhead.rotation.y=Math.PI/2;playhead.position.x=-trackLength/2;sculpture.add(playhead);buildPins();
}"""

    if old_norm in html:
        html = html.replace(old_norm, new_norm, 1)
    else:
        print("[visual-ui] patch target missing: band normalization")

    if old_sculpture in html:
        html = html.replace(old_sculpture, new_sculpture, 1)
    else:
        print("[visual-ui] patch target missing: sculpture")

    # Let the existing mode switch dim transient accents correctly.
    old_mode_piece = """if(mode==='transient')op=b==='skin'?.04:.31;m.material.opacity=op;"""
    new_mode_piece = """if(mode==='transient')op=b==='skin'?.04:(b==='transient_accent'?.78:.22);if(b==='transient_accent'&&mode!=='transient')op=.16;m.material.opacity=op;"""
    if old_mode_piece in html:
        html = html.replace(old_mode_piece, new_mode_piece, 1)

    # Ask SoundLens should reason from the same v3 musical-event model the user sees.
    old_ai_context = """const compact={basic:report.basic,loudness:report.loudness,frequency:report.frequency,rhythm:report.rhythm,sections:report.sections,artist_comparison:report.artist_comparison,visual_map:{duration:visual.duration,pins:visual.pins,legend:visual.legend}};"""
    new_ai_context = """const compact={basic:report.basic,loudness:report.loudness,frequency:report.frequency,rhythm:report.rhythm,scores:report.scores,stem_balance:report.stem_balance,top_problems:report.top_problems,next_steps:report.next_steps,sections:(Array.isArray(visual.sections)&&visual.sections.length?visual.sections:report.sections),artist_comparison:report.artist_comparison,visual_map:{version:visual.version,duration:visual.duration,pins:visual.pins,legend:visual.legend,sections:visual.sections,event_count:visual.pin_count,ai_timeline_summary:visual.ai_timeline_summary}};"""
    if old_ai_context in html:
        html = html.replace(old_ai_context, new_ai_context, 1)
    else:
        print("[visual-ui] patch target missing: Ask SoundLens context")

    return html


production.hardened._remove_route("/", "GET")


@app.get("/", response_class=HTMLResponse)
def visual_index():
    html = _INDEX_PATH.read_text(encoding="utf-8")
    html = production._patch_3d_frontend(html)
    html = _patch_visual_hierarchy(html)
    return HTMLResponse(html)

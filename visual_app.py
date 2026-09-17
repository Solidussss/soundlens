from __future__ import annotations

from pathlib import Path

from fastapi.responses import HTMLResponse

import production_app as production

app = production.app
_INDEX_PATH = Path(__file__).with_name("index.html")


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

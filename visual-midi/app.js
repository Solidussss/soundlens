const $=s=>document.querySelector(s);
const $$=s=>[...document.querySelectorAll(s)];
const canvas=$('#roll'),ctx=canvas.getContext('2d');
const state={shape:'heart',notes:[],playing:false,seed:1};

const KEYS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const SCALE_STEPS={major:[0,2,4,5,7,9,11],minor:[0,2,3,5,7,8,10],dorian:[0,2,3,5,7,9,10],phrygian:[0,1,3,5,7,8,10]};
KEYS.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;if(k==='F#')o.selected=true;$('#key').appendChild(o)});

function seeded(){state.seed=(state.seed*1664525+1013904223)>>>0;return state.seed/4294967296}
function scaleNotes(rootName,mode,min=48,max=84){
  const root=KEYS.indexOf(rootName),steps=SCALE_STEPS[mode];
  return Array.from({length:max-min+1},(_,i)=>i+min).filter(n=>steps.includes((n-root+120)%12));
}
function nearest(list,target){return list.reduce((a,b)=>Math.abs(b-target)<Math.abs(a-target)?b:a,list[0])}
function heart(x){const a=Math.PI*(2*x-1);return {top:0.67+0.18*Math.cos(a)-0.06*Math.cos(2*a),bottom:0.31-0.20*Math.sqrt(Math.max(0,1-Math.pow(2*x-1,2)))+0.08*Math.cos(a)}}
function star(x){const teeth=5;const wave=Math.abs((((x*teeth)%1)*2)-1);const mid=.5;const spread=.12+.24*(1-wave);return {top:mid+spread,bottom:mid-spread}}
function butterfly(x){const wing=Math.sin(Math.PI*x);const ripple=.06*Math.sin(x*Math.PI*6);const spread=.10+.28*wing;const shift=.05*Math.sin(x*Math.PI*2);return {top:.5+shift+spread+ripple,bottom:.5+shift-spread-ripple}}
function wave(x){const c=.5+.23*Math.sin(x*Math.PI*3.2);return {top:c+.11,bottom:c-.11}}
function shapeAt(x){return ({heart,star,butterfly,wave}[state.shape]||heart)(x)}

function params(){return {key:$('#key').value,scale:$('#scale').value,bpm:+$('#bpm').value,bars:+$('#bars').value,density:$('#density').value,balance:+$('#balance').value}}
function progression(root,mode,bars){
  const scale=SCALE_STEPS[mode],rootPc=KEYS.indexOf(root);
  const degrees=[0,5,3,6];
  return Array.from({length:bars},(_,i)=>new Set([0,2,4].map(off=>(rootPc+scale[(degrees[i%4]+off)%7])%12)));
}
function generate(){
  const p=params(),allowed=scaleNotes(p.key,p.scale),chords=progression(p.key,p.scale,p.bars);
  const perBar=p.density==='sparse'?4:p.density==='dense'?12:8;
  const count=p.bars*perBar,step=4/perBar;
  const freedom=p.balance/100;
  const notes=[];
  for(let i=0;i<count;i++){
    const x=i/(count-1||1),s=shapeAt(x),bar=Math.min(p.bars-1,Math.floor((i*step)/4)),chord=chords[bar];
    const targets=[s.bottom,s.top];
    targets.forEach((v,voice)=>{
      let target=50+v*30;
      const chordCandidates=allowed.filter(n=>chord.has(n%12));
      const musicalTarget=chordCandidates.length?nearest(chordCandidates,target):nearest(allowed,target);
      let pitch=nearest(allowed,target*(1-freedom*.45)+musicalTarget*(freedom*.45));
      if(freedom>.45&&seeded()<freedom*.18){
        const options=allowed.filter(n=>Math.abs(n-pitch)<=5);
        if(options.length)pitch=options[Math.floor(seeded()*options.length)];
      }
      const velocity=Math.round((voice?72:82)+(seeded()-.5)*10);
      const dur=step*(.72+(seeded()*.16));
      notes.push({beat:i*step,pitch,dur,velocity,chord:chord.has(pitch%12)});
    });
  }
  state.notes=notes;
  updateMeta();draw();
}
function updateMeta(){
  const p=params();
  $('#balanceReadout').textContent=`${100-p.balance} / ${p.balance}`;
  $('#metaText').textContent=`${p.key} ${p.scale} · ${p.bpm} BPM · ${p.bars} bars`;
  $('#noteCount').textContent=`${state.notes.length} notes`;
  $('#statusText').textContent=state.notes.length?'Generated':'Ready';
}
function resize(){
  const wrap=canvas.parentElement,dpr=Math.min(devicePixelRatio||1,2);
  const cssW=Math.max(wrap.clientWidth,900),cssH=wrap.clientHeight;
  canvas.style.width=cssW+'px';canvas.style.height=cssH+'px';
  canvas.width=cssW*dpr;canvas.height=cssH*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw();
}
function draw(){
  const w=parseFloat(canvas.style.width)||canvas.parentElement.clientWidth,h=parseFloat(canvas.style.height)||canvas.parentElement.clientHeight;
  const p=params(),keysW=62,top=28,totalBeats=p.bars*4,rowH=(h-top)/37,minPitch=48,maxPitch=84;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#090b0e';ctx.fillRect(0,0,w,h);
  for(let n=minPitch;n<=maxPitch;n++){
    const y=top+(maxPitch-n)*rowH,isBlack=[1,3,6,8,10].includes(n%12);
    ctx.fillStyle=isBlack?'#0b0d10':'#0f1216';ctx.fillRect(keysW,y,w-keysW,rowH);
    ctx.strokeStyle='#171b21';ctx.beginPath();ctx.moveTo(keysW,y);ctx.lineTo(w,y);ctx.stroke();
    ctx.fillStyle=isBlack?'#171a1f':'#e8eaee';ctx.fillRect(0,y,keysW-2,rowH-1);
    ctx.fillStyle=isBlack?'#aeb3bb':'#121418';ctx.font='10px system-ui';ctx.fillText(KEYS[n%12]+(Math.floor(n/12)-1),6,y+rowH*.68);
  }
  const beatW=(w-keysW)/totalBeats;
  for(let b=0;b<=totalBeats;b++){
    const x=keysW+b*beatW;ctx.strokeStyle=b%4===0?'#3a4050':'#20242b';ctx.lineWidth=b%4===0?1.2:1;
    ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,h);ctx.stroke();
    if(b%4===0&&b<totalBeats){ctx.fillStyle='#777e89';ctx.font='10px system-ui';ctx.fillText(String(b/4+1),x+5,18)}
  }
  state.notes.forEach(n=>{
    const x=keysW+n.beat*beatW,y=top+(maxPitch-n.pitch)*rowH+.8,nw=Math.max(3,n.dur*beatW-1),nh=Math.max(4,rowH-1.6);
    ctx.fillStyle=n.chord?'#a8b0ff':'#707cff';ctx.fillRect(x,y,nw,nh);
    ctx.fillStyle='rgba(255,255,255,.18)';ctx.fillRect(x,y,nw,1);
  });
}
function writeVarLen(value){let buffer=value&0x7f;const out=[];while(value>>=7){buffer<<=8;buffer|=((value&0x7f)|0x80)}while(true){out.push(buffer&0xff);if(buffer&0x80)buffer>>=8;else break}return out}
function u32(n){return[(n>>>24)&255,(n>>>16)&255,(n>>>8)&255,n&255]}
function u16(n){return[(n>>>8)&255,n&255]}
function exportMidi(){
  if(!state.notes.length)generate();
  const p=params(),ppq=480,events=[],tempo=Math.round(60000000/p.bpm);
  events.push({tick:0,data:[0xff,0x51,0x03,(tempo>>16)&255,(tempo>>8)&255,tempo&255]});
  state.notes.forEach(n=>{const st=Math.round(n.beat*ppq),en=Math.round((n.beat+n.dur)*ppq);events.push({tick:st,data:[0x90,n.pitch,n.velocity]});events.push({tick:en,data:[0x80,n.pitch,0]})});
  events.sort((a,b)=>a.tick-b.tick||((a.data[0]&0xf0)===0x80?-1:1));
  let track=[],last=0;events.forEach(e=>{track.push(...writeVarLen(e.tick-last),...e.data);last=e.tick});track.push(0,0xff,0x2f,0);
  const bytes=[...Array.from(new TextEncoder().encode('MThd')),...u32(6),...u16(0),...u16(1),...u16(ppq),...Array.from(new TextEncoder().encode('MTrk')),...u32(track.length),...track];
  const blob=new Blob([new Uint8Array(bytes)],{type:'audio/midi'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`soundlens-${state.shape}-${p.key.replace('#','sharp')}-${p.scale}.mid`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
function preview(){
  if(!state.notes.length)generate();
  const AC=window.AudioContext||window.webkitAudioContext;if(!AC)return alert('Audio preview is not supported in this browser.');
  const ac=new AC(),p=params(),secPerBeat=60/p.bpm,start=ac.currentTime+.05;
  state.notes.slice(0,180).forEach(n=>{const o=ac.createOscillator(),g=ac.createGain();o.type='sine';o.frequency.value=440*Math.pow(2,(n.pitch-69)/12);g.gain.setValueAtTime(0,start+n.beat*secPerBeat);g.gain.linearRampToValueAtTime(.025,start+n.beat*secPerBeat+.01);g.gain.exponentialRampToValueAtTime(.001,start+(n.beat+n.dur)*secPerBeat);o.connect(g).connect(ac.destination);o.start(start+n.beat*secPerBeat);o.stop(start+(n.beat+n.dur)*secPerBeat+.03)});
}
$$('.shape').forEach(b=>b.addEventListener('click',()=>{$$('.shape').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.shape=b.dataset.shape;generate()}));
$('#generateBtn').onclick=()=>{state.seed=Date.now()>>>0;generate()};$('#randomizeBtn').onclick=()=>{state.seed=(Date.now()^0x9e3779b9)>>>0;generate()};$('#exportBtn').onclick=exportMidi;$('#playBtn').onclick=preview;$('#clearBtn').onclick=()=>{state.notes=[];updateMeta();draw()};
['balance','key','scale','bpm','bars','density'].forEach(id=>$('#'+id).addEventListener('input',()=>{updateMeta();if(id==='balance')draw()}));
window.addEventListener('resize',resize);resize();generate();
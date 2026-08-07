"""Browser dashboard. stdlib http.server, no framework, no dependencies.

The terminal view redraws by clearing the screen, which fights with scrollback
and wedges if the window is too small. A browser page has none of those
problems: it repaints in place, resizes itself, and can be put on a second
screen. The app pushes a snapshot dict here; the page polls it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Latest state, replaced wholesale by the app thread. Assignment is atomic under
# the GIL, so the server thread never needs a lock to read a consistent value.
SNAPSHOT: dict = {"status": "waiting"}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>F1 26 ERS</title><style>
:root{
  --bg:#0F1216;--card:#161A20;--line:#232930;--text:#DBDEE3;--mut:#8B9098;
  --amber:#DFA84C;--teal:#54C3B9;--rose:#E37E94;--red:#E4574F;--green:#5FBF7F;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--mono);
  font-size:14px;padding:14px;-webkit-font-smoothing:antialiased}
.grid{display:grid;gap:12px;max-width:1100px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px}
.row{display:flex;flex-wrap:wrap;gap:10px 26px;align-items:baseline}
h1{font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);
  margin:0 0 12px;font-weight:400}
.big{font-size:30px;font-variant-numeric:tabular-nums;line-height:1}
.lab{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);
  display:block;margin-bottom:5px}
.val{font-size:19px;font-variant-numeric:tabular-nums}
.bar{height:22px;background:#0C0F13;border:1px solid var(--line);border-radius:3px;
  overflow:hidden;position:relative;min-width:220px;flex:1}
.fill{height:100%;transition:width .12s linear}
.state{padding:5px 12px;border-radius:4px;font-size:12px;letter-spacing:.1em;
  text-transform:uppercase}
.live{background:rgba(95,191,127,.14);color:var(--green)}
.paused{background:rgba(223,168,76,.16);color:var(--amber)}
.lost{background:rgba(228,87,79,.16);color:var(--red)}
canvas{width:100%;height:150px;display:block}
.verdict{border-left:3px solid var(--red);padding-left:12px}
.vcost{font-size:26px;color:var(--red);font-variant-numeric:tabular-nums}
.advice{color:var(--teal);margin-top:8px}
.det{color:var(--mut);margin-top:5px;line-height:1.5}
.also{color:var(--mut);font-size:12px;margin-top:9px}
.cue{background:rgba(223,168,76,.16);color:var(--amber);padding:10px 13px;
  border-radius:4px;margin-top:11px;font-size:15px}
.key{display:flex;gap:16px;font-size:11px;color:var(--mut);margin-top:7px}
.key i{font-style:normal}
.dim{color:var(--mut)}
.nexttip{font-size:20px;line-height:1.35;padding:12px 14px;border-radius:5px;
  background:rgba(84,195,185,.10);border-left:3px solid var(--teal)}
.nexttip .nw{color:var(--mut);font-size:12px;letter-spacing:.1em;
  text-transform:uppercase;margin-bottom:4px}
.nexttip .na{color:var(--teal);font-size:26px;font-weight:600;letter-spacing:.02em}
.nexttip .ny{color:var(--text);font-size:14px;margin-top:6px;opacity:.85}
.nexttip .g{color:var(--mut);font-size:15px;font-weight:400}
.tip .ta{color:var(--text);font-weight:600;letter-spacing:.02em}
.race{padding:10px 0;border-bottom:1px solid var(--line)}
.race:last-child{border-bottom:0}
.race .ra{color:var(--amber);font-size:19px;font-weight:600}
.race .rw{color:var(--mut);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;margin-top:2px}
.race .ry{color:var(--text);font-size:13px;margin-top:5px;opacity:.85}
.tip{display:flex;gap:12px;align-items:baseline;padding:9px 0;
  border-bottom:1px solid var(--line)}
.tip:last-child{border-bottom:0}
.tip .w{color:var(--teal);min-width:9rem;flex:none}
.tip .g{color:var(--mut);font-variant-numeric:tabular-nums;flex:none}
.tip .d{color:var(--mut);font-size:12px}
.sc{background:rgba(223,168,76,.18);color:var(--amber);padding:10px 14px;
  border-radius:5px;margin-top:11px;font-size:15px;letter-spacing:.08em}
@media(max-width:640px){.big{font-size:24px}canvas{height:120px}}
</style></head><body>
<div class="grid">
  <div class="card">
    <div class="row" style="justify-content:space-between">
      <h1 style="margin:0">F1 26 ERS Optimiser</h1>
      <span id="state" class="state lost">no telemetry</span>
    </div>
    <div class="row" style="margin-top:12px">
      <div><span class="lab">Lap</span><span class="big" id="lap">–</span></div>
      <div><span class="lab">Sector</span><span class="big" id="sec">–</span></div>
      <div><span class="lab">Speed km/h</span><span class="big" id="spd">–</span></div>
      <div><span class="lab">MGU-K kW</span><span class="big" id="kw">–</span></div>
      <div><span class="lab">Lap time</span><span class="big" id="lt">–</span></div>
    </div>
  </div>

  <div class="card">
    <span class="lab">Battery</span>
    <div class="row">
      <div class="bar"><div class="fill" id="socbar" style="width:0"></div></div>
      <div class="val" id="soc">–</div>
    </div>
    <div class="row" style="margin-top:13px">
      <div><span class="lab">Deployed</span><span class="val" id="dep">–</span></div>
      <div><span class="lab">Harvested</span><span class="val" id="har">–</span></div>
      <div><span class="lab">Mode</span><span class="val" id="mode">–</span></div>
      <div><span class="lab">vs plan</span><span class="val" id="vsplan">–</span></div>
      <div><span class="lab">Energy price</span><span class="val" id="lam">–</span></div>
    </div>
    <div id="cuebox"></div>
  </div>

  <div class="card">
    <span class="lab">Last lap — deployment around the circuit</span>
    <canvas id="c" width="1040" height="150"></canvas>
    <div class="key"><i style="color:var(--mut)">speed</i>
      <i style="color:var(--rose)">you</i>
      <i style="color:var(--teal)">optimal</i>
      <i style="color:var(--amber)">charge</i></div>
  </div>

  <div class="card" id="rcard" style="display:none">
    <span class="lab">Right now</span>
    <div id="racelist"></div>
  </div>

  <div class="card" id="tcard" style="display:none">
    <span class="lab">Coming up</span>
    <div id="nexttip" class="nexttip"></div>
    <div class="lab" style="margin-top:14px">Fix these, in order</div>
    <div id="tiplist"></div>
  </div>

  <div class="card" id="pcard" style="display:none">
    <div class="row" style="justify-content:space-between">
      <span class="lab" style="margin:0">Car ahead</span>
      <span id="pverdict" class="state">–</span>
    </div>
    <div class="row" style="margin-top:11px">
      <div><span class="lab">Gap</span><span class="val" id="pgap">–</span></div>
      <div><span class="lab">Costing you</span><span class="val" id="pdef">–</span></div>
      <div><span class="lab">Laps left</span><span class="val" id="plaps">–</span></div>
      <div><span class="lab">Attack costs</span><span class="val" id="pe">–</span></div>
      <div><span class="lab">Break-even</span><span class="val" id="pbe">–</span></div>
    </div>
    <div class="det" id="pdet" style="margin-top:10px"></div>
    <div class="advice" id="padv"></div>
  </div>

  <div class="card" id="vcard" style="display:none">
    <span class="lab">Biggest loss last lap</span>
    <div class="verdict">
      <div class="row"><span class="vcost" id="vcost">–</span>
        <span id="vname" style="font-size:17px"></span>
        <span id="vpool" class="dim"></span></div>
      <div class="det" id="vdet"></div>
      <div class="det" id="vwhere"></div>
      <div class="advice" id="vadv"></div>
      <div class="also" id="valso"></div>
    </div>
  </div>

  <div class="card">
    <div class="row">
      <div><span class="lab">Best lap</span><span class="val" id="best">–</span></div>
      <div><span class="lab">ERS loss / lap</span><span class="val" id="loss">–</span></div>
      <div><span class="lab">Model error</span><span class="val" id="fid">–</span></div>
      <div><span class="lab">Laps</span><span class="val" id="laps">–</span></div>
    </div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
let lastTrace=-1;
function fmt(x,d){return x==null?'–':x.toFixed(d===undefined?2:d)}

function draw(t){
  const c=$('c'),x=c.getContext('2d'),W=c.width,H=c.height;
  x.clearRect(0,0,W,H);
  if(!t||!t.u_you||!t.u_you.length)return;
  const n=t.u_you.length,pad=6;
  const sx=i=>i*(W/(n-1));
  const line=(arr,col,lo,hi,fill)=>{
    const rng=(hi-lo)||1;
    x.beginPath();
    for(let i=0;i<n;i++){
      const y=H-pad-((arr[i]-lo)/rng)*(H-2*pad);
      i?x.lineTo(sx(i),y):x.moveTo(sx(i),y);
    }
    if(fill){x.lineTo(W,H);x.lineTo(0,H);x.closePath();x.fillStyle=col;x.fill();}
    else{x.strokeStyle=col;x.lineWidth=1.6;x.stroke();}
  };
  const pk=Math.max(...t.u_you,...t.u_opt,1);
  line(t.u_opt,'rgba(84,195,185,.30)',0,pk,true);
  line(t.u_you,'rgba(227,126,148,.34)',0,pk,true);
  line(t.u_opt,'#54C3B9',0,pk,false);
  line(t.u_you,'#E37E94',0,pk,false);
  line(t.v,'#6C737C',0,Math.max(...t.v)*1.05,false);
  line(t.soc,'#DFA84C',0,Math.max(...t.soc,1)*1.05,false);
}

async function tick(){
  let s;
  try{ s=await (await fetch('/state',{cache:'no-store'})).json(); }
  catch(e){ $('state').className='state lost'; $('state').textContent='app stopped'; return; }

  const st=$('state');
  st.className='state '+(s.status==='live'?'live':s.status==='paused'?'paused':'lost');
  st.textContent=s.status==='live'?'live':s.status==='paused'?'paused':'no telemetry';

  $('lap').textContent=s.lap??'–';
  $('sec').textContent=s.sector!=null?s.sector+1:'–';
  $('spd').textContent=fmt(s.speed,0);
  $('kw').textContent=fmt(s.mguk,0);
  $('lt').textContent=s.lap_time!=null?fmt(s.lap_time):'–';

  const f=s.soc_frac||0;
  $('socbar').style.width=(f*100).toFixed(1)+'%';
  $('socbar').style.background=f<.18?'#E4574F':f<.4?'#DFA84C':'#54C3B9';
  $('soc').textContent=s.soc!=null?fmt(s.soc)+' MJ':'–';
  $('dep').textContent=s.deployed!=null?fmt(s.deployed)+' MJ':'–';
  $('har').textContent=s.harvested!=null?fmt(s.harvested)+' MJ':'–';
  $('mode').textContent=s.mode||'–';
  $('lam').textContent=s.lam!=null?fmt(s.lam,3)+' s/MJ':'–';
  const vp=$('vsplan');
  if(s.vs_plan==null){vp.textContent='–';vp.style.color='';}
  else{vp.textContent=(s.vs_plan>=0?'+':'')+fmt(s.vs_plan)+' MJ';
       vp.style.color=Math.abs(s.vs_plan)<.3?'#5FBF7F':s.vs_plan>0?'#DFA84C':'#54C3B9';}

  $('cuebox').innerHTML=s.cue?'<div class="cue">▲ '+s.cue+'</div>':'';

  if(s.safety_car){
    $('cuebox').innerHTML+='<div class="sc">'+s.safety_car+
      ' — bank everything, the restart is where it pays</div>';
  }

  const race=s.race||[];
  if(race.length){
    $('rcard').style.display='';
    $('racelist').innerHTML = race.map(r=>
      '<div class="race"><div class="ra">'+r.action+'</div>'+
      '<div class="rw">'+r.where+'</div>'+
      '<div class="ry">'+r.why+'</div></div>').join('');
  } else { $('rcard').style.display='none'; }

  const tips=s.tips||[];
  const nt=s.next_tip;
  if(tips.length||nt){
    $('tcard').style.display='';
    $('nexttip').innerHTML = nt
      ? '<div class="nw">'+nt.where+'</div>'+
        '<div class="na">'+nt.action+'<span class="g"> +'+fmt(nt.gain)+'s</span></div>'+
        '<div class="ny">'+nt.why+'</div>'
      : '<span class="g">nothing flagged on this stretch</span>';
    $('tiplist').innerHTML = tips.map(t=>
      '<div class="tip"><span class="w">'+t.where+'</span>'+
      '<span class="g">+'+fmt(t.gain)+'s</span>'+
      '<span><div class="ta">'+t.action+'</div>'+
      '<div class="d">'+t.why+'</div></span></div>'
    ).join('');
  } else { $('tcard').style.display='none'; }

  const p=s.pass;
  if(p){
    $('pcard').style.display='';
    const pv=$('pverdict');
    pv.textContent=p.verdict;
    pv.className='state '+(p.verdict==='attack'?'live':p.verdict==='hold'?'lost':'paused');
    $('pgap').textContent=s.gap!=null?fmt(s.gap)+'s':'–';
    $('pdef').textContent=fmt(p.deficit)+'s/lap';
    $('plaps').textContent=s.laps_left??'–';
    $('pe').textContent=fmt(p.energy)+' MJ';
    $('pbe').textContent=p.breakeven!=null?(p.breakeven*100).toFixed(0)+'%':'>900%';
    $('pdet').textContent=p.detail;
    $('padv').textContent='→ '+p.advice;
  } else { $('pcard').style.display='none'; }

  if(s.trace&&s.trace_id!==lastTrace){lastTrace=s.trace_id;draw(s.trace);}

  if(s.verdict){
    $('vcard').style.display='';
    $('vcost').textContent=fmt(s.verdict.cost)+'s';
    $('vname').textContent=s.verdict.name;
    $('vpool').textContent=s.verdict.pool==='next'?'(costs next lap)':'';
    $('vdet').textContent=s.verdict.detail;
    $('vwhere').textContent=s.verdict.where||'';
    $('vadv').textContent='→ '+s.verdict.advice;
    $('valso').textContent=(s.also||[]).map(a=>'also '+fmt(a.cost)+'s  '+a.name).join('   ');
  }

  $('best').textContent=s.best!=null?fmt(s.best,3)+'s':'–';
  $('loss').textContent=s.loss!=null?fmt(s.loss)+'s':'–';
  $('fid').textContent=s.fidelity!=null?fmt(s.fidelity,2)+'%':'–';
  $('laps').textContent=s.laps??'–';
}
setInterval(tick,250); tick();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/state"):
            body = json.dumps(SNAPSHOT).encode()
            ctype = "application/json"
        elif self.path == "/":
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # a request log every 250 ms would bury the terminal output


def serve(port: int = 8765) -> str:
    """Starts the dashboard server on a daemon thread; returns its URL."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/"

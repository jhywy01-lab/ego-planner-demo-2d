const API = window.EGO_API || 'http://localhost:8000';
const canvas = document.querySelector('#mapCanvas');
const ctx = canvas.getContext('2d');
const ui = { mode:'start', environment:null, start:null, goal:null, paths:[], trajectory:null };
const $ = id => document.getElementById(id);
const message = text => { $('message').textContent = text; };
const fmt = p => p ? `(${p[0].toFixed(2)}, ${p[1].toFixed(2)})` : '未设置';

function worldFromCanvas(e) { const r=canvas.getBoundingClientRect(); return [((e.clientX-r.left)/r.width)*10, (1-(e.clientY-r.top)/r.height)*10]; }
function canvasFromWorld([x,y]) { return [x/10*canvas.width, canvas.height-y/10*canvas.height]; }
function drawLine(points, color, width=2, dash=[]) { if(!points?.length)return; ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();points.forEach((p,i)=>{const [x,y]=canvasFromWorld(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.restore(); }
function render() { const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#0b1220';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#25324a';ctx.lineWidth=1;for(let i=0;i<=10;i++){let x=i*w/10,y=i*h/10;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const occ=ui.environment?.occupied||[];ctx.fillStyle='#64748b';for(let y=0;y<occ.length;y++)for(let x=0;x<(occ[y]?.length||0);x++)if(occ[y][x])ctx.fillRect(x*w/100,h-(y+1)*h/100,w/100+1,h/100+1);
  ui.paths.forEach(p=>drawLine(p.waypoints,'#fb923c',2,[5,5]));
  if(ui.trajectory)drawLine(ui.trajectory.samples.map(s=>s.position),'#22d3ee',4);
  if(ui.start) marker(ui.start,'#34d399','S'); if(ui.goal) marker(ui.goal,'#f472b6','G');
}
function marker(point,color,label){const [x,y]=canvasFromWorld(point);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill();ctx.fillStyle='#06101b';ctx.font='bold 10px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(label,x,y);}
async function request(path, options={}) { const res=await fetch(API+path,{headers:{'Content-Type':'application/json'},...options});const data=await res.json();if(data.status==='error')throw Error(data.message);return data; }
async function loadEnvironment(){ui.environment=await request('/api/environment');$('occupiedValue').textContent=ui.environment.occupied_cells;render();}
async function setPoint(point,type){try{const data=await request(`/api/planning/set-${type}?x=${point[0]}&y=${point[1]}`,{method:'POST'});ui[type]=data[type];$(type+'Value').textContent=fmt(ui[type]);message(`${type==='start'?'起点':'目标'}已设置。`);render();}catch(e){message(e.message);}}
async function search(){try{const t=performance.now();const data=await request('/api/planning/search',{method:'POST'});ui.paths=data.paths; $('expandedValue').textContent=data.cells_expanded;$('searchTime').textContent=data.search_time_ms.toFixed(2)+' ms';$('totalTime').textContent=(performance.now()-t).toFixed(2)+' ms';message(`A* 找到 ${data.num_paths} 条候选路径。`);render();}catch(e){message(e.message);}}
async function initialize(){try{const data=await request('/api/planning/initialize-trajectory?path_index=0',{method:'POST'});ui.trajectory=data.trajectory;$('totalTime').textContent=data.trajectory.initialization_time_ms.toFixed(2)+' ms';message('已生成分段五次多项式初始轨迹。');render();}catch(e){message(e.message);}}
let costChart,dynamicsChart;
function updateCharts(reports=[]){const labels=reports.map((r,i)=>r.iteration??i);const costs=['cost_total','cost_smooth','cost_obstacle','cost_dynamics'].map((key,i)=>({label:key,data:reports.map(r=>r[key]),borderColor:['#22d3ee','#34d399','#fb923c','#f472b6'][i],tension:.25}));if(costChart){costChart.data.labels=labels;costChart.data.datasets=costs;costChart.update();}if(dynamicsChart){dynamicsChart.data.labels=labels;dynamicsChart.data.datasets=[{label:'最大速度',data:reports.map(r=>r.max_speed),borderColor:'#22d3ee'},{label:'最大加速度',data:reports.map(r=>r.max_acceleration),borderColor:'#fb923c'}];dynamicsChart.update();}}
async function optimize(){try{const t=performance.now();const data=await request('/api/planning/optimize',{method:'POST'});ui.trajectory=data.trajectory;$('optimizationTime').textContent=data.optimization_time_ms.toFixed(2)+' ms';message(`优化完成，共 ${data.num_iterations} 次迭代。`);const history=await request('/api/planning/optimization-history');updateCharts(history.reports);render();}catch(e){message(e.message);}}
async function runAll(){await search();if(ui.paths.length){await initialize();await optimize();}}
function reset(){request('/api/planning/reset',{method:'POST'}).then(()=>{ui.start=ui.goal=null;ui.paths=[];ui.trajectory=null;$('startValue').textContent=$('goalValue').textContent='未设置';['expandedValue','searchTime','optimizationTime','totalTime'].forEach(k=>$(k).textContent='—');message('规划状态已重置。');render();});}
canvas.addEventListener('click',e=>{const p=worldFromCanvas(e);if(ui.mode==='start')setPoint(p,'start');else if(ui.mode==='goal')setPoint(p,'goal');});
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));b.classList.add('active');ui.mode=b.dataset.mode;});$('searchBtn').onclick=search;$('initBtn').onclick=initialize;$('optimizeBtn').onclick=optimize;$('runAllBtn').onclick=runAll;$('resetBtn').onclick=reset;$('safetyInput').oninput=e=>$('safetyOut').textContent=e.target.value+' m';
function initCharts(){const common={responsive:true,animation:false,plugins:{legend:{labels:{color:'#cbd5e1'}}},scales:{x:{ticks:{color:'#94a3b8'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}}};costChart=new Chart($('costChart'),{type:'line',data:{labels:[],datasets:[]},options:common});dynamicsChart=new Chart($('dynamicsChart'),{type:'line',data:{labels:[],datasets:[]},options:common});}
function connect(){const dot=document.querySelector('.connection');const ws=new WebSocket(API.replace('http','ws')+'/ws');ws.onopen=()=>{dot.classList.add('connected');$('statusText').textContent='已连接';};ws.onclose=()=>{dot.classList.remove('connected');$('statusText').textContent='未连接';setTimeout(connect,3000);};}
initCharts();loadEnvironment().catch(e=>message('后端未启动：'+e.message));connect();

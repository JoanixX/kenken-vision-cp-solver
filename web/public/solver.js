// Browser-native CSP solver: table propagation + MRV search, independent of Python.
const aliases = {'×':'*',x:'*',X:'*','·':'*','÷':'/',':':'/','−':'-','–':'-','':'='};
export function validate(raw) {
  if (!raw || !Number.isInteger(raw.n) || raw.n < 1 || raw.n > 9 || !Array.isArray(raw.cages)) throw Error('n debe ser un entero entre 1 y 9 y cages una lista.');
  const n = raw.n, seen = new Set();
  const cages = raw.cages.map((c,k) => {
    if (!c || !Array.isArray(c.cells) || !c.cells.length || !Number.isSafeInteger(c.target)) throw Error(`Jaula ${k}: indica cells y un objetivo entero.`);
    const op = aliases[c.op] ?? c.op ?? '=';
    if (!['+','-','*','/','='].includes(op)) throw Error(`Jaula ${k}: operación desconocida.`);
    if ((op === '=' && c.cells.length !== 1) || (op !== '=' && c.cells.length === 1) || (['-','/'].includes(op) && c.cells.length !== 2)) throw Error(`Jaula ${k}: cantidad de celdas incompatible con ${op}.`);
    const cells = c.cells.map(cell => {
      if (!Array.isArray(cell) || cell.length!==2 || cell.some(v=>!Number.isInteger(v)||v<0||v>=n)) throw Error(`Jaula ${k}: celda fuera del tablero.`);
      const id=cell[0]*n+cell[1]; if(seen.has(id)) throw Error(`La celda ${cell} aparece más de una vez.`); seen.add(id); return [...cell];
    });
    const reached=new Set([cells[0].join(',')]); let changed=true;
    while(changed){changed=false;for(const [r,c] of cells) if(!reached.has(`${r},${c}`)&&[[r-1,c],[r+1,c],[r,c-1],[r,c+1]].some(x=>reached.has(x.join(',')))) {reached.add(`${r},${c}`);changed=true;}}
    if(reached.size!==cells.length) throw Error(`Jaula ${k}: las celdas deben estar conectadas.`);
    return {cells,target:c.target,op};
  });
  if(seen.size!==n*n) throw Error(`Faltan ${n*n-seen.size} celdas por asignar a una jaula.`);
  return {n,cages};
}
export function satisfies(c,v){switch(c.op){case '=':return v[0]===c.target;case '+':return v.reduce((a,b)=>a+b,0)===c.target;case '*':return v.reduce((a,b)=>a*b,1)===c.target;case '-':return Math.abs(v[0]-v[1])===c.target;case '/':return v[0]===c.target*v[1]||v[1]===c.target*v[0];default:return false;}}
export function checkSolution(raw,grid){try{const {n,cages}=validate(raw);if(!Array.isArray(grid)||grid.length!==n||grid.some(row=>!Array.isArray(row)||row.length!==n||row.some(v=>!Number.isInteger(v)||v<1||v>n)))return false;for(let i=0;i<n;i++)if(new Set(grid[i]).size!==n||new Set(grid.map(r=>r[i])).size!==n)return false;return cages.every(c=>satisfies(c,c.cells.map(([r,c])=>grid[r][c])));}catch{return false;}}
export function solve(raw,{timeLimit=10000}={}){
  const start=performance.now(),deadline=start+timeLimit,{n,cages}=validate(raw),full=(1<<n)-1;
  let branches=0,conflicts=0,ticks=0;
  function guard(){if((++ticks&255)===0&&performance.now()>deadline)throw Error('TIMEOUT');}
  const count=mask=>{let k=0;for(;mask;mask&=mask-1)k++;return k;};
  const bits=mask=>{const out=[];for(let v=1;v<=n;v++)if(mask&(1<<(v-1)))out.push(v);return out;};
  function tuples(c){const result=[],v=[];function enumerate(k,total){guard();if(k===c.cells.length){if(satisfies(c,v))result.push([...v]);return;}for(let x=1;x<=n;x++){if(c.op==='+'&&(total+x+c.cells.length-k-1>c.target||total+x+n*(c.cells.length-k-1)<c.target))continue;if(c.op==='*'&&(total*x>c.target||c.target%(total*x)!==0))continue;if(v.some((y,j)=>y===x&&(c.cells[j][0]===c.cells[k][0]||c.cells[j][1]===c.cells[k][1])))continue;v.push(x);enumerate(k+1,c.op==='*'?total*x:total+x);v.pop();}}enumerate(0,c.op==='*'?1:0);return result;}
  const groups=[];for(let i=0;i<n;i++){groups.push(Array.from({length:n},(_,j)=>i*n+j));groups.push(Array.from({length:n},(_,j)=>j*n+i));}
  try{
    const tables=cages.map(c=>({ids:c.cells.map(([r,c])=>r*n+c),tuples:tuples(c)}));
    function propagate(d){let changed=true;while(changed){guard();changed=false;for(const group of groups){let singles=0;for(const id of group)if(count(d[id])===1){if(singles&d[id])return false;singles|=d[id];}for(const id of group)if(count(d[id])>1){const next=d[id]&~singles;if(!next)return false;if(next!==d[id]){d[id]=next;changed=true;}}for(let value=1;value<=n;value++){const bit=1<<(value-1),places=group.filter(id=>d[id]&bit);if(!places.length)return false;if(places.length===1&&d[places[0]]!==bit){d[places[0]]=bit;changed=true;}}}
      for(const table of tables){const support=table.ids.map(()=>0);let valid=false;for(const t of table.tuples){guard();if(t.every((v,k)=>d[table.ids[k]]&(1<<(v-1)))){valid=true;t.forEach((v,k)=>support[k]|=1<<(v-1));}}if(!valid)return false;for(let k=0;k<table.ids.length;k++){const id=table.ids[k],next=d[id]&support[k];if(!next)return false;if(next!==d[id]){d[id]=next;changed=true;}}}}return true;}
    function search(d){guard();if(!propagate(d)){conflicts++;return null;}let id=-1,min=10;d.forEach((mask,k)=>{const size=count(mask);if(size>1&&size<min){id=k;min=size;}});if(id<0)return d;for(const v of bits(d[id])){branches++;const child=[...d];child[id]=1<<(v-1);const result=search(child);if(result)return result;}return null;}
    const result=search(Array(n*n).fill(full));const grid=result?Array.from({length:n},(_,r)=>result.slice(r*n,(r+1)*n).map(mask=>bits(mask)[0])):null;
    if(grid&&!checkSolution(raw,grid))throw Error('La verificación independiente de la solución falló.');
    return {status:grid?'FEASIBLE':'INFEASIBLE',grid,instance:{n,cages},branches,conflicts,solver_ms:performance.now()-start,engine:'Navegador · CSP de tablas',fidelity:100,num_changed:0,cached:false};
  }catch(error){if(error.message==='TIMEOUT')return {status:'UNKNOWN',grid:null,instance:{n,cages},branches,conflicts,solver_ms:performance.now()-start,engine:'Navegador · CSP de tablas',cached:false};throw error;}
}
export function generate(n=4){const shuffled=a=>a.map(x=>({x,k:Math.random()})).sort((a,b)=>a.k-b.k).map(o=>o.x);const rows=shuffled([...Array(n).keys()]),cols=shuffled([...Array(n).keys()]),symbols=shuffled(Array.from({length:n},(_,i)=>i+1));const grid=rows.map(r=>cols.map(c=>symbols[(r+c)%n]));const cages=[];for(let r=0;r<n;r++)for(let c=0;c<n;){if(c+1<n&&Math.random()>.2){const a=grid[r][c],b=grid[r][c+1];const op=Math.random()>.5?'+':'*';cages.push({cells:[[r,c],[r,c+1]],target:op==='+'?a+b:a*b,op});c+=2;}else{cages.push({cells:[[r,c]],target:grid[r][c],op:'='});c++;}}return {n,cages};}

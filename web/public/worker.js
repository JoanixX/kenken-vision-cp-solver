import {solve} from './solver.js';
self.onmessage=({data})=>{try{self.postMessage({result:solve(data.instance,data.options)});}catch(error){self.postMessage({error:error.message});}};

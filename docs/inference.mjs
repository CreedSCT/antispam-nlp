export function vectorize(text,model){
 const normalized=text.toLowerCase().normalize('NFKD').replace(/\p{M}/gu,'');
 const tokens=normalized.match(/[\p{L}\p{N}_]{2,}/gu)||[];
 const counts=new Map();
 function add(s){const i=Object.hasOwn(model.vocabulary,s)?model.vocabulary[s]:undefined;if(i!==undefined)counts.set(i,(counts.get(i)||0)+1)}
 for(const token of tokens)add(token);
 for(let i=0;i+1<tokens.length;i++)add(tokens[i]+' '+tokens[i+1]);
 let norm=0;const values=[];
 for(const [i,n] of counts){const value=(1+Math.log(n))*model.idf[i];values.push([i,value]);norm+=value*value}
 norm=Math.sqrt(norm)||1;return values.map(([i,v])=>[i,v/norm]);
}
export function predict(text,model){
 let logit=0;for(const [i,v] of vectorize(text,model))logit+=v*model.weights[i];
 const score=1/(1+Math.exp(-logit));return {label:score>=model.threshold?'spam':'ham',spam_score:score,threshold:model.threshold};
}
export function category(text,model){
 const logits=model.intercepts.slice();for(const [i,v] of vectorize(text,model))for(let k=0;k<logits.length;k++)logits[k]+=v*model.weights[k][i];
 const max=Math.max(...logits);const probs=logits.map(x=>Math.exp(x-max));const sum=probs.reduce((a,b)=>a+b,0);const k=logits.indexOf(max);
 return {label:model.classes[k],score:probs[k]/sum};
}

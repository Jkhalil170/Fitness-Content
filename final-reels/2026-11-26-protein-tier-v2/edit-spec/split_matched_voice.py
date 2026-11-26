from pathlib import Path
import numpy as np,json,soundfile as sf
r=Path(__file__).parent;rows=json.loads((r/'narration.json').read_text());words=[w for s in json.loads((r/'narration/context-alignment.json').read_text())for w in s['words']];a,sr=sf.read(r/'narration/liam-context-full.wav',dtype='float32');starts=[0,2,4,6,8,9,11,15,16,23,25,29,31];expected=['peanut','chicken','whole','protein','shrimp','regular','non','almonds','lean','egg','low','turkey','white'];bounds=[0]
for i in range(1,len(starts)):
 prev=words[starts[i]-1]['end'];nxt=words[starts[i]]['start'];mid=(prev+nxt)/2
 # Select a low-energy boundary within the inter-word gap; never seek inside the preceding/following vowel.
 lo=max(prev-.035,mid-.10);hi=min(nxt+.035,mid+.10)
 if hi<=lo:lo=mid-.025;hi=mid+.025
 times=np.arange(lo,hi,.002);energies=[float(np.mean(a[max(0,int((t-.003)*sr)):min(len(a),int((t+.003)*sr))]**2))for t in times];best=min(range(len(times)),key=lambda k:energies[k]+1e-6*abs(times[k]-mid));bounds.append(float(times[best]))
bounds.append(len(a)/sr)
for i,row in enumerate(rows):
 assert words[starts[i]]['word'].strip().lower().startswith(expected[i]);clip=a[round(bounds[i]*sr):round(bounds[i+1]*sr)].copy();active=np.flatnonzero(abs(clip)>.002)
 if len(active):clip=clip[max(0,active[0]-round(.025*sr)):min(len(clip),active[-1]+round(.06*sr))]
 # Same perceived loudness as the approved sample. Preserve pitch and playback rate.
 fade=min(round(.003*sr),len(clip)//4);clip[:fade]*=np.linspace(0,1,fade);clip[-fade:]*=np.linspace(1,0,fade);clip=np.pad(clip,(round(.025*sr),round(.06*sr)));out=r/f'narration/item-{i:02}.wav';sf.write(out,clip,sr,subtype='PCM_16');row.update(file=str(out),duration=len(clip)/sr,voice='am_liam',generation='approved Liam conditioning, full-context synthesis, no pitch or speed processing',context_range=[bounds[i],bounds[i+1]]);print(i,row['item'],round(row['duration'],3),flush=True)
(r/'narration.json').write_text(json.dumps(rows,indent=2))

from pathlib import Path
from faster_whisper import WhisperModel
import json,numpy as np,soundfile as sf,subprocess,re
r=Path(__file__).parent;m=WhisperModel('small.en',device='cpu',compute_type='int8',download_root='/workspace/work/fitness-ranking/models',cpu_threads=2);out=[]
for voice in ['am_michael','am_liam','am_eric']:
 wav=r/f'{voice}.wav';a,s=sf.read(wav);segs,_=m.transcribe(str(wav),language='en',word_timestamps=True,beam_size=5,vad_filter=False);segs=list(segs);words=[w for x in segs for w in x.words];text=' '.join(x.text.strip()for x in segs);norm=re.sub('[^a-z ]','',text.lower());assert norm.split()==['chicken','breast','greek','yogurt','peanut','butter'],text
 clips=[];d=r/'split-clips'/voice;d.mkdir(parents=True,exist_ok=True)
 for i,name in enumerate(['chicken-breast','greek-yogurt','peanut-butter']):
  left=0 if i==0 else (words[i*2-1].end+words[i*2].start)/2;right=len(a)/s if i==2 else (words[i*2+1].end+words[i*2+2].start)/2
  # Cut only at quiet samples close to the aligned inter-phrase boundary.
  def boundary(t):
   center=round(t*s);lo=max(0,center-round(.055*s));hi=min(len(a),center+round(.055*s));return lo+int(np.argmin(np.convolve(a[lo:hi]**2,np.ones(120)/120,mode='same')))
  start=0 if i==0 else boundary(left);end=len(a) if i==2 else boundary(right);clip=a[start:end].copy();n=min(round(.004*s),len(clip)//2);clip[:n]*=np.linspace(0,1,n);clip[-n:]*=np.linspace(1,0,n);sf.write(d/f'{name}.wav',clip,s,subtype='PCM_24');clips.append({'name':name,'start':start/s,'end':end/s,'file':f'split-clips/{voice}/{name}.wav'})
 out.append({'voice':voice,'transcript':text,'words':[{'word':w.word,'start':w.start,'end':w.end}for w in words],'duration':len(a)/s,'peak':float(max(abs(a))),'clips':clips});print(voice,text,flush=True)
(r/'verification.json').write_text(json.dumps(out,indent=2))

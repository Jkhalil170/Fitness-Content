from pathlib import Path
from faster_whisper import WhisperModel
import json,numpy as np,soundfile as sf,re
r=Path(__file__).parent;m=WhisperModel('small.en',device='cpu',compute_type='int8',download_root='/workspace/work/fitness-ranking/models',cpu_threads=2);out=[]
for voice in ['af_bella','af_nicole','af_nova']:
 p=r/f'{voice}.wav';a,s=sf.read(p);segs,_=m.transcribe(str(p),language='en',beam_size=5,word_timestamps=True,vad_filter=False);segs=list(segs);text=' '.join(x.text.strip()for x in segs);expected='Joseph which are the hardest muscles for men to build Abs Chest Quads Triceps Traps Calves'.lower().split();norm=re.sub('[^a-z ]','',text.lower()).split();assert norm==expected,(voice,text);assert np.isfinite(a).all() and max(abs(a))<1
 out.append({'voice':voice,'transcript':text,'words':[{'word':w.word,'start':w.start,'end':w.end}for x in segs for w in x.words],'duration':len(a)/s,'wording_verified':True,'intonation_requires_listener_judgment':True});print(voice,text,flush=True)
(r/'verification.json').write_text(json.dumps(out,indent=2))

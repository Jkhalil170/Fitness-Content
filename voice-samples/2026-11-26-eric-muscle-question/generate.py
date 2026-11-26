import os
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from kokoro_onnx import Kokoro
import numpy as np,soundfile as sf,json,subprocess,re
r=Path(__file__).parent;k=Kokoro('/workspace/work/fitness-ranking/tts-models/kokoro-v1.0.onnx','/workspace/work/fitness-ranking/tts-models/voices-v1.0.bin');text='Joseph, which are the hardest muscles for men to build? Abs. Chest. Quads. Triceps. Traps. Calves.';rows=[]
for voice in ['am_eric']:
 a,s=k.create(text,voice=voice,speed=1.0,lang='en-us',sentence_pause=.33,clause_pause=.10,continuous=False,trim=True);assert s==24000 and np.isfinite(a).all();f=r/f'{voice}-raw.wav';sf.write(f,a,s,subtype='PCM_24')
 proc=subprocess.run(['ffmpeg','-v','info','-i',str(f),'-af','loudnorm=I=-18:TP=-2:LRA=11:print_format=json','-f','null','-'],capture_output=True,text=True,check=True);stats=json.loads(re.findall(r'\{[^{}]+\}',proc.stderr)[-1]);gain=min(-18-float(stats['input_i']),-2-float(stats['input_tp']));a=a*10**(gain/20);n=int(.006*s);a[:n]*=np.linspace(0,1,n);a[-n:]*=np.linspace(1,0,n);a=np.pad(a,(int(.08*s),int(.16*s)));wav=r/f'{voice}.wav';sf.write(wav,a,s,subtype='PCM_24');mp3=r/f'{voice}.mp3';subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-c:a','libmp3lame','-b:a','192k','-metadata',f'title={voice} — Joseph opening question and item names',str(mp3)],check=True)
 rows.append({'voice':voice,'text':text,'speed':1.0,'duration':len(a)/s,'sample_rate':s,'input_integrated_lufs':stats['input_i'],'gain_db':gain,'processing':'constant gain only; 6ms edge fades; no time stretch, pitch shift, denoise, compression or limiter','generated_together':True});(r/'manifest.json').write_text(json.dumps({'model':'Kokoro v1.0 82M ONNX','runner':'kokoro-onnx0.6.1','model_license':'Apache-2.0','runner_license':'MIT','paid_services_used':False,'samples':rows},indent=2));print(rows[-1],flush=True)

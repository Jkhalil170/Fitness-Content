import os
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from kokoro_onnx import Kokoro
import soundfile as sf,numpy as np,json,hashlib
r=Path(__file__).parent;k=Kokoro('/workspace/work/fitness-ranking/tts-models/kokoro-v1.0.onnx','/workspace/work/fitness-ranking/tts-models/voices-v1.0.bin');sample=json.load(open('/workspace/work/kokoro-voice-comparison-2026-11-26/samples-manifest.json'));reference=sample['sample_text'];phonemes=k.tokenizer.phonemize(reference,'en-us');batches=k._split_phonemes(phonemes);assert len(batches)==1;tokens=k.tokenizer.tokenize(batches[0]);style=k._style_for(k.get_voice_style('am_liam'),len(tokens));matched_voice=np.repeat(style[None,:],510,axis=0)
text=' '.join(x['announcement']for x in json.loads((r/'narration.json').read_text()));a,sr=k.create(text,voice=matched_voice,speed=1.02,lang='en-us',sentence_pause=.45,clause_pause=.12,continuous=False,trim=True);sf.write(r/'narration/liam-context-full.wav',a,sr,subtype='PCM_16');print('APPROVED LIAM CONDITIONING',len(tokens),'tokens; duration',len(a)/sr,flush=True)
settings=json.loads((r/'narration/context-settings.json').read_text());settings.update({'approved_sample_style_token_count':len(tokens),'approved_sample_style_sha256':hashlib.sha256(style.tobytes()).hexdigest(),'matched_reference_style':True,'note':'Use the exact Liam style vector used to generate the approved four-phrase sample; retain that conditioning across the full food-name context, then split announcements without resynthesis.'});(r/'narration/context-settings.json').write_text(json.dumps(settings,indent=2))

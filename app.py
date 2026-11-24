import streamlit as st, streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, cv2, numpy as np, pandas as pd, threading,time,random, json
from pathlib import Path
from src.engine import Engine
ROOT=Path(__file__).parent
st.set_page_config(page_title='Signify AI',page_icon='🤖',layout='wide')
st.markdown('''<style>.stApp{background:#090d19;color:#e8edff} [data-testid="stHeader"]{background:#090d19} .stButton button{border:1px solid #6466ae;background:#202544;color:white;border-radius:12px} h1,h2,h3{color:#c4b5fd}</style>''',unsafe_allow_html=True)
@st.cache_resource
def engine():return Engine()
class Processor(VideoProcessorBase):
 def __init__(self):self.lock=threading.Lock();self.letter='';self.conf=0.;self.word='';self.message='';self.candidate='';self.since=0.;self.latched='';self.auto=True;self.target='';self.xp=0;self.streak=0;self.learned=set();self.feedback=''
 def recv(self,frame):
  now=time.monotonic()

  if now-getattr(self,'last_inference',0)<0.125:
   return frame

  self.last_inference=now

  img=frame.to_ndarray(format='bgr24')

  if img.shape[1]>640:
   ratio=640/img.shape[1]
   img=cv2.resize(img,(640,int(img.shape[0]*ratio)))

  img,l,p=engine().predict(img)
  with self.lock:
   self.letter=l;self.conf=p
   if not l:self.latched=''
   valid=l if p>=.6 else ''
   if valid!=self.candidate:self.candidate=valid;self.since=now
   if valid and now-self.since>=.8 and self.latched!=valid:
    self.latched=valid
    if self.auto:self.word+=valid
    if self.target:
     if valid==self.target:self.xp+=10;self.streak+=1;self.learned.add(valid);self.feedback='Correct! +10 XP';self.target=''
     else:self.streak=0;self.feedback='Try again'
  return av.VideoFrame.from_ndarray(img,format='bgr24')

st.markdown('''
<style>

[data-testid="stSidebar"] {
    display: none !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stRadio"] {
    background: rgba(18, 24, 45, 0.96);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 16px;
    padding: 12px 18px;
    margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}

[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 12px;
}

[data-testid="stRadio"] label {
    padding: 8px 12px;
    border-radius: 8px;
    transition: all 0.25s ease;
}

[data-testid="stRadio"] label:hover {
    background: rgba(139,92,246,0.18);
    transform: translateY(-2px);
}

[data-testid="stRadio"] label > div:first-child {
    display: none;
}

[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(
        135deg, #7c3aed, #4338ca
    );
}

[data-testid="stImage"] img {
    border-radius: 12px;
    transition: transform 0.3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.04);
}

[data-testid="stMetric"] {
    background: #171d35;
    border: 1px solid #303957;
    border-radius: 15px;
    padding: 20px;
}

.stButton button {
    transition: all 0.25s ease;
}

.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 20px #7c3aed55;
}

</style>

<div style="
    text-align:center;
    font-size:30px;
    font-weight:850;
    letter-spacing:3px;
    margin-bottom:16px;
">
    SIGNIFY
    <span style="color:#a78bfa">AI</span>
</div>
''', unsafe_allow_html=True)

page = st.radio(
    "Navigation",
    [
        "Home",
        "Live Translator",
        "Learn ASL",
        "Image Recognition",
        "ML Analytics",
        "About"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

letters='ABCDEFGHIKLMNOPQRSTUVWXY'
def chart():
 st.subheader('ASL gesture reference')
 with st.expander('Explore alphabet photographs',expanded=True):
  for start in range(0,24,8):
   cols=st.columns(8)
   for c,l in zip(cols,letters[start:start+8]):
    with c:
     p=ROOT/'assets'/'gestures'/f'{l}.jpg'
     if p.exists():st.image(str(p),width='stretch')
     st.caption(l)
  st.caption('J and Z require motion and are not supported. Photos are examples from the training dataset.')
if page=='Home':
 components.html((ROOT/'assets/hero.html').read_text(),height=810,scrolling=True)
 st.markdown('## Communication without barriers')
 photos=list((ROOT/'assets/photos').glob('*')) if (ROOT/'assets/photos').exists() else []
 if photos:
  cols=st.columns(min(3,len(photos)))
  for c,p in zip(cols,photos):c.image(str(p),width='stretch')
 st.write('Experience real-time AI fingerspelling directly below. Turn your gestures into letters, words and messages.')
 a,b,c=st.columns(3);a.metric('Held-out test accuracy','96.91%');b.metric('Live evaluation','85.42%');c.metric('Static letters',24)
 chart()
if page in ['Home','Live Translator','Learn ASL']:
 st.title('Live AI Translator' if page=='Home' else page)
 if page!='Home':chart()
 st.info('Hold a clear gesture for about one second. Remove your hand and sign again to repeat a letter.')
 engine()
 ctx=webrtc_streamer(key='camera-'+page,mode=WebRtcMode.SENDRECV,video_processor_factory=Processor,media_stream_constraints={
        'video': {
            'width': {'ideal': 640, 'max': 640},
            'height': {'ideal': 480, 'max': 480},
            'frameRate': {'ideal': 12, 'max': 15}
        },
        'audio': False
    },rtc_configuration={
        'iceServers': [
            {
                'urls': [
                    'stun:stun.l.google.com:19302',
                    'stun:stun1.l.google.com:19302',
                    'stun:stun2.l.google.com:19302'
                ]
            }
        ]
    },async_processing=True)
 @st.fragment(run_every=1.0)
 def panel():
  p=ctx.video_processor
  if p is None:st.caption('Start the camera to begin.');return
  with p.lock:
   p.auto=page in ['Home','Live Translator'] and st.checkbox('Automatically add stable letters',value=True,key='auto')
   l,conf,word,msg,target,xp,streak,learned,feedback=p.letter,p.conf,p.word,p.message,p.target,p.xp,p.streak,set(p.learned),p.feedback
  a,b=st.columns(2);a.metric('Detected alphabet',l or '—');b.metric('Model confidence',f'{conf:.1%}');st.progress(float(conf))
  if page=='Learn ASL':
   st.metric('Challenge',target or 'Ready');st.write(f'XP: {xp} • Streak: {streak} • Letters mastered: {len(learned)}/24');st.progress(len(learned)/24);st.write(feedback)
   if st.button('New challenge'):
    with p.lock:p.target=random.choice(letters);p.feedback='';p.latched=''
   if target:
    photo=ROOT/'assets/gestures'/f'{target}.jpg'
    if photo.exists():st.image(str(photo),width=140)
   return
  st.subheader('Current word: '+(word or '—'));st.write('Message: '+(msg or '—'))
  a,b,c,d=st.columns(4)
  with p.lock:
   if a.button('Add Letter') and l:p.word+=l
   if b.button('End Word') and p.word:p.message=(p.message+' '+p.word).strip();p.word=''
   if c.button('Backspace'):
    if p.word:p.word=p.word[:-1]
    else:p.message=p.message[:-1]
   if d.button('Clear'):p.word='';p.message=''
  text=(msg+' '+word).strip();st.code(text or 'Start signing',language=None)
  safe=json.dumps(text).replace('<','\\u003c');components.html(f'<button onclick="speechSynthesis.speak(new SpeechSynthesisUtterance({safe.replace(chr(34),"&quot;")}))">Speak message</button>',height=45)
 panel()
elif page=='Image Recognition':
 st.title(page);f=st.file_uploader('Upload a hand gesture',type=['png','jpg','jpeg']);cam=st.camera_input('Or take a photograph')
 if f or cam:
  img=cv2.imdecode(np.frombuffer((f or cam).getvalue(),np.uint8),cv2.IMREAD_COLOR)
  if img is not None:
   out,l,p=engine().predict(img);st.image(out,channels='BGR');st.metric('Prediction',l or 'No hand detected');st.write(f'Confidence: {p:.1%}')
elif page=='ML Analytics':
 st.title('ML intelligence dashboard');a,b,c=st.columns(3);a.metric('Test accuracy','96.91%');b.metric('Macro F1','96.76%');c.metric('Live accuracy','85.42%');st.caption('938 held-out dataset samples; 48 live trials. Results are not guarantees of generalization.')
 for f in sorted((ROOT/'reports').glob('*.csv')):st.subheader(f.stem.replace('_',' ').title());st.dataframe(pd.read_csv(f),width='stretch')
 f=ROOT/'reports/final_confusion_matrix.png'
 if f.exists():st.image(str(f))
else:st.title('About Signify AI');st.write('Static American Sign Language fingerspelling recognition using MediaPipe, engineered hand features and an RBF SVM. The robotic header is an illustrative simulation, not validated ASL instruction. The model does not recognize continuous ASL or motion-based J and Z.')

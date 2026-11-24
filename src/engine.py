import joblib
import cv2, numpy as np, mediapipe as mp, pickle, threading
from itertools import combinations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
def engineer_features(features):
 p=np.asarray(features,dtype=np.float32).reshape(21,3); out=list(p.flatten()); fingers=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]; tips=[4,8,12,16,20]
 def d(a,b):return np.linalg.norm(a-b)
 for f in fingers:
  c=[0]+f
  for i in range(3):
   a,b=p[c[i]]-p[c[i+1]],p[c[i+2]]-p[c[i+1]]; den=np.linalg.norm(a)*np.linalg.norm(b);out.append(np.arccos(np.clip(np.dot(a,b)/den,-1,1))/np.pi if den>=1e-8 else 0.)
 for i,j in combinations(tips,2):out.append(d(p[i],p[j]))
 for i in tips:out.append(d(p[i],p[0]))
 for f in fingers:out.append(d(p[f[0]],p[f[-1]])/(sum(d(p[f[i]],p[f[i+1]]) for i in range(3))+1e-8))
 for j in [6,7,8,10,11,12,14,15,16,18,19,20]:out.append(d(p[4],p[j]))
 return np.asarray(out,dtype=np.float32)
class Engine:
 def __init__(self):
  with open(ROOT/'models/asl_advanced_svm.pkl','rb') as f:self.model=joblib.load(f)
  v=mp.tasks.vision;self.det=v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/hand_landmarker.task')),running_mode=v.RunningMode.IMAGE,num_hands=1,min_hand_detection_confidence=.3,min_hand_presence_confidence=.3));self.lock=threading.Lock()
 def predict(self,bgr):
  h,w=bgr.shape[:2];s=max(h,w);top=(s-h)//2;left=(s-w)//2
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);square=cv2.copyMakeBorder(rgb,top,s-h-top,left,s-w-left,cv2.BORDER_REPLICATE);im=cv2.resize(square,(224,224));lab=cv2.cvtColor(im,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0]);enh=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
  for img,pad in [(im,0),(cv2.copyMakeBorder(im,30,30,30,30,cv2.BORDER_REPLICATE),30),(enh,0)]:
   with self.lock:r=self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(img)))
   if not r.hand_landmarks:continue
   p=np.array([[q.x,q.y,q.z] for q in r.hand_landmarks[0]],dtype=np.float32);xy=(p[:,:2]*(224+2*pad)-pad)/224*s-[left,top];xy=np.rint(xy).astype(int);p-=p[0].copy();p/=max(np.linalg.norm(p,axis=1).max(),1e-8);f=engineer_features(p)[None,:];letter=str(self.model.predict(f)[0]);prob=float(self.model.predict_proba(f)[0][list(self.model.classes_).index(letter)])
   for a,b in EDGES:cv2.line(bgr,tuple(xy[a]),tuple(xy[b]),(255,220,0),2)
   for x,y in xy:cv2.circle(bgr,(x,y),4,(0,220,255),-1)
   lo=np.maximum(xy.min(0)-12,0);hi=np.minimum(xy.max(0)+12,[w-1,h-1]);cv2.rectangle(bgr,tuple(lo),tuple(hi),(255,220,0),2);cv2.putText(bgr,f'{letter} {prob:.0%}',(int(lo[0]),max(20,int(lo[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,220,0),2);return bgr,letter,prob
  return bgr,'',0.

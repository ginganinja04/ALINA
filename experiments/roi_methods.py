"""Small, self-contained ROI methods for the ALINA experiment."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class ROI:
    bl: tuple[float, float]; tl: tuple[float, float]; tr: tuple[float, float]; br: tuple[float, float]
    def corners(self): return np.asarray([self.bl, self.tl, self.tr, self.br], np.float32)
    def valid(self, w, h):
        p = self.corners()
        return bool(np.all(p[:,0]>=0) and np.all(p[:,0]<w) and np.all(p[:,1]>=0) and np.all(p[:,1]<h)
                    and self.tl[1] < self.bl[1] and self.tr[1] < self.br[1]
                    and self.tr[0] > self.tl[0] and self.br[0] > self.bl[0])
    def alina_points(self):
        bl, tl, tr, br = np.round(self.corners()).astype(np.int32)
        return np.asarray([[bl, tl, tl, tr, tr, br]], np.int32)
    @classmethod
    def average(cls, a, b): return cls(*map(tuple, (a.corners()+b.corners())/2))

MANUAL = {
    "vidd_1": ROI((570,760),(782,563),(1040,561),(1456,750)),
    "vidd_2": ROI((780,778),(860,707),(940,709),(1031,791)),
    "vidd_3": ROI((753,776),(844,704),(1016,698),(1251,796)),
}
METHODS = ("fixed", "hsv", "hough", "adaptive", "consensus")

def yellow_mask(image):
    return cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (12,65,70), (42,255,255))

def prior(video, image):
    h,w = image.shape[:2]; scale=np.array([w/1920,h/1080],np.float32)
    pts=np.median(np.stack([r.corners()*scale for k,r in MANUAL.items() if k != video]),axis=0)
    return ROI(*map(tuple,pts))

def fit(points, image):
    h,w=image.shape[:2]
    if len(points)<25: return None
    keep=(points[:,0]>.30*w)&(points[:,0]<.70*w)&(points[:,1]>.48*h)&(points[:,1]<.88*h)
    points=points[keep]
    if len(points)<25: return None
    y=points[:,1]; top,bottom=np.percentile(y,[8,95])
    if bottom-top<.05*h: return None
    slope,intercept=np.polyfit(y,points[:,0],1); xt=slope*top+intercept; xb=slope*bottom+intercept
    residual=np.abs(points[:,0]-(slope*y+intercept)); spread=max(np.percentile(residual,80),.018*w)
    th=max(22,min(.06*w,spread+12)); bh=max(55,min(.16*w,2.2*spread+35))
    roi=ROI((xb-bh,bottom),(xt-th,top),(xt+th,top),(xb+bh,bottom))
    return roi if roi.valid(w,h) else None

class Fixed:
    def __init__(self,video): self.video=video
    def predict(self,image): return prior(self.video,image)

class HSV:
    def __init__(self,video): self.video=video; self.failures=0
    def predict(self,image):
        mask=yellow_mask(image); mask[:int(.42*mask.shape[0])]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        y,x=np.nonzero(mask); roi=fit(np.column_stack((x,y)),image)
        if roi is None: self.failures+=1; return prior(self.video,image)
        return roi

class Hough:
    def __init__(self,video): self.video=video; self.failures=0
    def predict(self,image):
        h,_=image.shape[:2]; near=cv2.dilate(yellow_mask(image),np.ones((7,7),np.uint8))
        edges=cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(5,5),0),60,160); edges[near==0]=0
        lines=cv2.HoughLinesP(edges,1,np.pi/180,25,minLineLength=max(25,h//25),maxLineGap=20); pts=[]
        if lines is not None:
            for x1,y1,x2,y2 in lines.reshape(-1,4):
                a=abs(np.degrees(np.arctan2(y2-y1,x2-x1))); a=min(a,180-a)
                if 25<=a<=88: pts += [(x1,y1),(x2,y2)]
        roi=fit(np.asarray(pts),image) if pts else None
        if roi is None: self.failures+=1; return prior(self.video,image)
        return roi

class Adaptive:
    def __init__(self,video): self.video=video; self.failures=0
    def predict(self,image):
        h,w=image.shape[:2]; g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mask=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,-6); mask[:int(.42*h)]=0
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); pts=[]
        for c in contours:
            if 60<=cv2.contourArea(c)<=.015*h*w: pts += list(map(tuple,c.reshape(-1,2)))
        roi=fit(np.asarray(pts),image) if pts else None
        if roi is None: self.failures+=1; return prior(self.video,image)
        return roi

class Consensus:
    def __init__(self,video): self.video=video; self.a=HSV(video); self.b=Hough(video)
    def predict(self,image): return ROI.average(self.a.predict(image),self.b.predict(image))

def build(name,video): return {"fixed":Fixed,"hsv":HSV,"hough":Hough,"adaptive":Adaptive,"consensus":Consensus}[name](video)

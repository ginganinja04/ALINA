"""One resumable command for all methods, videos, and frames.

Example: python -m experiments.run_batch --output-root outputs/simple_dynamic
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import cv2, numpy as np
from alina.config import PipelineConfig
from alina.pipeline import process_image
from .roi_methods import MANUAL, METHODS, build

VIDEOS={"vidd_1":Path("data/Raw_Data_1080/vidd_1"),"vidd_2":Path("data/Raw_Data/vidd_2"),"vidd_3":Path("data/Raw_Data/vidd_3")}
FIELDS=("video","frame","method","status","had_lines","pixels","roi_ms","alina_ms","error")

def load(path):
    if not path.exists(): return {}
    with path.open(newline="") as f: return {(r["video"],r["frame"],r["method"]):r for r in csv.DictReader(f)}
def save(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows[k] for k in sorted(rows))

def add_visual_evidence(image, roi, manual, method, pixel_count):
    """Add ROI evidence to ALINA's own red-pixel annotated output."""
    def draw(corners, color, label):
        points=np.round(corners).astype(np.int32)
        cv2.polylines(image,[points],True,color,3,cv2.LINE_AA)
        x,y=points[1]
        cv2.putText(image,label,(int(x),int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
    draw(roi.corners(),(0,255,0),"predicted ROI")
    draw(manual.corners(),(255,0,255),"manual ROI")
    text=f"{method}: {pixel_count} ALINA pixels"
    cv2.putText(image,text,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(image,text,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),1,cv2.LINE_AA)
    return image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-root",default="outputs/simple_dynamic")
    ap.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    ap.add_argument("--videos",nargs="+",choices=VIDEOS,default=list(VIDEOS))
    ap.add_argument("--max-frames",type=int); ap.add_argument("--save-images",action="store_true")
    ap.add_argument("--retry-failed",action="store_true"); args=ap.parse_args()
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True)
    settings={"methods":args.methods,"videos":args.videos,"max_frames":args.max_frames}; cfg_path=root/"config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text())!=settings: ap.error("different settings already exist; choose another --output-root")
    cfg_path.write_text(json.dumps(settings,indent=2)+"\n")
    result_path=root/"results.csv"; rows=load(result_path)
    for video in args.videos:
        predictors={m:build(m,video) for m in args.methods}
        paths=sorted(VIDEOS[video].glob("*.jpg"))[:args.max_frames]
        for path in paths:
            image=cv2.imread(str(path))
            if image is None: continue
            for method,predictor in predictors.items():
                key=(video,path.name,method); old=rows.get(key)
                if old and (old["status"]=="ok" or not args.retry_failed): continue
                try:
                    t=time.perf_counter(); roi=predictor.predict(image); roi_ms=(time.perf_counter()-t)*1000
                    dest=root/video/method; config=PipelineConfig(path.parent,dest/"annotated",dest/"coords")
                    t=time.perf_counter(); annotated,coords,found=process_image(image,roi.alina_points(),config); alina_ms=(time.perf_counter()-t)*1000
                    np.savetxt(config.output_coords_dir/f"{path.stem}.txt",coords,fmt="%d")
                    if args.save_images:
                        evidence=add_visual_evidence(annotated,roi,MANUAL[video],method,len(coords))
                        cv2.imwrite(str(config.output_images_dir/path.name),evidence)
                    row=dict(video=video,frame=path.name,method=method,status="ok",had_lines=int(found),pixels=len(coords),roi_ms=f"{roi_ms:.2f}",alina_ms=f"{alina_ms:.2f}",error="")
                except Exception as exc:
                    row=dict(video=video,frame=path.name,method=method,status="failed",had_lines="",pixels="",roi_ms="",alina_ms="",error=f"{type(exc).__name__}: {exc}")
                rows[key]=row; save(result_path,rows); print(f"[{video}/{path.name}/{method}] {row['status']} pixels={row['pixels']}",flush=True)
    print(f"Done: {result_path}")
if __name__=="__main__": main()

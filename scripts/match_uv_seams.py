#!/usr/bin/env python3
"""Match paired material UV boundaries without changing mesh, UV, alpha or detail.

Requires a read-only mesh export and a reviewed job. Colors are corrected in Lab
with a shared boundary profile and a tapered cubic correction on both sides.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from harmonize_aluminum import rgb_to_lab, lab_to_rgb, gaussian, digest, inward_weight
from extract_uv_seams import build as extract_chains


def sample(image, points):
    h,w=image.shape[:2]
    p=np.asarray(points);x=np.clip(p[...,0],0,w-1);y=np.clip(p[...,1],0,h-1)
    x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    fx=(x-x0)[...,None];fy=(y-y0)[...,None]
    return (image[y0,x0]*(1-fx)+image[y0,x1]*fx)*(1-fy)+(image[y1,x0]*(1-fx)+image[y1,x1]*fx)*fy


def pixel(uv, size):
    w,h=size
    return np.asarray(uv)*[w,-h]+[-.5,h-.5]


def resample_side(side,arc,ss):
    p=np.stack([np.interp(ss,arc,side['points'][:,i])for i in range(2)],axis=1)
    n=np.stack([np.interp(ss,arc,side['normals'][:,i])for i in range(2)],axis=1)
    n/=np.linalg.norm(n,axis=1)[:,None]
    return p,n


def apply_band(shape, side, arc, ss, offset, slope, width_units, guard):
    h,w=shape;pts=side['points'];width=width_units*side['px_per_unit']
    # Locate each image pixel relative to its closest actual UV edge segment.
    pad=int(np.ceil(max(width,guard)))+2
    lo=np.maximum(np.floor(pts.min(0)-pad).astype(int),[0,0]);hi=np.minimum(np.ceil(pts.max(0)+pad).astype(int),[w,h])
    yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];grid=np.stack([xx,yy],axis=-1)
    best=np.full(xx.shape,np.inf);coord=np.zeros_like(xx,dtype=float);dist=np.zeros_like(coord)
    endpoint_excess=np.zeros_like(coord)
    for i,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
        v=b-a;raw=np.sum((grid-a)*v,axis=-1)/np.dot(v,v);t=np.clip(raw,0,1)
        near=a+t[...,None]*v;delta=grid-near;d2=np.sum(delta*delta,axis=-1);sel=d2<best
        normal=(1-t[...,None])*side['normals'][i]+t[...,None]*side['normals'][i+1]
        normal/=np.linalg.norm(normal,axis=-1)[...,None]
        best[sel]=d2[sel];coord[sel]=(arc[i]+t*(arc[i+1]-arc[i]))[sel]
        dist[sel]=np.sum(delta*normal,axis=-1)[sel]
        endpoint_excess[sel]=(np.abs(raw-t)*np.linalg.norm(v))[sel]
    t=np.clip(dist/width,0,1)
    delta=np.stack([np.interp(coord,ss,offset[:,c]) for c in range(3)],axis=-1)
    deriv=np.stack([np.interp(coord,ss,slope[:,c]) for c in range(3)],axis=-1)
    correction=delta*(1-3*t*t+2*t*t*t)[...,None]+deriv*width_units*(t-2*t*t+t*t*t)[...,None]
    outside=dist<0
    ext_weight=np.clip(1+dist/guard,0,1);ext_weight=ext_weight**2*(3-2*ext_weight)
    correction[outside]=(delta+deriv*(dist/side['px_per_unit'])[...,None])[outside]*ext_weight[outside,None]
    active=(dist>=-guard)&(dist<=width)&(endpoint_excess<=guard)
    # A short continuation beyond the ends supports bilinear filtering at bevels.
    cap=np.clip(1-endpoint_excess/guard,0,1);cap=cap*cap*(3-2*cap)
    correction*=active[...,None]*cap[...,None]
    full=np.zeros((h,w,3));full[lo[1]:hi[1],lo[0]:hi[0]]=correction
    mask=np.zeros((h,w),bool);mask[lo[1]:hi[1],lo[0]:hi[0]]=active&(cap>0)
    return full,mask


def run(job_path):
    job=json.loads(Path(job_path).read_text());scene=json.loads(Path(job['scene']).read_text())
    output=Path(job['output_dir']);records=Path(job['records_dir'])
    if records.resolve()==output.resolve() or output.resolve()in records.resolve().parents:raise ValueError('Keep records outside delivery')
    ims={n:Image.open(v['source']).convert('RGBA')for n,v in job['views'].items()}
    originals={n:np.array(im)for n,im in ims.items()}
    if any((output/(n+'.png')).exists()for n in ims):raise FileExistsError('Choose new output directory')
    sources=[Path(v['source']).resolve()for v in job['views'].values()]
    if any((output/(n+'.png')).resolve()in sources for n in ims):raise ValueError('Cannot overwrite source')
    hashes={n:digest(v['source'])for n,v in job['views'].items()}
    labs={n:rgb_to_lab(a[...,:3])for n,a in originals.items()}
    allows={};protects={};low={};weights={}
    for n,a in originals.items():
        v=job['views'][n]
        allow=np.array(Image.open(v['editable_mask']).convert('L'))>0
        protect=np.array(Image.open(v['protected_mask']).convert('L'))>0
        if allow.shape!=a.shape[:2]or protect.shape!=a.shape[:2]:raise ValueError('Mask dimensions differ')
        allows[n]=allow&(a[...,3]>0)&~protect;protects[n]=protect
        # Exclude nonmetal and transparency from local low-frequency samples.
        fit=allows[n]&(a[...,3]>=240);den=gaussian(fit.astype(float),job.get('sigma',3))
        low[n]=gaussian(labs[n]*fit[...,None],job.get('sigma',3))/np.maximum(den[...,None],1e-8)
        weights[n]=inward_weight(allows[n],job.get('protection_feather',3))
    if job.get('seam_chains'):
        paired=json.loads(Path(job['seam_chains']).read_text())
        if isinstance(paired,dict):paired=paired['chains']
        for c in paired:
            c['arc']=np.array(c['arc'],dtype=float)
            for s in c['sides']:
                s['points']=np.array(s['points'],dtype=float);s['normals']=np.array(s['normals'],dtype=float)
    else:
        for material in scene['materials']:
            if material['name'] in job['views']:
                for node in material['nodes']:
                    if node['type']=='ShaderNodeTexImage':node['path']=job['views'][material['name']]['source']
        paired=extract_chains(scene)['chains']
        for c in paired:
            c['arc']=np.array(c['arc'],dtype=float)
            for s in c['sides']:
                s['points']=np.array(s['points'],dtype=float);s['normals']=np.array(s['normals'],dtype=float)
    if not paired:raise ValueError('No paired UV seams found')
    for chain in paired:
        if not np.all(np.diff(chain['arc'])>0):raise ValueError('Seam arc must be strictly increasing')
        for side in chain['sides']:
            if side.get('image_size') and tuple(side['image_size'])!=ims[side['name']].size:
                raise ValueError('Seam UV pixel coordinates refer to a different image size')
    fields={n:np.zeros_like(l)for n,l in labs.items()};bands={n:np.zeros(a.shape[:2],bool)for n,a in originals.items()}
    info=[]
    for chain in paired:
        arc=chain['arc'];ss=np.linspace(0,arc[-1],192);samples=[];derivatives=[]
        metric_valid=np.ones(len(ss),bool)
        for side in chain['sides']:
            p,n=resample_side(side,arc,ss);step=job.get('derivative_step_units',.015)*side['px_per_unit']
            values=sample(low[side['name']],p)
            derivative=(sample(low[side['name']],p+n*step)-values)/job.get('derivative_step_units',.015)
            samples.append(values);derivatives.append(derivative)
            metric_valid &= sample(weights[side['name']][...,None],p)[:,0]>.98
        target=(samples[0]+samples[1])/2
        desired=[(derivatives[0]-derivatives[1])/2,(derivatives[1]-derivatives[0])/2]
        for k,side in enumerate(chain['sides']):
            offset=np.clip(target-samples[k],-np.array(job.get('max_lab_shift',[18,12,12])),np.array(job.get('max_lab_shift',[18,12,12])))
            slope=desired[k]-derivatives[k]
            # Limit derivative corrections so the transition cannot overshoot heavily.
            lim=np.array(job.get('max_slope_effect',[16,10,10]))/job.get('width_units',.20)
            slope=np.clip(slope,-lim,lim)
            field,mask=apply_band(originals[side['name']].shape[:2],side,arc,ss,offset,slope,job.get('width_units',.20),job.get('guard_pixels',6))
            if np.any(bands[side['name']]&mask):raise ValueError('Overlapping seam bands need joint solve')
            fields[side['name']]+=field;bands[side['name']]|=mask
        before=np.linalg.norm(samples[0]-samples[1],axis=-1)[metric_valid]
        if not len(before):raise ValueError('Seam contains no jointly editable samples')
        info.append({'pair':chain['names'],'edge_ids':chain['edges'],'before_mean_deltaE76':float(before.mean()),'before_p95_deltaE76':float(np.quantile(before,.95)),
                     'metric_samples':int(metric_valid.sum()),'width_pixels':{s['name']:job.get('width_units',.20)*s['px_per_unit']for s in chain['sides']}})
    output.mkdir(parents=True,exist_ok=True);records.mkdir(parents=True,exist_ok=True)
    checks={};outputs={}
    for n,src in originals.items():
        effective=allows[n]&bands[n]
        corrected=lab_to_rgb(labs[n]+fields[n]*weights[n][...,None]);dst=src.copy();dst[effective,:3]=corrected[effective]
        Image.fromarray(dst).save(output/(n+'.png'))
        saved=np.array(Image.open(output/(n+'.png')).convert('RGBA'))
        assert np.array_equal(saved[...,3],src[...,3])
        assert np.array_equal(saved[~effective],src[~effective])
        assert np.array_equal(saved[protects[n]],src[protects[n]])
        assert digest(job['views'][n]['source'])==hashes[n]
        outputs[n]=rgb_to_lab(saved[...,:3])
        Image.fromarray(effective.astype(np.uint8)*255).save(records/(n+'-seam-mask.png'))
        checks[n]={'source':job['views'][n]['source'],'source_sha256':hashes[n],'output':str(output/(n+'.png')),'size':list(ims[n].size),'alpha_changed':0,'protected_changed':0,'outside_changed':0,'changed_pixels':int(np.any(saved!=src,axis=-1).sum())}
    for chain,entry in zip(paired,info):
        arc=chain['arc'];ss=np.linspace(0,arc[-1],192);samples=[]
        metric_valid=np.ones(len(ss),bool)
        for side in chain['sides']:
            p,n=resample_side(side,arc,ss)
            name=side['name'];fit=allows[name]&(originals[name][...,3]>=240)
            den=gaussian(fit.astype(float),job.get('sigma',3))
            normalized=gaussian(outputs[name]*fit[...,None],job.get('sigma',3))/np.maximum(den[...,None],1e-8)
            samples.append(sample(normalized,p))
            metric_valid &= sample(weights[name][...,None],p)[:,0]>.98
        diff=np.linalg.norm(samples[0]-samples[1],axis=-1)[metric_valid]
        entry.update(after_mean_deltaE76=float(diff.mean()),after_p95_deltaE76=float(np.quantile(diff,.95)))
    report={'seams':info,'checks':checks,'validation_scope':'actual paired mesh UV samples; render separately'}
    (records/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (records/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--job',required=True);run(ap.parse_args().job)

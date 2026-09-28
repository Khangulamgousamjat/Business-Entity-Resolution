#!/usr/bin/env python3
"""
Ultra-Scale Business Entity Resolution Pipeline
High-Throughput Matching & Deduplication Engine (Scalable to 10M+ Records)

Scoring Formula (Macro-averaged F_0.5):
  Per entity:
    - True matches == 0 AND Predicted == 0  -> score = 1.0  (correct singleton)
    - True matches == 0 AND Predicted  > 0  -> score = 0.0  (false positive)
    - True matches  > 0 AND Predicted == 0  -> score = 0.0  (missed match)
    - Otherwise: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
  Final score = mean of per-entity scores across ALL S1 entities
  Precision is weighted 2x over recall - AVOID FALSE POSITIVES

BUGS FIXED vs original:
  - blocker.cands() -> blocker.candidates()
  - compute_features() -> feats()
  - Blocker.recs stores RAW strings so feats() can internally clean correctly
  - target dict preserved during inference for feature computation
"""

import argparse, json, os, re, sys, unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import joblib
import numpy as np

try:
    from lightgbm import LGBMClassifier
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError as e:
    print(f"[ERR] Missing package: {e}. Run: pip install lightgbm rapidfuzz")
    sys.exit(1)

# ===================================================
# PREPROCESSING
# ===================================================

LEGAL_RE = re.compile(
    r'\b(private limited|pvt\.? ?ltd\.?|ltd\.?|limited|incorporated|'
    r'inc\.?|corp\.?|corporation|llc|llp|co\.?|company|'
    r'sarl|sas|eurl|snc|sci|sa|soc|ste|societe|'
    r'traders|enterprises|brothers|bros)\b',
    flags=re.IGNORECASE)

ADDR_ABBR = [(re.compile(p, re.I), r) for p, r in {
    r'\brd\b':'road', r'\bst\b':'street', r'\bave?\b':'avenue',
    r'\bblvd\b':'boulevard', r'\bapt\b':'apartment', r'\bdr\b':'drive',
    r'\bln\b':'lane', r'\bct\b':'court', r'\bpkwy\b':'parkway',
    r'\bhwy\b':'highway', r'\bpl\b':'place', r'\bsq\b':'square',
    r'\bste\b':'suite', r'\bflr\b':'floor',
}.items()]

STOPS = {
    'the','and','of','in','at','to','a','an','for','on','by','with',
    'from','is','it','its','as','be','was','are',
    'de','du','des','le','la','les','et',
    'ki','ke','ka','ko','se','aur',
}

def strip_acc(t):
    if not t: return ''
    return ''.join(c for c in unicodedata.normalize('NFKD',t) if not unicodedata.combining(c))

def clean(t):
    if not t or not isinstance(t,str): return ''
    t = strip_acc(t.lower()).replace('&',' and ')
    t = re.sub(r'[^\w\s]',' ',t)
    return re.sub(r'\s+',' ',t).strip()

def clean_name(n):
    t = clean(n)
    t = LEGAL_RE.sub(' ',t)
    return re.sub(r'\s+',' ',t).strip()

def clean_addr(a):
    t = clean(a)
    for pat,rep in ADDR_ABBR:
        t = pat.sub(rep,t)
    return re.sub(r'\s+',' ',t).strip()

def toks(t, ml=3):
    return [w for w in t.split() if len(w)>=ml and w not in STOPS]

def bigrams(t):
    ws=t.split()
    return {f'{ws[i]} {ws[i+1]}' for i in range(len(ws)-1)} if len(ws)>=2 else set()

def nums(t):
    if not t: return set()
    return set(re.findall(r'\b\d{2,}\b',t))


# ===================================================
# BLOCKING
# ===================================================

class Blocker:
    def __init__(self, max_freq=3000, top_k=30):
        self.max_freq=max_freq; self.top_k=top_k
        self.idx=defaultdict(lambda:defaultdict(list))
        self.freq=defaultdict(lambda:defaultdict(int))
        self.recs={}  # eid->(raw_name, raw_addr) - stored RAW for feats()

    def fit(self, records):
        """records: list of (eid, raw_name, raw_addr, country)"""
        for eid,name,addr,country in records:
            c=country.strip().upper()
            cn=clean_name(name); ca=clean_addr(addr)
            self.recs[eid]=(name,addr)  # store RAW strings
            keys=(
                {f'n:{t}' for t in set(toks(cn,3))} |
                {f'a:{t}' for t in set(toks(ca,4))} |
                {f'b:{b}' for b in bigrams(cn)} |
                {f'z:{n}' for n in nums(addr)}
            )
            for k in keys:
                self.freq[c][k]+=1
                self.idx[c][k].append(eid)

    def candidates(self, name, addr, country):
        """Get top-K candidate IDs for a query entity."""
        c=country.strip().upper()
        ci=self.idx.get(c)
        if not ci: return []
        cn=clean_name(name); ca=clean_addr(addr)
        sc=defaultdict(float)
        for t,w in (
            [(f'n:{t}',3.0) for t in set(toks(cn,3))]+
            [(f'a:{t}',1.0) for t in set(toks(ca,4))]+
            [(f'b:{b}',5.0) for b in bigrams(cn)]+
            [(f'z:{n}',2.5) for n in nums(addr)]
        ):
            fr=self.freq[c].get(t,0)
            if 0<fr<=self.max_freq:
                idf=w/(fr**0.5)
                for eid in ci[t]:
                    sc[eid]+=idf
        if not sc: return []
        return sorted(sc,key=lambda x:-sc[x])[:self.top_k]


# ===================================================
# FEATURES (19)
# ===================================================

FEAT_NAMES=[
    'name_ratio','name_tok_sort','name_tok_set','name_jaro',
    'name_tok_jac','name_ng3','name_ng2','name_len_diff',
    'addr_ratio','addr_tok_sort','addr_tok_jac','addr_ng3',
    'addr_len_diff','addr_missing',
    'num_jac','num_match','num_conflict',
    'prefix4','is_s3'
]

def jac(a,b):
    if not a or not b: return 0.0
    u=len(a|b); return len(a&b)/u if u else 0.0

def ngjac(s1,s2,n=3):
    if not s1 or not s2: return 0.0
    g1={s1[i:i+n] for i in range(max(0,len(s1)-n+1))} or {s1}
    g2={s2[i:i+n] for i in range(max(0,len(s2)-n+1))} or {s2}
    return jac(g1,g2)

def feats(s1n, s1a, cn, ca, cid=''):
    """
    Compute 19-feature vector. All inputs are RAW strings (cleaned internally).
    s1n, s1a: Source 1 raw name/address
    cn,  ca:  Candidate raw name/address
    cid:      Candidate entity_id (for is_s3 flag)
    """
    cn1,cn2=clean_name(s1n),clean_name(cn)
    ca1,ca2=clean_addr(s1a),clean_addr(ca)
    nr=fuzz.ratio(cn1,cn2)/100
    nts=fuzz.token_sort_ratio(cn1,cn2)/100
    ntset=fuzz.token_set_ratio(cn1,cn2)/100
    nj=JaroWinkler.similarity(cn1,cn2)
    t1=set(toks(cn1,2)); t2=set(toks(cn2,2))
    ntjac=jac(t1,t2)
    ng3=ngjac(cn1,cn2,3); ng2=ngjac(cn1,cn2,2)
    nld=abs(len(cn1)-len(cn2))/(max(len(cn1),len(cn2))+1e-5)
    am=1.0 if (not ca1 or not ca2) else 0.0
    if am:
        ar=ats=atjac=ang3=ald=0.0; ald=1.0
    else:
        ar=fuzz.ratio(ca1,ca2)/100
        ats=fuzz.token_sort_ratio(ca1,ca2)/100
        at1=set(toks(ca1,3)); at2=set(toks(ca2,3))
        atjac=jac(at1,at2); ang3=ngjac(ca1,ca2,3)
        ald=abs(len(ca1)-len(ca2))/(max(len(ca1),len(ca2))+1e-5)
    n1=nums(s1a); n2=nums(ca)
    njac=jac(n1,n2)
    nm=float(len(n1&n2))
    nc=1.0 if (n1 and n2 and not n1&n2) else 0.0
    pref=1.0 if (cn1[:4]==cn2[:4] and len(cn1)>=4 and len(cn2)>=4) else 0.0
    is3=1.0 if cid.startswith('S3-') else 0.0
    return [nr,nts,ntset,nj,ntjac,ng3,ng2,nld,ar,ats,atjac,ang3,ald,am,njac,nm,nc,pref,is3]


# ===================================================
# EVALUATION
# ===================================================

def f05(p,r):
    """F_0.5 = (1.25*P*R)/(0.25*P+R)  - precision weighted 2x"""
    return 1.25*p*r/(0.25*p+r) if (p+r)>0 else 0.0

def ent_f05(gt_set,pred_set):
    """
    Per-entity scoring:
      (empty, empty) -> 1.0  correct singleton
      (empty, non-empty) -> 0.0  false positive on singleton
      (non-empty, empty) -> 0.0  missed all matches
      otherwise -> F_0.5(precision, recall)
    """
    if not gt_set and not pred_set: return 1.0
    if not gt_set: return 0.0
    if not pred_set: return 0.0
    tp=len(gt_set&pred_set)
    if tp==0: return 0.0
    return f05(tp/len(pred_set), tp/len(gt_set))

def macro_f05(gt,preds):
    return float(np.mean([ent_f05(gt[s],preds.get(s,set())) for s in gt]))


# ===================================================
# DATA I/O
# ===================================================

def load_gt(path, max_rows=None):
    gt={}
    with open(path,'r',encoding='utf-8') as f:
        f.readline()
        for line in f:
            p=line.rstrip('\r\n').split('\t')
            if not p: continue
            gt[p[0]]=set(m.strip() for m in (p[1] if len(p)>1 else '').split(',') if m.strip())
            if max_rows and len(gt)>=max_rows: break
    return gt

def single_pass_country(path, country):
    """One pass: return {eid:(raw_name,raw_addr,country)} for matching country."""
    recs = {}
    cu = country.strip().upper()
    with open(path, 'r', encoding='utf-8') as f:
        h = f.readline().rstrip('\r\n').split('\t')
        ei = h.index('entity_id')        if 'entity_id'        in h else 0
        ni = h.index('business_name')    if 'business_name'    in h else 1
        ai = h.index('business_address') if 'business_address' in h else 2
        ci = h.index('country')          if 'country'          in h else 3
        mi = max(ei, ni, ai, ci)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) <= mi: continue
            if p[ci].strip().upper() != cu: continue
            recs[p[ei]] = (p[ni], p[ai], p[ci])
    return recs

def discover_countries(path):
    cs=set()
    with open(path,'r',encoding='utf-8') as f:
        h=f.readline().rstrip('\r\n').split('\t')
        ci=h.index('country') if 'country' in h else 3
        for line in f:
            p=line.rstrip('\r\n').split('\t')
            if len(p)>ci: cs.add(p[ci].strip())
    return cs


# ===================================================
# TRAINING
# ===================================================

def train(train_dir, model_dir, val_ratio=0.15):
    print('='*70)
    print('MAXIMUM-POWER TRAINING - ALL 2.2M ENTITIES')
    print('='*70)
    print()
    print('SCORING FORMULA:')
    print('  F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)')
    print('  Per-entity: correct singleton=1.0, false positive on singleton=0.0')
    print('  PRECISION weighted 2x over recall - avoid false positives!')
    print()
    os.makedirs(model_dir, exist_ok=True)

    gt_path=os.path.join(train_dir,'train_ground_truth.tsv')
    s1_path=os.path.join(train_dir,'train_source1.tsv')
    s2_path=os.path.join(train_dir,'train_source2.tsv')
    s3_path=os.path.join(train_dir,'train_source3.tsv')

    print('\n[1/6] Loading ALL ground truth...')
    gt_map=load_gt(gt_path)
    print(f'    {len(gt_map):,} ground truth entries loaded.')

    all_s1=list(gt_map.keys())
    np.random.seed(42); np.random.shuffle(all_s1)
    n_val=max(5000, int(len(all_s1)*val_ratio))
    val_ids=set(all_s1[:n_val])
    train_ids=set(all_s1[n_val:])
    print(f'    Train: {len(train_ids):,}  Val: {len(val_ids):,}')

    print('\n[2/6] Discovering countries...')
    countries=sorted(discover_countries(s1_path))
    print(f'    Countries: {countries}')

    print('\n[3/6] Extracting features COUNTRY BY COUNTRY...')
    X_train_all=[]; y_train_all=[]
    X_val_all=[];   y_val_all=[]
    val_entity_data={}

    for country in countries:
        print(f'\n  -- Country: {country} --')
        s1_recs={}
        with open(s1_path,'r',encoding='utf-8') as f:
            h=f.readline().rstrip('\r\n').split('\t')
            ei=h.index('entity_id') if 'entity_id' in h else 0
            ni=h.index('business_name') if 'business_name' in h else 1
            ai=h.index('business_address') if 'business_address' in h else 2
            ci=h.index('country') if 'country' in h else 3
            for line in f:
                p=line.rstrip('\r\n').split('\t')
                if len(p)<=max(ei,ni,ai,ci): continue
                if p[ci].strip().upper()!=country.upper(): continue
                eid=p[ei]
                if eid in gt_map:
                    s1_recs[eid]=(p[ni],p[ai],p[ci])
        print(f'    S1 with GT: {len(s1_recs):,}')
        if not s1_recs: continue

        print(f'    Loading S2...', flush=True)
        s2c = single_pass_country(s2_path, country)
        print(f'    S2: {len(s2c):,}', flush=True)
        print(f'    Loading S3...', flush=True)
        s3c = single_pass_country(s3_path, country)
        print(f'    S3: {len(s3c):,}', flush=True)
        target_recs = {**s2c, **s3c}
        del s2c, s3c
        print(f'    Target pool: {len(target_recs):,}', flush=True)

        blocker=Blocker(max_freq=3000, top_k=25)
        blocker.fit([(eid,v[0],v[1],v[2]) for eid,v in target_recs.items()])
        print(f'    Blocker built.', flush=True)

        country_train_X=[]; country_train_y=[]
        country_val_X=[];   country_val_y=[]
        n_proc=0

        for sid in s1_recs:
            s1n,s1a,s1c=s1_recs[sid]
            true_m=gt_map.get(sid,set())
            is_val=sid in val_ids

            cands=set(blocker.candidates(s1n,s1a,s1c))
            if not is_val:
                cands|=true_m  # inject true positives for training

            for cid in cands:
                if cid not in target_recs: continue
                cn,ca,_=target_recs[cid]
                fv=feats(s1n,s1a,cn,ca,cid)  # RAW strings - feats() cleans internally
                label=1 if cid in true_m else 0
                if is_val:
                    country_val_X.append(fv); country_val_y.append(label)
                else:
                    country_train_X.append(fv); country_train_y.append(label)

            if is_val:
                val_entity_data[sid]=(list(cands), true_m)

            n_proc+=1
            if n_proc%100000==0:
                print(f'    ...{n_proc:,} processed...', flush=True)

        print(f'    {country} train: {len(country_train_X):,}  val: {len(country_val_X):,}', flush=True)
        X_train_all.extend(country_train_X); y_train_all.extend(country_train_y)
        X_val_all.extend(country_val_X);     y_val_all.extend(country_val_y)
        del target_recs, blocker, country_train_X, country_val_X

    X_train=np.array(X_train_all,dtype=np.float32); y_train=np.array(y_train_all)
    X_val  =np.array(X_val_all,  dtype=np.float32); y_val  =np.array(y_val_all)
    del X_train_all, y_train_all, X_val_all, y_val_all

    pos=np.sum(y_train==1); neg=np.sum(y_train==0)
    print(f'\n  TOTAL TRAIN: {len(y_train):,} pairs  Pos:{pos:,}  Neg:{neg:,}')
    print(f'  TOTAL VAL  : {len(y_val):,} pairs  Pos:{np.sum(y_val==1):,}')

    print('\n[4/6] Training LightGBM on ALL data...')
    spw=min(neg/max(pos,1), 40.0)
    print(f'    scale_pos_weight = {spw:.2f}')
    model=LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30,
        scale_pos_weight=spw,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_train,y_train,eval_set=[(X_val,y_val)],eval_metric='binary_logloss')
    imps=dict(zip(FEAT_NAMES,model.feature_importances_.tolist()))
    print('\n  Top features:')
    for k,v in sorted(imps.items(),key=lambda x:-x[1])[:8]:
        print(f'    {k:25s}: {v}')

    print('\n[5/6] Threshold sweep (optimising macro-F0.5)...')
    val_probs=model.predict_proba(X_val)[:,1]
    del X_train, X_val

    val_idx=0
    val_ep={}
    for sid,(cands,true_m) in val_entity_data.items():
        n=len(cands)
        probs_slice=val_probs[val_idx:val_idx+n]
        pairs=list(zip(cands,probs_slice.tolist()))
        val_idx+=n
        val_ep[sid]=(pairs,true_m)

    val_gt_sub={sid:tm for sid,(_,tm) in val_ep.items()}

    best_t,best_f=0.5,0.0
    print('  thresh  | F0.5')
    print('  --------+------')
    for t in np.arange(0.20,0.97,0.02):
        t=round(float(t),2)
        preds={sid:{cid for cid,p in pairs if p>=t} for sid,(pairs,_) in val_ep.items()}
        sc=macro_f05(val_gt_sub,preds)
        print(f'   {t:.2f}    | {sc:.4f}')
        if sc>best_f: best_f=sc; best_t=t
    print(f'\n  BEST THRESHOLD: {best_t:.2f}  -->  Val F0.5: {best_f:.4f}')

    mp=os.path.join(model_dir,'lgbm_max.joblib')
    mmp=os.path.join(model_dir,'max_metadata.json')
    joblib.dump(model,mp)
    with open(mmp,'w') as f:
        json.dump({'optimal_threshold':best_t,'val_f05':round(best_f,5),
                   'n_train':int(len(y_train)),'features':FEAT_NAMES,'importances':imps},f,indent=2)
    print(f'[OK] Model: {mp}')
    print(f'[OK] Meta:  {mmp}')
    return model,best_t


# ===================================================
# INFERENCE (BUGS FIXED)
# ===================================================

def infer(test_dir, output_dir, model, threshold):
    print('\n'+'='*70)
    print(f'INFERENCE ON TEST SET  (threshold={threshold:.2f})')
    print('='*70)
    os.makedirs(output_dir,exist_ok=True)
    s1p=os.path.join(test_dir,'test_source1.tsv')
    s2p=os.path.join(test_dir,'test_source2.tsv')
    s3p=os.path.join(test_dir,'test_source3.tsv')
    mp=os.path.join(output_dir,'matching_results.tsv')
    cp=os.path.join(output_dir,'candidate_pairs.tsv')

    countries=sorted(discover_countries(s1p))
    print(f'  Countries: {countries}')

    total=matched=singletons=0
    with open(mp,'w',encoding='utf-8',newline='\n') as fm,\
         open(cp,'w',encoding='utf-8',newline='\n') as fc:
        fm.write('source1_entity_id\tmatched_entity_ids\n')
        fc.write('source1_entity_id\tcandidate_entity_ids\n')

        for country in countries:
            print(f'\n--- [{country}] ---', flush=True)
            s2c=single_pass_country(s2p,country)
            s3c=single_pass_country(s3p,country)
            # target: eid -> (raw_name, raw_addr, country)
            target={**s2c,**s3c}
            del s2c, s3c
            print(f'  Targets: {len(target):,}', flush=True)

            # Blocker stores RAW strings internally
            blocker=Blocker(max_freq=3000,top_k=30)
            blocker.fit([(eid,v[0],v[1],v[2]) for eid,v in target.items()])
            print(f'  Blocker ready.', flush=True)
            part=pm=ps=0

            with open(s1p,'r',encoding='utf-8') as f:
                h=f.readline().rstrip('\r\n').split('\t')
                ei=h.index('entity_id') if 'entity_id' in h else 0
                ni=h.index('business_name') if 'business_name' in h else 1
                ai=h.index('business_address') if 'business_address' in h else 2
                ci=h.index('country') if 'country' in h else 3
                mi=max(ei,ni,ai,ci)
                for line in f:
                    p=line.rstrip('\r\n').split('\t')
                    if len(p)<=mi: continue
                    if p[ci].strip().upper()!=country.upper(): continue
                    sid,sn,sa=p[ei],p[ni],p[ai]

                    # FIX 1: blocker.candidates() not blocker.cands()
                    clist=blocker.candidates(sn,sa,country)
                    valid=[c for c in clist if c in target]

                    if not valid:
                        fm.write(f'{sid}\t\n'); fc.write(f'{sid}\t\n')
                        part+=1; total+=1; ps+=1; singletons+=1; continue

                    # FIX 2: feats() not compute_features(), pass RAW strings from target
                    X=np.array([feats(sn,sa,target[c][0],target[c][1],c) for c in valid],dtype=np.float32)
                    probs=model.predict_proba(X)[:,1]
                    matched_ids=[c for c,pr in zip(valid,probs) if pr>=threshold]
                    fm.write(f'{sid}\t{",".join(matched_ids)}\n')
                    fc.write(f'{sid}\t{",".join(valid)}\n')
                    part+=1; total+=1
                    if matched_ids: pm+=1; matched+=1
                    else: ps+=1; singletons+=1
                    if part%100000==0:
                        print(f'  {part:,} done...', flush=True)

            del blocker, target
            print(f'  [{country}] {part:,} entities | Matched:{pm:,} | Singletons:{ps:,}', flush=True)

    print(f'\nDONE. Total:{total:,} Matched:{matched:,} Singletons:{singletons:,}')
    print(f'Submit: {mp}')


# ===================================================
# MAIN
# ===================================================

def main():
    ap=argparse.ArgumentParser(description='Max-Power Entity Resolution (Fixed)')
    ap.add_argument('--train-dir',  default='train')
    ap.add_argument('--test-dir',   default='test')
    ap.add_argument('--output-dir', default='output')
    ap.add_argument('--model-dir',  default='max_models')
    ap.add_argument('--skip-train', action='store_true',
                    help='Skip training and load existing model')
    ap.add_argument('--threshold',  type=float, default=None,
                    help='Override the saved optimal threshold')
    args=ap.parse_args()

    mp=os.path.join(args.model_dir,'lgbm_max.joblib')
    mmp=os.path.join(args.model_dir,'max_metadata.json')

    if args.skip_train and os.path.exists(mp):
        print(f'[*] Loading model from {mp}')
        model=joblib.load(mp)
        with open(mmp) as f: meta=json.load(f)
        threshold=args.threshold or meta['optimal_threshold']
        print(f'[*] Threshold: {threshold:.2f}  Val F0.5: {meta.get("val_f05","?")}')
    else:
        model,threshold=train(args.train_dir,args.model_dir)
        if args.threshold: threshold=args.threshold

    infer(args.test_dir,args.output_dir,model,threshold)

if __name__=='__main__':
    main()

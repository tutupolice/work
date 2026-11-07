"""Non-destructive reconciliation of the documented 48-control candidate rule.

Uses the complete stored inner-CV results for all 48 control candidates in each
fold. Re-evaluates each selected candidate's 3 inner splits to test reproducibility,
and refits all 50 outer selected models. Original outputs are never overwritten.
This does not rerun all 7200 inner fits; the provenance distinction is reported.
"""
from pathlib import Path
import sys,importlib,json,hashlib,collections,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
m=importlib.import_module('03r_injury_nested_joint_selection')
OLD=m.OUT
NEW=Path(__file__).parent/'nested_48_reconciliation'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def one(f):
 old=json.loads(f.read_text()); fold=old['fold']; player=old['test_player']
 pool=[x for x in old['inner_candidate_results'] if x['strategy']=='control']
 key=lambda x:(x['window'],x['model'],x['strategy'])
 expected={key(c):c for c in m.make_candidates()}
 assert len(pool)==48 and {key(c) for c in pool}==set(expected)
 best=m.choose(pool); candidate=expected[key(best)]
 frames={w:pd.read_parquet(m.SOURCE_OUT/f'history_{w}d.parquet') for w in m.WINDOWS}
 frame=frames[candidate['window']]
 train_players=set(frame.player_name.unique())-{player}
 check=m.evaluate_candidate(candidate,frames,train_players)
 delta=max(abs(np.array(check['inner_AP_folds'])-np.array(best['inner_AP_folds'])))
 delta_auc=max(abs(np.array(check['inner_ROC_AUC_folds'])-np.array(best['inner_ROC_AUC_folds'])))
 if delta>1e-8 or delta_auc>1e-8:
  raise ValueError(f'fold {fold} inner recomputation mismatch AP={delta}, AUC={delta_auc}')
 train=np.flatnonzero(frame.player_name.ne(player)); test=np.flatnonzero(frame.player_name.eq(player))
 with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
  score,notes=m.tune.fit_predict(frame,m.feature_columns(frame),train,test,candidate['model'],{'strategy':'control','params':candidate['params']})
 pred=frame.iloc[test][['player_name','date','injury']].copy();pred['score']=score
 pred['selected_configuration']=f"{candidate['window']}d_{candidate['model']}_control"
 pred.to_csv(NEW/f'fold_{fold:02d}_predictions.csv',index=False)
 meta={'fold':fold,'original_inner_candidate_count':len(old['inner_candidate_results']),'candidate_count':48,'source_selection_file':str(f),'source_sha256':sha(f),'selected':candidate,'inner_check_max_AP_difference':float(delta),'inner_check_max_AUC_difference':float(delta_auc),'changed_selection':key(candidate)!=key(old['selected_configuration']),'warnings':notes+[str(w.message) for w in caught]}
 (NEW/f'fold_{fold:02d}_audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 return meta
if __name__=='__main__':
 NEW.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((OLD/'NESTED_RUN_MANIFEST.json').read_text())
 for w,h in manifest['input_history_sha256'].items(): assert sha(m.SOURCE_OUT/f'history_{w}d.parquet')==h
 records=[]; errors=[]
 with ProcessPoolExecutor(4) as ex:
  jobs={ex.submit(one,f):f for f in sorted(OLD.glob('outer_fold_*_selection.json'))}
  for job in as_completed(jobs):
   try: records.append(job.result()); print('completed',len(records),'fold',records[-1]['fold'],flush=True)
   except Exception as e: errors.append({'file':str(jobs[job]),'error':str(e)});print('ERROR',str(e),flush=True)
 if errors:
  (NEW/'FAILED_AUDIT.json').write_text(json.dumps(errors,indent=2));raise RuntimeError('Reconciliation did not pass; do not report final metrics')
 pred=pd.concat([pd.read_csv(p) for p in sorted(NEW.glob('fold_*_predictions.csv'))],ignore_index=True)
 assert len(records)==50 and len(pred)==19613 and pred.injury.sum()==149 and pred.player_name.nunique()==50 and not pred.duplicated(['player_name','date']).any()
 pred.to_csv(NEW/'nested_48_oof_predictions.csv',index=False)
 result={'method':'48 control configurations; cached complete inner results filtered to prespecified control rule; all 50 selected inner evaluations reproduced and all 50 outer models refitted','n':len(pred),'positive':int(pred.injury.sum()),'AP':average_precision_score(pred.injury,pred.score),'ROC_AUC':roc_auc_score(pred.injury,pred.score),'bootstrap':m.cluster_bootstrap(pred),'original_candidate_counts':dict(collections.Counter(x['original_inner_candidate_count'] for x in records)),'changed_selection_folds':[x['fold'] for x in records if x['changed_selection']],'selected_counts':dict(collections.Counter(f"{x['selected']['window']}d_{x['selected']['model']}_control" for x in records)),'script_sha256':sha(Path(__file__)),'input_hashes':manifest['input_history_sha256']}
 (NEW/'RECONCILED_REPORT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False),flush=True)

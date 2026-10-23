"""Pilot-only metadata diagnostic; no confidence-sequence claims.

All model choices below are fixed before evaluation. Completed-table labels are
read only to reconstruct the requested randomized pilot and evaluate predictions.
They are never used to select a model, tune a parameter, or fit a predictor.
"""
from pathlib import Path
import json
import os
import hashlib
import platform
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'conservatism_research'


def atomic_text(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def fit_boosted_metadata_scores(pilot_indices, pilot_verified_labels, *, metadata_path=None):
    """Fit the prespecified score predictor using passed revealed labels only.

    Returned correction[i] equals all_scores[i] minus the original pilot row
    baseline for the frozen direction. No nonpilot verified label is accessed.
    The population NPZ is used only for known rows, catalogue, and CSV alignment.
    All CSV columns read here are nonlabel metadata available before auditing.
    """
    pilot_indices = np.asarray(pilot_indices,dtype=int)
    pilot_verified_labels = np.asarray(pilot_verified_labels,dtype=int)
    if pilot_indices.shape != pilot_verified_labels.shape:
        raise ValueError('One revealed verified label per pilot index is required.')
    pop = np.load(ROOT/'work'/'replay_population.npz',allow_pickle=False)
    row = pop['y_row_index']
    N_r = pop['N_r']
    R,L,N = len(N_r),len(pop['labels']),len(row)
    if (len(np.unique(pilot_indices)) != len(pilot_indices) or
        np.any(pilot_indices<0) or np.any(pilot_indices>=N) or
        np.any(pilot_verified_labels<0) or np.any(pilot_verified_labels>=L)):
        raise ValueError('Invalid pilot records or catalogue label indices.')
    A = np.zeros((R,L),dtype=int)
    np.add.at(A,(row[pilot_indices],pilot_verified_labels),1)
    u = A.sum(1)
    beta = np.full((R,L),.5/(L-1))
    beta[np.arange(R),pop['row_to_label_index']] = .5
    Qrow = (A+beta)/(u+1)[:,None]
    ph = (A+(N_r-u)[:,None]*Qrow).sum(0)/N
    direction = -np.log(ph)
    row_baseline = Qrow@direction
    allowed = ['station_name','object_class','treatment',
               'observation_datetime','megadetector_confidence']
    metadata_path = metadata_path or ROOT/'upload'/'labels_keyword_cleaned(4).csv'
    df = pd.read_csv(metadata_path,usecols=allowed)
    df = df.iloc[pop['csv_line_number']-2].reset_index(drop=True)
    stamp = pd.to_datetime(df.observation_datetime,utc=True)
    hour = stamp.dt.hour.to_numpy()+stamp.dt.minute.to_numpy()/60
    period = np.floor(hour/6).astype(int)
    cat = np.column_stack((row.astype(str),df.station_name.astype(str),
                          df.object_class.astype(str),df.treatment.astype(str),
                          period.astype(str)))
    numeric = np.column_stack((df.megadetector_confidence.to_numpy(float),
                               np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)))
    encoding = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    X = np.column_stack((encoding.fit_transform(cat),numeric))
    target = direction[pilot_verified_labels]-row_baseline[row[pilot_indices]]
    weights = N_r[row[pilot_indices]]/u[row[pilot_indices]]
    weights /= weights.mean()
    model = HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,
                max_depth=4,min_samples_leaf=15,l2_regularization=10.,
                categorical_features=[0,1,2,3,4],random_state=93746)
    with threadpool_limits(limits=2):
        model.fit(X[pilot_indices],target,sample_weight=weights)
        scores = np.clip(row_baseline[row]+model.predict(X),direction.min(),direction.max())
    return dict(all_scores=scores,row_baseline_score=row_baseline,
                correction=scores-row_baseline[row],direction=direction,
                pilot_counts=A,pilot_indices=pilot_indices.copy(),row_indices=row,
                source_csv_line_numbers=pop['csv_line_number'].copy(),
                predictor_name='metadata_boost_score',
                training_verified_labels=len(pilot_verified_labels))


def main():
    start = time.perf_counter()
    seed = 1729
    pop = np.load(ROOT / 'work' / 'replay_population.npz', allow_pickle=False)
    df = pd.read_csv(ROOT / 'upload' / 'labels_keyword_cleaned(4).csv')
    df = df.iloc[pop['csv_line_number']-2].reset_index(drop=True)
    row = pop['y_row_index']
    labels = pop['v_label_index']
    N_r = pop['N_r']
    R, L = len(N_r), len(pop['labels'])
    N = len(labels)
    pilot_r = np.minimum(N_r, np.maximum(5, np.ceil(np.sqrt(N_r)).astype(int)))
    rng_indices, rng_values = np.random.default_rng(seed), np.random.default_rng(seed)
    indices = []
    for r in range(R):
        idx = np.flatnonzero(row == r)
        perm = rng_indices.permutation(idx)
        values = rng_values.permutation(labels[idx])
        assert np.array_equal(labels[perm], values)
        indices.extend(perm[:pilot_r[r]].tolist())
    train = np.array(indices)
    mask = np.ones(N, dtype=bool)
    mask[train] = False
    test = np.flatnonzero(mask)
    assert len(train) == 748 and len(test) == 32655
    A = np.zeros((R,L), dtype=int)
    np.add.at(A, (row[train], labels[train]), 1)
    beta = np.full((R,L), .5/(L-1))
    beta[np.arange(R), pop['row_to_label_index']] = .5
    Qrow = (A+beta)/(pilot_r+1)[:,None]
    qbase = Qrow[row]
    M = N_r-pilot_r
    ph = (A + M[:,None]*Qrow).sum(0)/N
    d = -np.log(ph)
    # Only pilot labels enter the direction and the fitted targets.
    pilot_target = d[labels[train]]
    pilot_prediction = qbase[train] @ d
    weights = N_r[row[train]]/pilot_r[row[train]]
    weights /= weights.mean()
    timefield = pd.to_datetime(df['observation_datetime'], utc=True)
    hour = timefield.dt.hour.to_numpy() + timefield.dt.minute.to_numpy()/60
    conf = df['megadetector_confidence'].to_numpy(float)
    conf_bin = np.digitize(conf, [.5,.7,.85])
    period = np.floor(hour/6).astype(int)
    station = df['station_name'].astype(str).to_numpy()
    obj = df['object_class'].astype(str).to_numpy()
    treatment = df['treatment'].astype(str).to_numpy()
    cat = np.column_stack((row.astype(str),station,obj,treatment,period.astype(str)))
    num = np.column_stack((conf,np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)))
    # Unlabeled covariates are known for all records and may define encodings.
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X = np.column_stack((onehot.fit_transform(cat),num))
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    Xord = np.column_stack((ordinal.fit_transform(cat),num))

    def eb(keys, prior):
        counts, sizes = {}, {}
        for i in train:
            key = keys[i]
            if key not in counts:
                counts[key] = np.zeros(L)
                sizes[key] = 0
            counts[key][labels[i]] += 1
            sizes[key] += 1
        result = np.empty((N,L))
        for i,key in enumerate(keys):
            result[i] = (counts.get(key,np.zeros(L))+10*prior[i])/(sizes.get(key,0)+10)
        return result

    qstation = eb([(int(r),str(s)) for r,s in zip(row,station)],qbase)
    qconf = eb([(int(r),int(c)) for r,c in zip(row,conf_bin)],qbase)
    qstationconf = eb([(int(r),str(s),int(c)) for r,s,c in zip(row,station,conf_bin)],qstation)
    predictors = {
        'row_only':(qbase,qbase@d),
        'row_station_EB10':(qstation,qstation@d),
        'row_confidence_EB10':(qconf,qconf@d),
        'row_station_confidence_EB10':(qstationconf,qstationconf@d),
    }
    with threadpool_limits(limits=2):
        forest = RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=5,
                                        max_features=1.0,n_jobs=2,random_state=93745)
        forest.fit(X[train],labels[train],sample_weight=weights)
        qforest = np.zeros((N,L))
        qforest[:,forest.classes_.astype(int)] = forest.predict_proba(X)
        # Fixed half-mixture guarantees positive probabilities for every label.
        qforest = .5*qbase+.5*qforest
        predictors['metadata_forest_half_row'] = (qforest,qforest@d)
        ridge = Ridge(alpha=10.)
        ridge.fit(X[train],pilot_target-pilot_prediction,sample_weight=weights)
        score = np.clip(qbase@d+ridge.predict(X),d.min(),d.max())
        predictors['metadata_ridge_score']=(None,score)
        boost = HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,
                    max_depth=4,min_samples_leaf=15,l2_regularization=10.,
                    categorical_features=[0,1,2,3,4],random_state=93746)
        boost.fit(Xord[train],pilot_target-pilot_prediction,sample_weight=weights)
        score = np.clip(qbase@d+boost.predict(Xord),d.min(),d.max())
        predictors['metadata_boost_score']=(None,score)

    # Evaluation starts here. No predictions/model parameters may change below.
    true_score = d[labels[test]]
    means = Qrow@d
    sd = np.sqrt(np.maximum(Qrow@(d*d)-means*means,1e-12))
    pi = .5*M/M.sum()+.5*M*sd/np.sum(M*sd)
    pi[M==0]=0
    actual_row = row[test]
    def diagnostics(name,q,score):
        residual = true_score-score[test]
        e1=np.bincount(actual_row,weights=residual,minlength=R)
        e2=np.bincount(actual_row,weights=residual*residual,minlength=R)
        vr = np.sum(np.divide(M*e2,pi,out=np.zeros(R),where=pi>0))-e1.sum()**2
        ans = dict(model=name,mean_score_residual=float(residual.mean()),
                   residual_mse=float(np.mean(residual**2)),
                   residual_variance=float(np.var(residual)),
                   next_observation_variance_srs=float((M.sum()/N)**2*np.var(residual)),
                   next_observation_variance_priority=float(vr/N**2))
        if q is not None:
            qt = q[test]
            ans.update(log_loss=float(-np.log(qt[np.arange(len(test)),labels[test]]).mean()),
                       accuracy=float(np.mean(qt.argmax(1)==labels[test])),
                       multiclass_brier=float(np.mean(np.sum(qt*qt,axis=1)-2*qt[np.arange(len(test)),labels[test]]+1)))
        return ans
    rows=[diagnostics(name,q,score) for name,(q,score) in predictors.items()]
    for r in rows:
        r['variance_ratio_srs_to_row']=r['next_observation_variance_srs']/rows[0]['next_observation_variance_srs']
        r['variance_ratio_priority_to_row']=r['next_observation_variance_priority']/rows[0]['next_observation_variance_priority']
    # Retrospective ceilings: unattainable fits, clearly separate from pilot models.
    ceilings=[]
    for name,key in [('row',row[test].astype(str)),('row_station',np.char.add(np.char.add(row[test].astype(str),'|'),station[test]))]:
        _,inv=np.unique(key,return_inverse=True)
        cnt=np.bincount(inv)
        avg=np.bincount(inv,weights=true_score)/cnt
        var=float(np.var(true_score-avg[inv]))
        ceilings.append(dict(infeasible_oracle=name,residual_variance=var,
            ratio_to_pilot_row_residual_variance=var/rows[0]['residual_variance']))
    result=dict(seed=seed,pilot_audits=len(train),unrevealed_evaluation_records=len(test),
        N=N,R=R,L=L,model_choices_fixed_before_evaluation=True,
        fitting_labels='only exact seed1729 common 748-audit pilot',
        direction='negative log pooled composition estimated from the pilot with existing row pseudocounts',
        prohibited_features=['updated_label','updated_label_raw','original_label_raw','image_composite_id','object_bbox_filename','camera_id','sd_card_id'],
        features=['original_label','station_name','object_class','treatment','six_hour_period','megadetector_confidence','hour_sine','hour_cosine'],
        metrics=rows,retrospective_oracle_ceilings=ceilings,
        metadata=dict(stations=int(df.station_name.nunique()),detector_confidence_min=float(conf.min()),detector_confidence_max=float(conf.max()),
                      station_counts={str(k):int(v) for k,v in df.station_name.value_counts().items()},object_class_counts={str(k):int(v) for k,v in df.object_class.value_counts().items()}),
        pilot_csv_line_numbers=[int(v) for v in pop['csv_line_number'][train]],
        limitations=['One development pilot, no model selected using the evaluation labels.',
                     'Residual-variance reduction is not confidence-sequence-width or audit-count reduction.',
                     'No new human labels used. Catalogue and completed labels retain existing replay limitations.',
                     'Record-specific predictions require generalizing the control-variate identity to known predictions for every remaining record.',
                     'Cluster and duplicate records remain part of the fixed row population; no identifiers are model features.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),seconds=time.perf_counter()-start)
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_text(OUT/'metadata_prediction_1729.json',json.dumps(result,indent=2)+'\n')
    atomic_text(OUT/'metadata_prediction_1729.csv',pd.DataFrame(rows).to_csv(index=False))
    np.savez_compressed(OUT/'metadata_predictions_1729.npz',
                       pilot_indices=train,test_indices=test,direction=d,
                       **{name:score for name,(_,score) in predictors.items()})
    print(json.dumps(dict(metrics=rows,oracle_ceilings=ceilings,seconds=result['seconds']),indent=2),flush=True)


if __name__=='__main__':
    main()

"""Transparent hypotheses for design follow-up. Suggestions are not validated improvements."""
def advise(diagnostic):
    suggestions=[]
    if diagnostic.get('trajectory_accuracy',0)>diagnostic.get('count_accuracy',0)+.05:
        suggestions.append({'bottleneck':'time_summary','candidate':'weighted/streaming readout',
            'test':'freeze encoder; equal readout-state budget; hold out delays',
            'status':'hypothesis, not an established recommendation'})
    if diagnostic.get('post_probe_accuracy',0)>diagnostic.get('output_probe_accuracy',0)+.05:
        suggestions.append({'bottleneck':'observation','candidate':'legal output interface or auxiliary state/readout',
            'test':'separate privileged membrane diagnostic from deployable observations',
            'status':'hypothesis, not an established recommendation'})
    if diagnostic.get('refit_accuracy',0)>diagnostic.get('original_accuracy',0)+.05:
        suggestions.append({'bottleneck':'readout_learning','candidate':'head retraining/loss change',
            'test':'same features and data; capacity/sample learning curves',
            'status':'hypothesis, not an established recommendation'})
    if diagnostic.get('history_mse_increase',0)>.05:
        suggestions.append({'bottleneck':'stale_history_dependency','candidate':'timescale or selective history-path change',
            'test':'R/C countervailing costs; matched gain/rate; heldout change age',
            'status':'hypothesis, not an established recommendation'})
    return {'suggestions':suggestions,'thresholds':'provisional operational values; lock on train/val before formal E05',
            'proven_predictive_utility':False}

import argparse
import json
from pathlib import Path
import numpy as np
from functional_anova import decompose, power_summary, error_risk_budget, conditional_mean_lattice, conditional_risk
from calibration_cases import (product_pairs, function_lattice, additive, ambiguous_product,
    referenced_product, harmful_prediction, focal_target, binding_target, cubic, known_conditional_panels,
    static_threshold, independent_write_reference)
from decoding import fit_probe, predict, mse, probe_metadata, fit_mlp


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serial(v) for v in x]
    return x


def controls(with_mlp=False):
    names = ('F','G','U','E'); population, a, b = product_pairs()
    functions = {'additive': additive, 'ambiguous_product': ambiguous_product,
                 'referenced_product': referenced_product, 'third_order': cubic,
                 'static_threshold_no_memory': static_threshold,
                 'nonlinear_per_write_encoder_linear_state': independent_write_reference}
    result = {'kind': 'known finite product populations and controlled decoder learning; not trained SNN results',
              'enumeration_pair_count': len(a), 'group_names': names, 'cases': {}}
    for name, fn in functions.items():
        fit = decompose(function_lattice(fn, a, b), names, population_enumeration=True)
        result['cases'][name] = power_summary(fit, ('F',))
    prediction = function_lattice(harmful_prediction, a, b)
    target = function_lattice(focal_target, a, b)
    _, risk = error_risk_budget(prediction-target, names, ('F',), population_enumeration=True)
    result['harmful_fixed_head'] = risk
    _, binding = error_risk_budget(function_lattice(ambiguous_product, a, b)-function_lattice(binding_target, a, b), names, ('F','G'), population_enumeration=True)
    result['same_interaction_as_legitimate_binding_target'] = binding
    _, wrong_task = error_risk_budget(function_lattice(ambiguous_product, a, b)-target, names, ('F',), population_enumeration=True)
    result['same_fixed_product_prediction_for_single_target'] = wrong_task
    _, constant = error_risk_budget(-target, names, ('F',), population_enumeration=True)
    result['constant_output_not_success'] = constant
    probability = function_lattice(lambda x:(1+.6*x[:,0]+.2*x[:,1]+.1*x[:,0]*x[:,1])/2,a,b)
    binary_label = function_lattice(lambda x:(1+x[:,0])/2,a,b)
    _,brier = error_risk_budget(probability-binary_label,names,('F',),population_enumeration=True)
    brier['classification_accuracy'] = float(np.mean((probability[-1]>=.5)==binary_label[-1]))
    result['probability_brier_control'] = brier
    reconstruction_error = function_lattice(lambda x:np.column_stack((.2*x[:,2],.3*x[:,0]*x[:,3])),a,b)
    _,reconstruction = error_risk_budget(reconstruction_error,names,('F','G'),population_enumeration=True)
    result['two_coordinate_reconstruction_control'] = reconstruction
    nested = conditional_mean_lattice(known_conditional_panels(harmful_prediction), names)
    result['nested_conditional_functions'] = {'reconstruction_max_error': nested['reconstruction_max_error'],
        'f_g_component_max_error': float(np.max(np.abs(nested['component_functions_estimated'][3, :, 0]-.7*population[:, 0]*population[:, 1]))),
        'note': 'All completions exhaustively enumerated; no iid finite-B correction applied to this population enumeration.'}
    rng = np.random.default_rng(6333)
    # Independent Rademacher scenes; a reference coordinate turns an ambiguous
    # product into a legally decodable target without providing latent labels.
    datasets = []
    for n in (512, 256, 4096):
        z = rng.choice((-1.,1.),size=(n,4))
        datasets.append((referenced_product(z), z[:, :1]))
    decoding = {}
    for family in ('linear','quadratic'):
        fit = fit_probe(*datasets[0], *datasets[1], family=family)
        decoding[family] = {'mse': mse(predict(fit, datasets[2][0]), datasets[2][1]), 'fit': probe_metadata(fit)}
    if with_mlp:
        decoding['mlp_seeds'] = []
        for seed in (1,2):
            fit = fit_mlp(*datasets[0], *datasets[1], seed=seed, width=32, epochs=200)
            decoding['mlp_seeds'].append({'mse': mse(predict(fit, datasets[2][0]), datasets[2][1]), 'fit': probe_metadata(fit)})
    result['legal_reference_decoding'] = decoding
    result['ambiguous_product_population_best_mse_for_F'] = 1.
    result['referenced_product_known_decoder_mse'] = 0.
    result['third_order_main_and_pair_power'] = 0.
    # Pooled encoding cancellation can arise solely from observed or hidden
    # context. The extra context is a distinct permission, not a free decoder.
    z = np.array(list(__import__('itertools').product((-1.,1.),repeat=2)))
    f, c = z[:,0], z[:,1]; r = c*f
    result['context_sign_control'] = {'pooled_cov_r_F': float(np.mean(r*f)),
        'per_context_known_decoder_mse': float(np.mean((c*r-f)**2)),
        'hidden_context_population_best_mse': 1.,
        'observed_context_quadratic_decoder_mse': float(np.mean((r*c-f)**2)),
        'extra_context_permission_must_be_declared': True}
    # Independent repeated completions validate the corrected conditional-risk
    # identity, separately from the exhaustive-population demonstration above.
    f = rng.normal(size=(2048, 1, 1)); g = rng.normal(size=(2048, 8, 1))
    e = rng.normal(size=(2048, 8, 1))
    p = f+.5*g+.7*f*g+.2*e
    cr = conditional_risk(p, f[:,0])
    result['iid_repeat_risk'] = {k:v for k,v in cr.items() if not k.startswith('per_scene')}
    result['iid_repeat_risk']['population_conditional_variability'] = .78
    return result


def figure(result, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11.8,8.1),layout='constrained')
    ax = axes[0,0]; ax.bar(['F main','F total'],[0,1],color=['#748ba6','#d77941']); ax.set_ylim(0,1.25)
    ax.set_title('Dependence can hide in an interaction'); ax.set_ylabel('Population variance')
    ax.text(.5,1.12,'r = F G, independent signs',ha='center')
    ax = axes[0,1]; vals=[1,0]
    ax.bar(['Only F G','Observe (F G, G)'],vals,color=['#d77941','#3a8f80']); ax.set_ylim(0,1.25)
    ax.set_title('A reference makes the same product recoverable'); ax.set_ylabel('Best population MSE for F')
    ax.text(.5,1.12,'Known decoder: F = r1 r2',ha='center')
    ax = axes[1,0]
    labels=['G','F x G','E']; vals=[.25,.49,.04]
    ax.bar(labels, vals,color=['#748ba6','#d77941','#3a8f80']); ax.set_ylim(0,.65)
    ax.set_title('Exclusive error effects count each interaction once'); ax.set_ylabel('MSE contribution')
    ax.text(1,.57,'g = F + 0.5G + 0.7FG + 0.2E; y = F',ha='center',fontsize=10)
    ax = axes[1,1]; ax.bar(['Target F','Target F G'],[2,0],color=['#d77941','#3a8f80']); ax.set_ylim(0,2.5)
    ax.set_title('Task dependency changes the interpretation'); ax.set_ylabel('MSE of the same prediction g = F G')
    ax.text(.5,2.22,'Interaction strength alone does not define harm',ha='center',fontsize=10)
    fig.suptitle('L3.3.3 known-answer controls | independent finite product populations | not SNN findings',fontsize=12)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); parser.add_argument('--figure'); parser.add_argument('--with-mlp',action='store_true'); args=parser.parse_args()
    if args.with_mlp:
        import torch
        torch.set_num_threads(1)
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    result=controls(args.with_mlp); output.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2))
    if args.figure: figure(result,args.figure)
    print(json.dumps({'output':str(output),'fixed_head_known_risk':result['harmful_fixed_head']['direct_mse'], 'error_budget_closure':result['harmful_fixed_head']['mse_identity_error'], 'probe_mse':{k:v['mse'] for k,v in result['legal_reference_decoding'].items() if isinstance(v,dict)}}))

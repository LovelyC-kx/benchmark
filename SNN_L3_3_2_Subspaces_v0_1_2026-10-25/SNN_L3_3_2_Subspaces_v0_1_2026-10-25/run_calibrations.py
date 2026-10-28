"""Known-answer geometry examples, separate from trained-SNN evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from bridge import gaussian_readout_reference, fit_joint
from subspace_geometry import basis, overlap, select_role_rank, null_overlaps
from readout_geometry import noise_metric_geometry, mapping_diagnostics


def calibrate():
    af = np.array([[1.], [0.]])
    au = np.array([[1.], [1.]])
    perfect = gaussian_readout_reference(af, au, np.zeros((2, 2)))
    projected = gaussian_readout_reference(np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1)))
    axes = overlap(basis(af), basis(au))
    mapping = mapping_diagnostics(perfect['w'], af, au, np.empty((2, 0)), au @ au.T)
    orthogonal_other = np.array([[0.], [1.]])
    noise = .04*np.eye(2); t = np.array([[1., 2.], [0., 1.]])
    before = gaussian_readout_reference(af, orthogonal_other, noise)
    after = gaussian_readout_reference(t @ af, t @ orthogonal_other, t @ noise @ t.T)
    metric_before = noise_metric_geometry(af, orthogonal_other, noise)
    metric_after = noise_metric_geometry(t @ af, t @ orthogonal_other, t @ noise @ t.T)
    near_parallel = []
    for angle in (0., 1., 3., 5., 10., 20., 45., 90.):
        a = np.array([[np.cos(np.deg2rad(angle))], [np.sin(np.deg2rad(angle))]])
        clean = gaussian_readout_reference(af, a, np.zeros((2, 2)))
        noisy = gaussian_readout_reference(af, a, .01*np.eye(2))
        near_parallel.append({'angle_degrees': angle,
            'noiseless_mmse': clean['mse_per_focal_coordinate'],
            'noise_std_0_1_mmse': noisy['mse_per_focal_coordinate'],
            'noiseless_decoder_norm': float(np.linalg.norm(clean['w']))})
    z = np.tile([[-1.], [1.]], (30, 1))
    fit = fit_joint(z, z, ['focal'])
    rank = select_role_rank(fit, 'focal', np.zeros_like(z), z)
    null = null_overlaps(np.eye(32)[:, :4], 12, np.eye(32), draws=1200, seed=777)
    return {'kind': 'known_answer_geometry_calibrations', 'new_snn_training_runs': 0,
        'nonorthogonal_but_exact': {'encoding_angle_degrees': axes['principal_angles_degrees'][0],
            'decoder': perfect['w'].tolist(), 'full_observation_mmse': perfect['mse_per_focal_coordinate'],
            'focal_encoding_only_mmse': projected['mse_per_focal_coordinate'],
            'constant_zero_mmse': 1., 'mapping_focal_gain_error': mapping['focal_gain_mixing_error_unit_latents'],
            'mapping_other_injection': mapping['cotarget_linear_injection_unit_latents'],
            'readout_encoding_subspace_overlap': mapping['readout_rowspace_vs_focal_encoding']['overlap_a_into_b'],
            'meaning': 'A direction with no marginal target encoding can supply a noise reference required for exact decoding.'},
        'same_subspace_wrong_readout': {'encoding': 1., 'readout': -1., 'subspace_overlap': 1., 'unit_target_mse': 4.},
        'invertible_mixing': {'raw_angle_before': 90.,
            'raw_angle_after': overlap(basis(t @ af), basis(t @ orthogonal_other))['principal_angles_degrees'][0],
            'noise_metric_angle_before': metric_before['metric_overlap']['principal_angles_degrees'][0],
            'noise_metric_angle_after': metric_after['metric_overlap']['principal_angles_degrees'][0],
            'mmse_before': before['mse_per_focal_coordinate'], 'mmse_after': after['mse_per_focal_coordinate'],
            'warning': 'Exact population covariance, transformed with the signal and reader. Isotropic shrinkage need not preserve this invariance.'},
        'near_parallel': near_parallel,
        'nonlinear_focal_signal': {'representation': '[F + e1, F^2 - 1 + e2]',
            'conditional_covariance': [[.04, 0.], [0., .04]],
            'linear_focal_remainder_covariance': [[.04, 0.], [0., 2.04]],
            'meaning': 'Average conditional variability and linear focal remainder have different contents.'},
        'validated_rank_zero': rank,
        'dimension_null': {'d': 32, 'rank_f': 4, 'rank_other': 12,
            'expected_focal_into_other': 12/32, 'expected_other_into_focal': 4/32,
            'simulation': null}}


def plot_controls(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(10.8, 7.5), layout='constrained')
    panel = ax[0, 0]
    vectors = [(1., 0., '#2074a9', 'F encoding'), (1., 1., '#d87a24', 'U encoding'),
               (1., -1., '#52616b', 'Readout [1,-1]')]
    for x, y, color, label in vectors:
        panel.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1,
                     width=.009, color=color)
        panel.text(x+.04, y, label, color=color, va='center', fontsize=9)
    panel.axhline(0, color='.8', lw=.7); panel.axvline(0, color='.8', lw=.7)
    panel.set(xlim=(-.3, 2.5), ylim=(-1.35, 1.35), xlabel='$r_1$', ylabel='$r_2$',
              title='A. Coding patterns and readout can differ')
    panel.set_aspect('equal', adjustable='box')
    panel = ax[0, 1]
    values = [0., .5, 1.]
    panel.bar(['Full [F+U, U]', 'F encoding axis', 'Constant zero'], values,
              color=['#2074a9', '#d87a24', '#b9c4ca'], width=.6)
    for i, v in enumerate(values): panel.text(i, v+.035, f'{v:g}', ha='center')
    panel.set(ylim=(0, 1.18), ylabel='Population optimal MSE',
              title='B. Encoding-only projection can lose a noise reference')
    panel.tick_params(axis='x', labelsize=8)
    panel = ax[1, 0]
    mix = result['invertible_mixing']; x = np.arange(2)
    panel.bar(x-.17, [mix['raw_angle_before'], mix['raw_angle_after']], .32, label='Euclidean', color='#d87a24')
    panel.bar(x+.17, [mix['noise_metric_angle_before'], mix['noise_metric_angle_after']], .32,
              label='Matched noise metric', color='#2074a9')
    panel.set(xticks=x, xticklabels=['Original', 'Invertible shear'], ylim=(0, 110), ylabel='Angle (degrees)',
              title='C. Coordinate change with matched covariance')
    panel.legend(frameon=False, fontsize=8)
    panel = ax[1, 1]
    rows = result['near_parallel'][1:]
    panel.semilogx([r['angle_degrees'] for r in rows], [r['noiseless_mmse'] for r in rows],
                   'o-', label='No observation noise', color='#2074a9')
    panel.semilogx([r['angle_degrees'] for r in rows], [r['noise_std_0_1_mmse'] for r in rows],
                   'o-', label='Noise std = 0.1', color='#d87a24')
    panel.set(ylim=(-.015, .53), xlabel='Encoding angle (degrees, log scale)', ylabel='Population optimal MSE',
              title='D. Near-parallel encoding can be fragile')
    panel.legend(frameon=False, fontsize=8)
    fig.suptitle('Known-answer controls for L3.3.2 — not trained-SNN results', fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.figure and args.figure.exists()): raise FileExistsError('Choose new outputs')
    result = calibrate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if args.figure: plot_controls(result, args.figure)
    print(str(args.output.resolve()))

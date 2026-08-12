"""
Parallel Monte Carlo launcher: splits --total-trials across --num-workers
separate OS PROCESSES (not threads -- this workload is CPU-bound IPOPT/RRT*
work, and Python threads can't parallelize that due to the GIL), each
running diagnostic_1A_homotopy_check.py end to end with the full validated
pipeline (bugfixes + damping + warm-start + windowed convergence + 1-B
multi-homotopy). All workers share the same environment (--rng-seed, so the
same map/sensor layout) but get distinct, reproducible per-trial randomness
via --seed-offset. Every worker saves its own JSON as it goes; when all
workers finish, this script merges them into one combined JSON.

Usage:
  python run_monte_carlo_parallel.py --total-trials 500 --num-workers 5

To resume/just merge already-finished worker output without re-running:
  python run_monte_carlo_parallel.py --merge-only --out-dir <dir>
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, 'diagnostic_1A_homotopy_check.py')


def launch_workers(args, out_dir):
    base = args.total_trials // args.num_workers
    remainder = args.total_trials % args.num_workers
    procs = []
    for w in range(args.num_workers):
        n_trials = base + (1 if w < remainder else 0)
        if n_trials == 0:
            continue
        prefix = f'worker{w}_'
        log_path = os.path.join(out_dir, f'{prefix}log.txt')
        cmd = [
            sys.executable, WORKER_SCRIPT,
            '--num-trials', str(n_trials),
            '--k-alt', str(args.k_alt),
            '--n-attk', str(args.n_attk),
            '--max-iters', str(args.max_iters),
            '--hard-break-iter', str(args.hard_break_iter),
            '--eta', str(args.eta),
            '--warm-start-attacker',
            '--cycle-window', str(args.cycle_window),
            '--tol-cost-windowed', str(args.tol_cost_windowed),
            '--tol-sensor-windowed', str(args.tol_sensor_windowed),
            '--rng-seed', str(args.rng_seed),
            '--seed-offset', str(w * 10000 + args.seed_offset_base),
            '--out-dir', out_dir,
            '--out-prefix', prefix,
        ]
        print(f'[launcher] worker {w}: {n_trials} trials, seed_offset={w*10000+args.seed_offset_base}, log={log_path}')
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((w, proc, log_f))

    print(f'[launcher] {len(procs)} workers running in parallel (PIDs: {[p.pid for _, p, _ in procs]})')
    t0 = time.time()
    exit_codes = {}
    for w, proc, log_f in procs:
        rc = proc.wait()
        log_f.close()
        exit_codes[w] = rc
        elapsed = time.time() - t0
        status = 'OK' if rc == 0 else f'FAILED (exit {rc})'
        print(f'[launcher] worker {w} finished: {status}  (total elapsed so far: {elapsed:.0f}s)')

    n_failed = sum(1 for rc in exit_codes.values() if rc != 0)
    if n_failed:
        print(f'[launcher] WARNING: {n_failed}/{len(procs)} workers failed -- check their log.txt files before trusting the merge')
    return exit_codes


def merge_results(out_dir):
    worker_finals = sorted(glob(os.path.join(out_dir, 'worker*_[0-9]*_1A_homotopy_diagnostic.json')))
    if not worker_finals:
        print(f'[merge] no worker*_..._1A_homotopy_diagnostic.json files found in {out_dir}')
        return None

    all_results = []
    per_worker_args = []
    for path in worker_finals:
        with open(path) as f:
            d = json.load(f)
        all_results.extend(d['results'])
        per_worker_args.append({'file': os.path.basename(path), 'args': d['args']})
        print(f'[merge] loaded {len(d["results"])} trials from {os.path.basename(path)}')

    # renumber trial indices to be unique across the merged set
    for i, r in enumerate(all_results):
        r['global_trial'] = i

    n_converged = sum(1 for r in all_results if r.get('converged'))
    n_converged_windowed = sum(1 for r in all_results if r.get('converged_windowed'))
    n_beat = sum(1 for r in all_results if r.get('beat'))
    margins = [r['margin_pct'] for r in all_results if r.get('beat')]

    summary = {
        'num_trials': len(all_results),
        'num_workers': len(worker_finals),
        'converged_strict_rate_pct': 100.0 * n_converged / len(all_results) if all_results else 0.0,
        'converged_windowed_rate_pct': 100.0 * n_converged_windowed / len(all_results) if all_results else 0.0,
        'beat_rate_pct': 100.0 * n_beat / len(all_results) if all_results else 0.0,
        'mean_margin_pct_when_beat': sum(margins) / len(margins) if margins else 0.0,
    }
    print('\n=== MERGED SUMMARY ===')
    print(json.dumps(summary, indent=2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f'{timestamp}_merged_monte_carlo.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'per_worker_args': per_worker_args, 'results': all_results}, f, indent=2)
    print(f'\n[merge] saved combined results ({len(all_results)} trials) to: {out_path}')
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total-trials', type=int, default=500)
    ap.add_argument('--num-workers', type=int, default=5)
    ap.add_argument('--k-alt', type=int, default=3)
    ap.add_argument('--n-attk', type=int, default=100)
    ap.add_argument('--max-iters', type=int, default=50)
    ap.add_argument('--hard-break-iter', type=int, default=30)
    ap.add_argument('--eta', type=float, default=0.3)
    ap.add_argument('--cycle-window', type=int, default=4)
    ap.add_argument('--tol-cost-windowed', type=float, default=1e-3)
    ap.add_argument('--tol-sensor-windowed', type=float, default=0.1)
    ap.add_argument('--rng-seed', type=int, default=2, help='shared across all workers -> same environment/map')
    ap.add_argument('--seed-offset-base', type=int, default=0)
    ap.add_argument('--out-dir', type=str, default='.')
    ap.add_argument('--merge-only', action='store_true', help='skip launching workers; just merge existing worker*_..._1A_homotopy_diagnostic.json files in --out-dir')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.merge_only:
        t0 = time.time()
        launch_workers(args, args.out_dir)
        print(f'[launcher] all workers done in {time.time()-t0:.0f}s')

    merge_results(args.out_dir)


if __name__ == '__main__':
    main()

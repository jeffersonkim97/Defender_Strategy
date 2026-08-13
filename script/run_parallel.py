"""
Generic parallel launcher: splits --total-trials across --num-workers
separate OS PROCESSES running ANY of our per-trial scripts (baseline_2A_2B_
defender.py, baseline_2C_static_fov.py, diagnostic_1A_homotopy_check.py, ...)
as long as they support --num-trials/--rng-seed/--seed-offset/--out-dir/
--out-prefix. Merges every worker's saved JSON ('results' list) into one
combined file when all workers finish (or on demand via --merge-only),
falling back to a worker's latest _partial.json if it was killed mid-run.

Usage:
  python run_parallel.py --worker-script baseline_2A_2B_defender.py \
      --total-trials 500 --num-workers 4 --out-dir baseline_2A2B_500 \
      --result-tag 2A2B_baseline --extra-args "--k-candidates 10 --num-sweeps 3"

  python run_parallel.py --worker-script baseline_2C_static_fov.py \
      --total-trials 500 --num-workers 4 --out-dir baseline_2C_500 \
      --result-tag 2C_static_fov
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def launch_workers(args, out_dir):
    worker_script = os.path.join(SCRIPT_DIR, args.worker_script)
    extra = shlex.split(args.extra_args) if args.extra_args else []

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
            sys.executable, worker_script,
            '--num-trials', str(n_trials),
            '--rng-seed', str(args.rng_seed),
            '--seed-offset', str(w * 10000 + args.seed_offset_base),
            '--out-dir', out_dir,
            '--out-prefix', prefix,
        ] + extra
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


def auto_summary(all_results):
    """Generic: mean/std of every numeric field present across all result dicts."""
    if not all_results:
        return {}
    numeric_keys = [k for k, v in all_results[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    summary = {}
    for k in numeric_keys:
        vals = [r[k] for r in all_results if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            summary[f'mean_{k}'] = mean
            summary[f'std_{k}'] = var ** 0.5
    return summary


def merge_results(out_dir, result_tag):
    all_files = glob(os.path.join(out_dir, f'worker*_*_{result_tag}*.json'))
    by_worker = {}
    for path in all_files:
        base = os.path.basename(path)
        m = re.match(r'worker(\d+)_', base)
        if not m:
            continue
        w = int(m.group(1))
        bucket = 'partial' if base.endswith('_partial.json') else 'final'
        by_worker.setdefault(w, {'final': [], 'partial': []})[bucket].append(path)

    if not by_worker:
        print(f'[merge] no worker*_..._{result_tag}*.json files found in {out_dir}')
        return None

    all_results = []
    per_worker_args = []
    for w in sorted(by_worker):
        finals = sorted(by_worker[w]['final'])
        partials = sorted(by_worker[w]['partial'])
        if finals:
            path, source = finals[-1], 'final'
        elif partials:
            path, source = partials[-1], 'partial (worker did not finish)'
        else:
            continue
        with open(path) as f:
            d = json.load(f)
        all_results.extend(d['results'])
        per_worker_args.append({'file': os.path.basename(path), 'source': source, 'args': d['args']})
        print(f'[merge] worker {w}: loaded {len(d["results"])} trials from {os.path.basename(path)} ({source})')

    for i, r in enumerate(all_results):
        r['global_trial'] = i

    summary = {'num_trials': len(all_results), 'num_workers': len(per_worker_args)}
    summary.update(auto_summary(all_results))
    print('\n=== MERGED SUMMARY ===')
    print(json.dumps(summary, indent=2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f'{timestamp}_merged_{result_tag}.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'per_worker_args': per_worker_args, 'results': all_results}, f, indent=2)
    print(f'\n[merge] saved combined results ({len(all_results)} trials) to: {out_path}')
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker-script', type=str, required=True, help='filename (in this dir) of the per-trial script to run, e.g. baseline_2A_2B_defender.py')
    ap.add_argument('--result-tag', type=str, required=True, help='the tag in the worker script\'s output filenames, e.g. "2A2B_baseline" or "2C_static_fov" (or "1A_homotopy_diagnostic")')
    ap.add_argument('--extra-args', type=str, default='', help='extra CLI args passed through verbatim to each worker, e.g. "--k-candidates 10 --num-sweeps 3"')
    ap.add_argument('--total-trials', type=int, default=500)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--rng-seed', type=int, default=2, help='shared across all workers -> same environment/map')
    ap.add_argument('--seed-offset-base', type=int, default=0)
    ap.add_argument('--out-dir', type=str, default='.')
    ap.add_argument('--merge-only', action='store_true', help='skip launching workers; just merge existing worker output in --out-dir')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.merge_only:
        t0 = time.time()
        launch_workers(args, args.out_dir)
        print(f'[launcher] all workers done in {time.time()-t0:.0f}s')

    merge_results(args.out_dir, args.result_tag)


if __name__ == '__main__':
    main()

"""Recompute the reported summaries and LaTeX tables from saved paths."""
from pathlib import Path
import argparse
import shutil
from atlas_auditing.reporting import report, collect_runs
from atlas_auditing.tables import write_tables

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--runs',type=Path,help='Collect a complete run_experiments.py run first.')
    args=parser.parse_args()
    if args.runs:
        collect_runs(args.runs,args.output)
        # These are fixed population summaries, not simulated outcomes.
        for name in ('confusion','synthetic/population.json'):
            src,dst=args.results/name,args.output/name
            if src.resolve()!=dst.resolve():
                if src.is_dir(): shutil.copytree(src,dst,dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        args.results=args.output
    targets=report(args.results,args.output)
    write_tables(targets,args.output)
    print(f'Wrote summaries and tables to {args.output}')


if __name__=='__main__':
    main()

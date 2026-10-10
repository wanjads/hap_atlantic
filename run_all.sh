#!/bin/bash
# all runs of the paper: capacity sweep (seed 0), latency for 4 more cloud seeds, ring ablation,
# maps, figures and the summary of all numbers (about 1.5 h on 2 cores)
cd "$(dirname "$0")"
export PYTHONPATH=src
python3 scripts/quick.py sweep rings
python3 scripts/quick.py --seed 1 --seed 2 --seed 3 --seed 4 latency
python3 scripts/quick.py map newyork frankfurt
python3 scripts/quick.py --plot sweep
python3 scripts/quick.py summary
echo DONE $(date)

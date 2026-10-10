#!/bin/bash
# main runs of the paper (sweep for all seeds and both days, then ring ablation, maps)
cd "$(dirname "$0")"
export PYTHONPATH=src
python3 scripts/quick.py sweep rings
python3 scripts/quick.py map newyork frankfurt
python3 scripts/quick.py --plot sweep
python3 scripts/quick.py summary
echo DONE $(date)

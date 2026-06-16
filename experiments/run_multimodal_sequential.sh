#!/bin/bash

cd /home/maialen/skin_lesion_PFG
source /home/maialen/pfg-venv/bin/activate

echo "Starting multimodal sequential experiments..."
echo "Started at: $(date)"

for config in multimodal_e01_sex multimodal_e02_age multimodal_e03_loc multimodal_e04_sex_age multimodal_e05_sex_loc multimodal_e06_age_loc multimodal_e07_all; do
    echo ""
    echo "=============================="
    echo "Launching: $config"
    echo "Time: $(date)"
    echo "=============================="
    python experiments/run.py --config experiments/configs/${config}.yaml > outputs/logs/${config}.log 2>&1
    echo "Finished: $config at $(date)"
done

echo ""
echo "All multimodal experiments completed at: $(date)"

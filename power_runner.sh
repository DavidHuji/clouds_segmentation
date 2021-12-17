#!/bin/bash
#PBS -q gpu2 -l ngpus=2
#PBS -lmem=10gb,pmem=10gb,vmem=10gb,pvmem=10gb
#PBS -v PBS_O_WORKDIR=/msegalrolab/msegalro/ACMAP
module load  miniconda/miniconda3-4.7.12-environmentally
conda activate "/powerapps/share/centos7/miniconda/miniconda3-4.7.12-environmentally/envs/pytorch"
cd /a/home/cc/stud_csguests/nukraidavid/bbb
python main.py

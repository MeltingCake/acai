#!/bin/bash
bash launch_slurm_job.sh "t4,p100" train_acai_fso_lake 1 "python acai.py --train_dir=/h/wangale/project/acai/fs_omniglot_lake --dataset=fs_omniglot --fso_config=\"fs_omniglot/lake/train\""
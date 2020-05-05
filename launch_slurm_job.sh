#!/bin/bash

d=`date +%Y-%m-%d`
partition=$1
j_name=$2
resource=$3
cmd=$4
hdd=/scratch/hdd001/home/$USER
ssd=/scratch/ssd001/home/$USER
j_dir=$ssd/slurm/$d/$j_name

mkdir -p $j_dir/scripts

# build slurm script
mkdir -p $j_dir/log
if [ "${partition}" == "p100" ] || [ "${partition}" == "t4" ] || [ "${partition}" == "t4,p100" ]; then
  echo "#!/bin/bash
#SBATCH --job-name=${j_name}
#SBATCH --output=${j_dir}/log/%j.out
#SBATCH --error=${j_dir}/log/%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=$[8 * $resource]
#SBATCH --ntasks-per-node=1
#SBATCH --mem=$[24*$resource]G
#SBATCH --gres=gpu:${resource}
#SBATCH --nodes=1

bash ${j_dir}/scripts/${j_name}.sh
  " > $j_dir/scripts/${j_name}.slrm
elif [ "${partition}" == "cpu" ]; then
  echo "#!/bin/bash
#SBATCH --job-name=${j_name}
#SBATCH --output=${j_dir}/log/%j.out
#SBATCH --error=${j_dir}/log/%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=$[8 * $resource]
#SBATCH --ntasks-per-node=1
#SBATCH --mem=$[38*$resource]G
#SBATCH --nodes=1

bash ${j_dir}/scripts/${j_name}.sh
  " > $j_dir/scripts/${j_name}.slrm
else
  echo "#!/bin/bash
#SBATCH --job-name=${j_name}
#SBATCH --output=${j_dir}/log/%j.out
#SBATCH --error=${j_dir}/log/%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=$[4 * $resource]
#SBATCH --ntasks-per-node=1
#SBATCH --mem=$[24*$resource]GB
#SBATCH --gres=gpu:${resource}
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --exclude=gpu070

bash ${j_dir}/scripts/${j_name}.sh
  " > $j_dir/scripts/${j_name}.slrm
fi

# build bash script
echo -n "#!/bin/bash
$cmd
" > $j_dir/scripts/${j_name}.sh

sbatch $j_dir/scripts/${j_name}.slrm
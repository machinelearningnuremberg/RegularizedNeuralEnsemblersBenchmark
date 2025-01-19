#project_name=RQ1_qt_micro_nll
#n=30
project_name=$1
n=$2
experiment_id=$3
grid_size=$4
r=3
echo ${project_name} ${n} ${experiment_id} ${grid_size}
sbatch --exclude=dlcgpu04 --array=1-$(($grid_size*$n*$r))%30 cluster_scripts/slurm_pineda/run.sh ${project_name}/${experiment_id}.args

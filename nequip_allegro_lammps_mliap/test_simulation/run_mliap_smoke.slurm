#!/bin/bash -e
#SBATCH --job-name=lmp_mliap_smoke
#SBATCH --account=project_465002275
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=lmp_mliap_smoke_%j.out
#SBATCH --error=lmp_mliap_smoke_%j.err

echo "Started at $(date) on $(hostname)"

module --force purge
module load LUMI/24.03 partition/G

PROJECT_DIR=/scratch/project_465002275/sveinsso/test_allegro_installation_henrik
SIF=$PROJECT_DIR/from_greatfacet/local_build/stage1.sif
HOSTPREFIX=$PROJECT_DIR/from_greatfacet/local_build/hostprefix
WORKDIR=$PROJECT_DIR/test_simulation

echo "SIF: $SIF"
echo "HOSTPREFIX: $HOSTPREFIX"
echo "WORKDIR: $WORKDIR"

export OMP_NUM_THREADS=1

srun --ntasks=1 --gpus-per-task=1 \
  singularity exec --rocm \
  --bind $PROJECT_DIR:/work \
  --pwd /work/test_simulation \
  "$SIF" bash -lc '
    set -e
    export LD_LIBRARY_PATH=/opt/miniconda3/envs/pytorch/lib:/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib/llvm/lib:/work/from_greatfacet/local_build/hostprefix/lib64:/work/from_greatfacet/local_build/hostprefix/lib:$LD_LIBRARY_PATH
    /work/from_greatfacet/local_build/hostprefix/bin/lmp -in in.mliap_nequip_smoke -k on g 1 -sf kk -pk kokkos newton on neigh half
  '

echo "Finished at $(date)"
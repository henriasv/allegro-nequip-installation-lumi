Reproducible LAMMPS + ML-IAP (NequIP/Allegro) build on LUMI using Singularity

This bundle reproduces the working setup: LAMMPS (22 Jul 2025) with KOKKOS HIP for gfx90a, Python 3.12, and ML-IAP with NequIP/Allegro (torch >= 2.6 via LUMI PyTorch container).

Contents
- defs/stage1.def            # builds stage1.sif from LUMI PyTorch container and installs cython/nequip/allegro
- version.conf               # central versions
- build_hostprefix.sh        # compiles LAMMPS inside stage1.sif into ./hostprefix
- env.sh                     # helpful runtime env exports for manual runs
- test_simulation/
  - in.mliap_nequip_smoke    # minimal ML-IAP input using a NequIP model
  - run_mliap_smoke.slurm    # Slurm script to run the smoke test on 1 GPU

Prerequisites
- LUMI modules: LUMI/24.03, partition/G
- Off-LUMI build host with Singularity/Apptainer and root or fakeroot
- A NequIP/Allegro `.pt` model file (example: /scratch/.../output.nequip.lmp.pt)

Step 1: Load modules
```bash
module --force purge
module load LUMI/24.03 partition/G
```

Step 2: Build stage1.sif on another machine (required)
1) On LUMI, find the base PyTorch SIF path to copy:
```bash
module --force purge
module load LUMI/24.03 partition/G
module use /scratch/project_465002275/sveinsso/EasyBuild/modules/container
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
echo "$SIF"   # copy this file off LUMI as stage1-base.sif
```

2) On your other machine (with root or fakeroot):
```bash
cd nequip_allegro_lammps_mliap
ln -sf /path/to/stage1-base.sif stage1-base.sif
# optional: local cache/tmp
export SINGULARITY_CACHEDIR=$PWD/.singularity-cache
export SINGULARITY_TMPDIR=$PWD/.singularity-tmp
# prefer root/fakeroot to avoid proot issues
sudo singularity build stage1.sif defs/stage1.def
```

Optional: use /dev/shm (tmpfs) on that machine
```bash
export SINGULARITY_TMPDIR=/dev/shm/sing_tmp
export SINGULARITY_CACHEDIR=/dev/shm/sing_cache   # or keep cache on disk if downloads are large
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
sudo singularity build stage1.sif defs/stage1.def
```

3) Copy stage1.sif back to LUMI into this folder.

Step 3: Compile LAMMPS into ./hostprefix
```bash
# optional: limit parallelism to reduce memory usage during build
export CMAKE_BUILD_PARALLEL_LEVEL=8   # try 4 if memory is tight

# optional: move singularity tmp/cache to scratch to avoid /run OOM
mkdir /local/tmp/sveinsso/.singularity-tmp
mkdir /local/tmp/sveinsso/.singularity-cache
export SINGULARITY_CACHEDIR=/local/tmp/sveinsso/.singularity-cache
export SINGULARITY_TMPDIR=/local/tmp/sveinsso/.singularity-tmp

./build_hostprefix.sh  # creates ./hostprefix with bin/lmp and lib64/liblammps.so
```

Step 4: Run the GPU smoke test (single GPU)
Edit `test_simulation/run_mliap_smoke.slurm` if needed to point MODEL_PATH to your model (default expects /work/output.nequip.lmp.pt via bind).
```bash
cd test_simulation
sbatch run_mliap_smoke.slurm
```

Manual run inside the container (optional)
```bash
cd nequip_allegro_lammps_mliap
export LD_LIBRARY_PATH=$PWD/hostprefix/lib64:$PWD/hostprefix/lib:$LD_LIBRARY_PATH
singularity exec --rocm --bind $PWD:/work --pwd /work stage1.sif bash -lc \
  'export LD_LIBRARY_PATH=/opt/miniconda3/envs/pytorch/lib:/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib/llvm/lib:/work/hostprefix/lib64:/work/hostprefix/lib:$LD_LIBRARY_PATH; \
   /work/hostprefix/bin/lmp -h | head'
```

Troubleshooting
- If `lmp` complains about missing libs (e.g., libomp or libjpeg), ensure LD_LIBRARY_PATH includes:
  - /opt/miniconda3/envs/pytorch/lib:/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib/llvm/lib
  - ./hostprefix/lib64:./hostprefix/lib
- If ML-IAP errors that "newton pair on" is required, add `newton on` to your LAMMPS input.
- Warnings about libtinfo "no version information" are benign on LUMI and can be ignored.



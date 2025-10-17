#!/bin/bash

source container_builder.conf

if [[ -z ${SINGULARITY_CACHEDIR+x} ]]; then
 export SINGULARITY_CACHEDIR="${XDG_RUNTIME_DIR}/singularity/cache"
fi

if [[ -z ${SINGULARITY_TMPDIR+x} ]]; then
 export SINGULARITY_TMPDIR="${XDG_RUNTIME_DIR}/singularity/tmp"
fi

mkdir -p $SINGULARITY_CACHEDIR
mkdir -p $SINGULARITY_TMPDIR

module --force purge
module load CrayEnv
module load systools

if [ "$#" -eq 1 ]; then
  start_stage=$1
else
  start_stage=1
fi

for stage in $(seq ${start_stage} ${num_stages}); do
  build_cmd="singularity build" 
  
  if [ -f binds/stage${stage}.bind ]; then
    build_cmd="${build_cmd} -B$(s=$(cat binds/stage${stage}.bind | tr '\n' ','); echo ${s%,})"
  fi

  if [[ $stage -eq $num_stages ]]; then
    if [ -f ${final_container}.sif ]; then
      rm -f ${final_container}.sif
    fi

    build_cmd="${build_cmd} ${final_container}.sif"
  else
    if [[ $stage -gt 1 ]]; then
      build_cmd="${build_cmd} -B$(realpath version.conf):/version.conf"
    fi

    if [ -f stage${stage}.sif ]; then
      rm -f stage${stage}.sif
    fi

    build_cmd="${build_cmd} stage${stage}.sif"
  fi

  build_cmd="${build_cmd} defs/stage${stage}.def"  

  echo "Running build command: ${build_cmd}"

  ${build_cmd}
done

rm -rf $SINGULARITY_CACHEDIR/*
rm -rf $SINGULARITY_TMPDIR/*

echo "Final container: ${final_container}"

cat <<EOF

To install the container as a module, run the following commands:

  module purge
  module load LUMI/24.03 partition/container EasyBuild-user
  eb -r. -f $easyconfig


EOF

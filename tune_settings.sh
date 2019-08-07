#!/usr/bin/env bash
#set -e
#set -x

echo 'Running with parameters:'
echo "    MODEL_DIR: ${MODEL_DIR}"
echo "    DATA_DIR: ${DATA_DIR}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    OUTPUT_DIR: ${OUTPUT_DIR}"


# Check docker
if ! [[ $(which docker) && $(docker --version) ]]; then
    echo "Docker not found, please install docker to proceed."
    exit 1
fi

timestamp=`date +%Y%m%d_%H%M%S`
LOG_FILENAME="tune_${MODEL_NAME}_${timestamp}.log"
if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir ${OUTPUT_DIR}
fi

MKL_IMAGE_TAG=tensorflow/serving:latest-mkl

function docker_run(){
    docker run \
        --name=${CONTAINER_NAME} \
        --rm \
        -d \
        -p 8500:8500 \
        -v /tmp:/models/${MODEL_NAME} \
        -e MODEL_NAME=${MODEL_NAME} \
        -e OMP_NUM_THREADS=${omp} \
        -e TENSORFLOW_INTER_OP_PARALLELISM=${inter} \
        -e TENSORFLOW_INTRA_OP_PARALLELISM=${intra} \
        ${MKL_IMAGE_TAG}
}


function wide_deep(){
    RUNNING=$(docker ps --filter="expose=8501/tcp" -q | xargs)
    if [[ -n ${RUNNING} ]]; then
        docker rm -f ${RUNNING}
    fi

    CONTAINER_NAME=tfserving_${RANDOM}

    # Run container
    MKL_IMAGE_TAG=${MKL_IMAGE_TAG} CONTAINER_NAME=${CONTAINER_NAME} docker_run

    # Test
    python ../run_wide_deep.py --data_file ~/intel-models/models/eval_preprocessed.tfrecords --batch_size ${bs}

    # Clean up
    docker rm -f ${CONTAINER_NAME}
}

LOGFILE=${OUTPUT_DIR}/${LOG_FILENAME}

MODEL_NAME=$(echo ${MODEL_NAME} | tr 'A-Z' 'a-z')

batch_sizes=( 1 2 32 128 256 512 )
omp_num_threads=( 1 8 16 )
intra_threads=( 1 2 )
inter_threads=( 1 8 16 )

performance=()

for bs in "${batch_sizes[@]}"; do

for omp in "${omp_num_threads[@]}" ; do

for inter in "${inter_threads[@]}" ; do

for intra in "${intra_threads[@]}" ; do

if [ ${MODEL_NAME} == "wide_deep" ] || [ ${MODEL_NAME} == "resnet50" ] ; then
  wide_deep | tee -a ${LOGFILE}
else
  echo "Unsupported Model: ${MODEL_NAME}"
  exit 1
fi

performance+=( $(grep 'Throughput' ${LOGFILE} | tail -1 | sed 's/[^0-9.]*//g') )

done

done

done

done

echo "Log output location: ${LOGFILE}" | tee -a ${LOGFILE}

echo ${performance}

echo ${performance[3]}

echo "Batch Size | OMP_NUM_THREADS | INTER_OP | INTRA_OP | Performance"
i=0
for bs in "${batch_sizes[@]}"; do

for omp in "${omp_num_threads[@]}" ; do

for inter in "${inter_threads[@]}" ; do

for intra in "${intra_threads[@]}" ; do

printf "%-12s %-17s %-10s %-10s %s \n" ${bs} ${omp} ${inter} ${intra} ${performance[$i]}
i=$(( i + 1 ))

done

done

done

done


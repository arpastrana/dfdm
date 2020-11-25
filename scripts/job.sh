#!/bin/sh

# naming
EXPERIMENT="arch_state_25"  # arch_state_17
NETWORK_NAME="compression_network.json"
OPT_NETWORK_NAME="compression_network_opt.json"

# fdm + optimization controls
PZ="-0.01795"
Q0="-1.5"
BRICK_LENGTH="0.123"
OPT_METHOD="SLSQP"
MAX_ITERS="300"
TOL="1e-9"

# directories
LOCAL_FOLDER="../data/json/"
EXPERIMENT_FOLDER="${LOCAL_FOLDER}${EXPERIMENT}/"
EXPERIMENT_FILE="${LOCAL_FOLDER}${EXPERIMENT}.json"

# make dir if it does not exist
mkdir -p "$EXPERIMENT_FOLDER"

# loop
for SUPPORT in {0..53}  # 35 for arch_state_17, 53 for arch_state_24
do
    TEST_FOLDER="${EXPERIMENT_FOLDER}${SUPPORT}/"

    mkdir -p -- $TEST_FOLDER

    NETWORK_OUT="${TEST_FOLDER}${NETWORK_NAME}"
    NETWORK_OPT_OUT="${TEST_FOLDER}${OPT_NETWORK_NAME}"

    echo "=================="
    echo "Running experiment ${EXPERIMENT} with extra support at ${SUPPORT}"
    python 05_fdm_batch.py $EXPERIMENT_FILE $NETWORK_OUT $NETWORK_OPT_OUT \
        $SUPPORT $PZ $Q0 $BRICK_LENGTH $OPT_METHOD $MAX_ITERS $TOL
done

echo "*Optimization Sweep complete!!*"
echo "==============================="

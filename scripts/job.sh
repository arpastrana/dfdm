#!/bin/sh

# naming
EXPERIMENT="arch_state_25"  # arch_state_17
NETWORK_NAME="compression_network.json"
OPT_NETWORK_NAME="compression_network_opt.json"
NODES_CSV_NAME="nodes.csv"
EDGES_CSV_NAME="edges.csv"

# fdm + optimization controls
PZ="-0.01795"
Q0="-1.5"
BRICK_LENGTH="0.123"
OPT_METHOD="SLSQP"
MAX_ITERS="200"
TOL="1e-9"

# stats controls
NODE_SHIFT_BACK="2"
ROUND_PREC="3"

# directories
LOCAL_FOLDER="../data/json/"
EXPERIMENT_FOLDER="${LOCAL_FOLDER}${EXPERIMENT}"
EXPERIMENT_FILE="${LOCAL_FOLDER}${EXPERIMENT}.json"

mkdir -p "$EXPERIMENT_FOLDER"

for SUPPORT in {0..54}  # 35 for arch_state_17
do
    TEST_FOLDER="${EXPERIMENT_FOLDER}/${SUPPORT}/"

    mkdir -p -- $TEST_FOLDER

    NETWORK_OUT="${TEST_FOLDER}${NETWORK_NAME}"
    NETWORK_OPT_OUT="${TEST_FOLDER}${OPT_NETWORK_NAME}"

    echo "=================="
    echo "Running experiment ${EXPERIMENT} with extra support at ${SUPPORT}"
    python 05_fdm_batch.py $EXPERIMENT_FILE $NETWORK_OUT $NETWORK_OPT_OUT \
        $SUPPORT $PZ $Q0 $BRICK_LENGTH $OPT_METHOD $MAX_ITERS $TOL
    echo "*****Complete*****"

    NODES_OUT="${TEST_FOLDER}${NODES_CSV_NAME}"
    EDGES_OUT="${TEST_FOLDER}${EDGES_CSV_NAME}"

    echo "=================="
    echo "Computing stats of ${EXPERIMENT} with extra support at ${SUPPORT}"
    python 06_fdm_batch_stats.py $NETWORK_OUT $NETWORK_OPT_OUT \
        $NODES_OUT $EDGES_OUT $NODE_SHIFT_BACK $ROUND_PREC
    echo "*****Complete*****"

done

echo "*Sweep complete!!*"
echo "=================="

#!/bin/sh

# naming
EXPERIMENT="arch_state_17"  # arch_state_17
NETWORK_NAME="compression_network.json"
OPT_NETWORK_NAME="compression_network_opt.json"

# csv
NODES_CSV_NAME="nodes.csv"
EDGES_CSV_NAME="edges.csv"
SEQ_CSV_NAME="sequence.csv"

# stats controls
NODE_SHIFT_BACK="2"
ROUND_PREC="3"
REF_SUPPORTS="0,35"

# directories
LOCAL_FOLDER="../data/json/"
EXPERIMENT_FOLDER="${LOCAL_FOLDER}${EXPERIMENT}/"

# sequence csv
SEQUENCE_OUT="${EXPERIMENT_FOLDER}${SEQ_CSV_NAME}"

# Check the sequence csv exists or not
if [ -f "$SEQUENCE_OUT" ] ; then
    rm "$SEQUENCE_OUT"
    echo "$SEQUENCE_OUT was removed"
fi

# loop
for SUPPORT in {0..35}  # 35 for arch_state_17, 53 for arch_state_25
do
    TEST_FOLDER="${EXPERIMENT_FOLDER}${SUPPORT}/"
    NETWORK_OUT="${TEST_FOLDER}${NETWORK_NAME}"
    NETWORK_OPT_OUT="${TEST_FOLDER}${OPT_NETWORK_NAME}"
    NODES_OUT="${TEST_FOLDER}${NODES_CSV_NAME}"
    EDGES_OUT="${TEST_FOLDER}${EDGES_CSV_NAME}"

    echo "=================="
    echo "Computing stats of ${EXPERIMENT} with extra support at ${SUPPORT}"
    python 06_fdm_batch_stats.py $NETWORK_OUT $NETWORK_OPT_OUT \
        $NODES_OUT $EDGES_OUT $NODE_SHIFT_BACK $ROUND_PREC
    echo "Calculating sequence stats"
    python 07_fdm_batch_seq.py $NETWORK_OUT $NETWORK_OPT_OUT \
        $SEQUENCE_OUT $NODE_SHIFT_BACK $ROUND_PREC $REF_SUPPORTS
done

echo "*Stats Sweep complete!!*"
echo "========================"

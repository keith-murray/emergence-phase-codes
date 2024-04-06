#!/bin/bash

# Initialize and load modules
source /etc/profile
module load anaconda/2023a-tensorflow

# Output task info
echo "My task ID: $LLSUB_RANK"
echo "Number of Tasks: $LLSUB_SIZE"

# Execute the Python script for this task
python /home/gridsan/kmurray/mod-rnn/modularRNN/pipeline.py $LLSUB_RANK /home/gridsan/kmurray/mod-rnn/data/hyperparameter_2

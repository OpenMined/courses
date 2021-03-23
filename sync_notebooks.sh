#!/bin/bash

# Some of the notebooks used in this course are used as integration tests for PySyft. However, not all the 
# lessons in the course use PySyft, so we need to keep a separate repository of code related to the course.
# Then, we need to synchronize the notebooks used for PySyft tests and the notebooks used in the course. This
# script is intended to perform the synchronization.

# First, make sure everything is up to date. Assuming we've cloned the PySyft repo into this directory.
cd PySyft
git pull

cd ..
git pull

# Copy over Federated Learning content
# I made some changes to the notebooks that weren't propagated to the integration test notebooks,
# so commenting this out for now.
# cp -r PySyft/examples/private-ai-series/duet_basics/* federated-learning/duet_basics
# cp -r PySyft/examples/private-ai-series/duet_fl/* federated-learning/duet_fl
# cp -r PySyft/examples/private-ai-series/duet_iris_classifier/* federated-learning/duet_iris_classifier

# Copy over Split Learning content from PySft to 
cp -r PySyft/examples/private-ai-series/01_splitnn/* split-nn/splitnn
cp -r PySyft/examples/private-ai-series/02_multilimb_splitnn/* split-nn/multilimb-splitnn
cp -r PySyft/examples/private-ai-series/03_attacks_on_splitnn/* split-nn/attacks-on-splitnn



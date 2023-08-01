#!/bin/bash
echo "Pulling repo SZ_2D_thermal_structure."
git pull

echo "Pulling repo ParametricModelUtils."
cd ParametricModelUtils
git pull
cd ..

export OMP_NUM_THREADS=1
export PATH_TO_SZ_CODE=$PWD
export PATH_TO_PMU=$PWD/ParametricModelUtils
export PYTHONPATH=$PYTHONPATH:$PATH_TO_SZ_CODE
export PYTHONPATH=$PYTHONPATH:$PATH_TO_PMU

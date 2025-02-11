# nohup ./tmpgrpo.sh > grpo_phi4_0207.log 2>&1 &
export LD_LIBRARY_PATH=/anaconda/envs/azureml_py38/lib:$LD_LIBRARY_PATH 
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/mnt/batch/tasks/shared/LS_root/mounts/clusters/esus1h100a2/code/Cache/HF/"
export ACCELERATE_LOG_LEVEL=info
export DISABLE_MLFLOW_INTEGRATION='TRUE'
export TF_ENABLE_ONEDNN_OPTS=0
export CONDA_PYTHON_EXE="/anaconda/envs/azureml_py38/bin/python"
# export OMP_NUM_THREADS=1


accelerate launch --config_file phi4grpo_zero2.yaml --num_processes=2 phi4grpo.py --config phi4grpo_trl.yaml
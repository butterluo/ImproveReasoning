# nohup ./tmpgrpo.sh > grpo_phi4_0207.log 2>&1 &
export LD_LIBRARY_PATH=[Your lib of Conda Env]:$LD_LIBRARY_PATH 
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="[Your HuggingFace Cache Path]"
export ACCELERATE_LOG_LEVEL=info
export DISABLE_MLFLOW_INTEGRATION='TRUE'
export TF_ENABLE_ONEDNN_OPTS=0
export CONDA_PYTHON_EXE="[Your python exe of Conda Env]"
# export OMP_NUM_THREADS=1


accelerate launch --config_file phi4grpo_zero2.yaml --num_processes=2 phi4grpo.py --config phi4grpo_trl.yaml
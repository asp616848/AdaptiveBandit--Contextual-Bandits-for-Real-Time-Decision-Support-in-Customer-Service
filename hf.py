from huggingface_hub import upload_folder

upload_folder(
    repo_id="abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b",
    folder_path=r"Multi-Turn RL\Qwen2.5-7B-Instruct-merged",
)
# cd "b:\College\RL\AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service" && KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="Multi-Turn RL" MPLBACKEND=Agg TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "/b/ProgramFiles/miniconda3/envs/bandit/python.exe" "Multi-Turn RL/Simulation_4/scripts/phase10_full_pipeline.py" --artifacts-root "Multi-Turn RL/Simulation_4/artifacts" --output-subdir run_numerical --eval-episodes 200 --n-envs 1 --skip-training --skip-validation --demo-episodes 6 2>&1
# cd "b:\College\RL\AdaptiveBandit--Contextual-Bandits-for-Real-Time-Decision-Support-in-Customer-Service" && KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="Multi-Turn RL" MPLBACKEND=Agg TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "/b/ProgramFiles/miniconda3/envs/bandit/python.exe" "Multi-Turn RL/Simulation_4/scripts/phase10_full_pipeline.py" --artifacts-root "Multi-Turn RL/Simulation_4/artifacts" --output-subdir run_numerical --timesteps 1000000 --eval-episodes 500 --n-envs 1 --continue-from _none_ --skip-validation 2>&1
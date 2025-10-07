set -x


export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/alfred-debug/train.parquet" \
    --test_path "data/alfred-debug/test.parquet" \
    --force_gen


echo "start main ppo"

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae\
    algorithm.gamma=0.99\
    algorithm.lam=0.99 \
    data.train_files=data/alfred-debug/train.parquet \
    data.val_files=data/alfred-debug/test.parquet \
    data.train_batch_size=50 \
    data.val_batch_size=50 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_trajectory_length=7000 \
    data.image_key=images \
    data.truncation=error \
    actor_rollout_ref.model.path="your_actor_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    +actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.test_temp=0.01 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.temperature=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="your_critic_path" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    +critic.model.freeze_vision_tower=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=3 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='your_project_name' \
    trainer.experiment_name='your_experiment_name' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=3 \
    +trainer.save_start=6 \
    trainer.test_freq=3 \
    +trainer.test_start=0 \
    trainer.total_training_steps=15 \
    trainer.default_local_dir="your_local_dir" \
    rollout_manager.max_turns=3 \
    +rollout_manager.val_max_turns=3 \
    rollout_manager.window_size=1 \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=False  \
    trainer.val_generations_to_log_to_wandb=4 \
    +trainer.curriculum_learning=disable \
    +trainer.max_ckpt_to_keep=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=7200 \
    +rollout_manager.eval_only=False \
    rollout_manager.base_url="your_base_url" \
    2>&1 | tee alfred.log
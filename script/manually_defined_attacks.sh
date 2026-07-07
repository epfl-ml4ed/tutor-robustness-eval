# !/bin/bash

# # manual prompts vs tutor with reasoning
PROMPT_GENERATOR_PROMPTS=(
  collected_prompts/contextual_manipulation_manually_refined.json
  collected_prompts/direct_request_manually_refined.json
  collected_prompts/emotional_threat_manually_refined.json
  collected_prompts/intentional_wrong_answer_manually_refined.json
  collected_prompts/interpersonal_influence_manually_refined.json
  collected_prompts/request_shaping_manually_refined.json
)


for pg_prompt in "${PROMPT_GENERATOR_PROMPTS[@]}"; do
  strategy_name=$(basename "${pg_prompt%.*}")
  python evaluation_parallel.py \
      --config-name parallel_evaluation_prompt_generator \
      evaluation.strategies="[\"$(basename "${pg_prompt%.*}")\"]" \
      attacker.type="manual_prompts" \
      tutor.type="tutor_with_reasoning" \
      agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.tutor.vllm_port=8000 \
      agent.tutor.base_url="http://localhost:8000/v1" \
      agent.tutor.prompt_file="src/configs/prompt/tutor/default_with_reason.txt" \
      agent.tutor.response_format=GuidancePrompt \
      agent.prompt_generator.existing_prompts_path="$pg_prompt" \
      wandb.name="manual_prompts__tutor_with_reasoning__${strategy_name}"
done


# # manual prompts vs base incontext tutor
PROMPT_GENERATOR_PROMPTS=(
  collected_prompts/contextual_manipulation_manually_refined.json
  collected_prompts/direct_request_manually_refined.json
  collected_prompts/emotional_threat_manually_refined.json
  collected_prompts/intentional_wrong_answer_manually_refined.json
  collected_prompts/interpersonal_influence_manually_refined.json
  collected_prompts/request_shaping_manually_refined.json
)


for pg_prompt in "${PROMPT_GENERATOR_PROMPTS[@]}"; do
  strategy_name=$(basename "${pg_prompt%.*}")
  python evaluation_parallel.py \
      --config-name parallel_evaluation_prompt_generator \
      evaluation.strategies="[\"$(basename "${pg_prompt%.*}")\"]" \
      attacker.type="manual_prompts" \
      tutor.type="base_incontext_tutor" \
      agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.tutor.vllm_port=8000 \
      agent.tutor.base_url="http://localhost:8000/v1" \
      agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
      agent.tutor.response_format=null \
      agent.prompt_generator.existing_prompts_path="$pg_prompt" \
      wandb.name="manual_prompts__base_incontext_tutor__${strategy_name}"
done


# # manual prompts vs multi_agent_tutor
PROMPT_GENERATOR_PROMPTS=(
  collected_prompts/contextual_manipulation_manually_refined.json
  collected_prompts/direct_request_manually_refined.json
  collected_prompts/emotional_threat_manually_refined.json
  collected_prompts/intentional_wrong_answer_manually_refined.json
  collected_prompts/interpersonal_influence_manually_refined.json
  collected_prompts/request_shaping_manually_refined.json
)


for pg_prompt in "${PROMPT_GENERATOR_PROMPTS[@]}"; do
  strategy_name=$(basename "${pg_prompt%.*}")
  python evaluation_parallel.py \
      --config-name parallel_evaluation_prompt_generator \
      evaluation.strategies="[\"$(basename "${pg_prompt%.*}")\"]" \
      attacker.type="manual_prompts" \
      tutor.type="multi_agent_tutor" \
      agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.tutor.vllm_port=8000 \
      agent.tutor.base_url="http://localhost:8000/v1" \
      agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
      agent.tutor.response_format=null \
      agent.prompt_generator.existing_prompts_path="$pg_prompt" \
      agent.reflector.tutor_model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.reflector.tutor_vllm_port=8000 \
      agent.reflector.tutor_base_url="http://localhost:8000/v1" \
      playground.tutor_reflection=true \
      wandb.name="manual_prompts__multi_agent_tutor__${strategy_name}"
done
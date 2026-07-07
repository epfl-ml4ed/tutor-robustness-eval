# !/bin/bash


# llm generated attacks vs base incontext tutor
PROMPT_GENERATOR_PROMPTS=(
  src/configs/prompt/prompt_generator/contextual_manipulation.txt
  src/configs/prompt/prompt_generator/direct_request.txt
  src/configs/prompt/prompt_generator/emotional_threat.txt
  src/configs/prompt/prompt_generator/interpersonal_influence.txt
  src/configs/prompt/prompt_generator/intentional_wrong_answer.txt
  src/configs/prompt/prompt_generator/request_shaping.txt
)


for pg_prompt in "${PROMPT_GENERATOR_PROMPTS[@]}"; do
  strategy_name=$(basename "${pg_prompt%.*}")
  python evaluation_parallel.py \
      --config-name parallel_evaluation_prompt_generator \
      evaluation.strategies="[\"$(basename "${pg_prompt%.*}")\"]" \
      attacker.type="llm_based_questions" \
      tutor.type="base_incontext_tutor" \
      agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.tutor.vllm_port=8000 \
      agent.tutor.base_url="http://localhost:8000/v1" \
      agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
      agent.tutor.response_format=null \
      agent.prompt_generator.prompt_file="$pg_prompt" \
      wandb.name="llm_based_questions__base_incontext_tutor__${strategy_name}"
done


# llm generated attacks vs tutor_with_reasoning
PROMPT_GENERATOR_PROMPTS=(
  src/configs/prompt/prompt_generator/contextual_manipulation.txt
  src/configs/prompt/prompt_generator/direct_request.txt
  src/configs/prompt/prompt_generator/emotional_threat.txt
  src/configs/prompt/prompt_generator/interpersonal_influence.txt
  src/configs/prompt/prompt_generator/intentional_wrong_answer.txt
  src/configs/prompt/prompt_generator/request_shaping.txt
)


for pg_prompt in "${PROMPT_GENERATOR_PROMPTS[@]}"; do
  strategy_name=$(basename "${pg_prompt%.*}")
  python evaluation_parallel.py \
      --config-name parallel_evaluation_prompt_generator \
      evaluation.strategies="[\"$(basename "${pg_prompt%.*}")\"]" \
      attacker.type="llm_based_questions" \
      tutor.type="tutor_with_reasoning" \
      agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.tutor.vllm_port=8000 \
      agent.tutor.base_url="http://localhost:8000/v1" \
      agent.tutor.prompt_file="src/configs/prompt/tutor/default_with_reason.txt" \
      agent.tutor.response_format=GuidancePrompt \
      agent.prompt_generator.prompt_file="$pg_prompt" \
      wandb.name="llm_based_questions__tutor_with_reasoning__${strategy_name}"
done




# llm generated attacks vs multi_agent_tutor
PROMPT_GENERATOR_PROMPTS=(
  src/configs/prompt/prompt_generator/contextual_manipulation.txt
  src/configs/prompt/prompt_generator/direct_request.txt
  src/configs/prompt/prompt_generator/emotional_threat.txt
  src/configs/prompt/prompt_generator/interpersonal_influence.txt
  src/configs/prompt/prompt_generator/intentional_wrong_answer.txt
  src/configs/prompt/prompt_generator/request_shaping.txt
)


for pg_prompt in "${PROMPT_GENERATOR_PROMPTS[@]}"; do
  strategy_name=$(basename "${pg_prompt%.*}")
  python evaluation_parallel.py \
      --config-name parallel_evaluation_prompt_generator \
      evaluation.strategies="[\"$(basename "${pg_prompt%.*}")\"]" \
      attacker.type="llm_based_questions" \
      tutor.type="multi_agent_tutor" \
      agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.tutor.vllm_port=8000 \
      agent.tutor.base_url="http://localhost:8000/v1" \
      agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
      agent.tutor.response_format=null \
      playground.tutor_reflection=true \
      agent.prompt_generator.prompt_file="$pg_prompt" \
      agent.reflector.tutor_model_name="Qwen/Qwen2.5-7B-Instruct" \
      agent.reflector.tutor_vllm_port=8000 \
      agent.reflector.tutor_base_url="http://localhost:8000/v1" \
      wandb.name="llm_based_questions__multi_agent_tutor__${strategy_name}"
done
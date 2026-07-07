# !/bin/bash

# student with reasoning vs base incontext tutor
python evaluation_parallel.py \
    --config-name parallel_evaluation \
    evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
    attacker.type="student_with_reasoning" \
    tutor.type="base_incontext_tutor" \
    agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
    agent.tutor.vllm_port=8000 \
    agent.tutor.base_url="http://localhost:8000/v1" \
    agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
    agent.student.prompt_file="src/configs/prompt/student/default_with_reason.txt" \
    agent.tutor.response_format=null \
    wandb.name="student_with_reasoning__base_incontext_tutor"



# student with reasoning vs multi_agent_tutor
python evaluation_parallel.py \
    --config-name parallel_evaluation \
    evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
    attacker.type="student_with_reasoning" \
    tutor.type="multi_agent_tutor" \
    agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
    agent.tutor.vllm_port=8000 \
    agent.tutor.base_url="http://localhost:8000/v1" \
    agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
    agent.student.prompt_file="src/configs/prompt/student/default_with_reason.txt" \
    agent.tutor.response_format=null \
    agent.reflector.tutor_model_name="Qwen/Qwen2.5-7B-Instruct" \
    agent.reflector.tutor_vllm_port=8000 \
    agent.reflector.tutor_base_url="http://localhost:8000/v1" \
    playground.tutor_reflection=true \
    wandb.name="student_with_reasoning__multi_agent_tutor"


# student with reasoning vs tutor with reasoning
python evaluation_parallel.py \
          --config-name parallel_evaluation \
          evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
          attacker.type="student_with_reasoning" \
          tutor.type="tutor_with_reasoning" \
          agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
          agent.tutor.vllm_port=8000 \
          agent.tutor.base_url="http://localhost:8000/v1" \
          agent.tutor.prompt_file="src/configs/prompt/tutor/default_with_reason.txt" \
          agent.student.prompt_file="src/configs/prompt/student/default_with_reason.txt" \
          agent.tutor.response_format=GuidancePrompt \
          wandb.name="student_with_reasoning__tutor_with_reasoning"

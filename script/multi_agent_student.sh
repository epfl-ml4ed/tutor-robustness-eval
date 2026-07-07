# !/bin/bash

# multi_agent_student vs base_incontext_tutor
python evaluation_parallel.py \
          --config-name parallel_evaluation \
          evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
          attacker.type="multi_agent_student" \
          tutor.type="base_incontext_tutor" \
          agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
          agent.tutor.vllm_port=8000 \
          agent.tutor.base_url="http://localhost:8000/v1" \
          agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
          agent.student.prompt_file="src/configs/prompt/student/default.txt" \
          agent.student.response_format="AdversarialPromptWithoutReason" \
          agent.tutor.response_format=null \
          playground.student_reflection=true \
          wandb.name="multi_agent_student__base_incontext_tutor"



# multi_agent_student vs tutor_with_reasoning
python evaluation_parallel.py \
          --config-name parallel_evaluation \
          evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
          attacker.type="multi_agent_student" \
          tutor.type="tutor_with_reasoning" \
          agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
          agent.tutor.vllm_port=8000 \
          agent.tutor.base_url="http://localhost:8000/v1" \
          agent.tutor.prompt_file="src/configs/prompt/tutor/default_with_reason.txt" \
          agent.student.prompt_file="src/configs/prompt/student/default.txt" \
          agent.student.response_format="AdversarialPromptWithoutReason" \
          agent.tutor.response_format="GuidancePrompt" \
          playground.student_reflection=true \
          wandb.name="multi_agent_student__tutor_with_reasoning"



# multi_agent_student vs multi_agent_tutor
python evaluation_parallel.py \
          --config-name parallel_evaluation \
          evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
          attacker.type="multi_agent_student" \
          tutor.type="multi_agent_tutor" \
          agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
          agent.tutor.vllm_port=8000 \
          agent.tutor.base_url="http://localhost:8000/v1" \
          agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
          agent.student.prompt_file="src/configs/prompt/student/default.txt" \
          agent.student.response_format="AdversarialPromptWithoutReason" \
          agent.tutor.response_format=null \
          agent.reflector.tutor_model_name="Qwen/Qwen2.5-7B-Instruct" \
          agent.reflector.tutor_vllm_port=8000 \
          agent.reflector.tutor_base_url="http://localhost:8000/v1" \
          playground.student_reflection=true \
          playground.tutor_reflection=true \
          wandb.name="multi_agent_student__multi_agent_tutor"
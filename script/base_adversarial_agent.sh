# !/bin/bash

# base_student_adversarial_agent vs base_incontext_tutor
python evaluation_parallel.py \
          --config-name parallel_evaluation \
          evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
          attacker.type="base_student_adversarial_agent" \
          tutor.type="base_incontext_tutor" \
          agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
          agent.tutor.vllm_port=8000 \
          agent.tutor.base_url="http://localhost:8000/v1" \
          agent.tutor.prompt_file="src/configs/prompt/tutor/default.txt" \
          agent.student.prompt_file="src/configs/prompt/student/default_6.txt" \
          agent.student.response_format="AdversarialPromptWithoutReason" \
          agent.tutor.response_format=null \
          wandb.name="base_student_adversarial_agent__base_incontext_tutor"



# base_student_adversarial_agent vs tutor with reasoning
python evaluation_parallel.py \
          --config-name parallel_evaluation \
          evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
          attacker.type="base_student_adversarial_agent" \
          tutor.type="tutor_with_reasoning" \
          agent.tutor.model_name="Qwen/Qwen2.5-7B-Instruct" \
          agent.tutor.vllm_port=8000 \
          agent.tutor.base_url="http://localhost:8000/v1" \
          agent.tutor.prompt_file="src/configs/prompt/tutor/default_with_reason.txt" \
          agent.student.prompt_file="src/configs/prompt/student/default.txt" \
          agent.student.response_format="AdversarialPromptWithoutReason" \
          agent.tutor.response_format=GuidancePrompt \
          wandb.name="base_student_adversarial_agent__tutor_with_reasoning"



# base_student_adversarial_agent vs multi_agent_tutor
python evaluation_parallel.py \
          --config-name parallel_evaluation \
          evaluation.strategies="['direct_request','emotional_threat','intentional_wrong_answer']" \
          attacker.type="base_student_adversarial_agent" \
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
          playground.tutor_reflection=true \
          wandb.name="base_student_adversarial_agent__multi_agent_tutor"
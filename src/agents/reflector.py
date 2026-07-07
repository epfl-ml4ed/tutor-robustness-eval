"""
Judge agent implementation for evaluating tutor-student interactions.
"""
import json
from typing import Optional

from src.agents.llm_agent import LLMAgent
from src.utils.prompt_loader import PromptLoader
from pydantic import BaseModel

class ReflectorFormat(BaseModel):
    revised_response: str


class Reflector(LLMAgent):
    """
    LLM-as-a-Reflector Agent.
    """

    def __init__(
        self,
        student_model_name: str = "",
        student_use_vllm: bool = False,
        student_vllm_base_url: str = None,
        tutor_model_name: str = "",
        tutor_use_vllm: bool = False,
        tutor_vllm_base_url: str = None,
        device: str = "auto",
        temperature: float = 1e-5,
        max_new_tokens: int = 500,
        to_tutor_prompt_file: str = "",
        to_student_prompt_file: str = "",
        problem: str = "",
        answer: str = "",
        experiment_logger=None,
        dataset_name: str = None,
        domain: str = None,
        answer_text: str = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Reflector with role-specific prompts.

        Args:
            model_name: Name of the HuggingFace model to load
            device: Device to run the model on
            temperature: Sampling temperature
            problem: The math problem
            answer: The correct answer
            to_tutor_prompt_file: Prompt file specifically for tutor reflection
            to_tutor_system_prompt: System prompt specifically for tutor reflection
            to_student_prompt_file: Prompt file specifically for student reflection
            to_student_system_prompt: System prompt specifically for student reflection
            experiment_logger: ExperimentLogger instance
        """
        # Initialize parent LLMAgent (loads model) with empty system prompt initially
        self.device = device
        self.student_model_name = student_model_name
        self.student_use_vllm = student_use_vllm
        self.student_vllm_base_url = student_vllm_base_url
        self.tutor_model_name = tutor_model_name
        self.tutor_use_vllm = tutor_use_vllm
        self.tutor_vllm_base_url = tutor_vllm_base_url
        
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.experiment_logger = experiment_logger
        self.seed = seed
        

        # super().__init__(
        #     role="Reflector",
        #     model_name=model_name,
        #     device=device,
        #     temperature=temperature,
        #     max_new_tokens=max_new_tokens,
        #     system_prompt="",
        #     experiment_logger=experiment_logger,
        #     use_vllm=use_vllm,
        #     vllm_base_url=vllm_base_url,
        # )

        # Load and store role-specific prompts
        self.to_tutor_prompt = None
        self.to_student_prompt = None

        if dataset_name == "gsm8k":

            if to_tutor_prompt_file:
                self.to_tutor_prompt = PromptLoader.load_prompt(
                    prompt_file=to_tutor_prompt_file,
                    replacements={
                        "${experiment.problem}": problem,
                        "${experiment.answer}": answer,
                    },
                )

            if to_student_prompt_file:
                self.to_student_prompt = PromptLoader.load_prompt(
                    prompt_file=to_student_prompt_file,
                    replacements={
                        "${experiment.problem}": problem,
                        "${experiment.answer}": answer,
                    },
                )
        
        elif dataset_name == "mmlu":
            
            if to_tutor_prompt_file:
                self.to_tutor_prompt = PromptLoader.load_prompt(
                    prompt_file=to_tutor_prompt_file,
                    replacements={
                        "${experiment.problem}": problem,
                        "${experiment.answer}": answer,
                        "${experiment.answer_text}": answer_text,
                    },
                )

            if to_student_prompt_file:
                self.to_student_prompt = PromptLoader.load_prompt(
                    prompt_file=to_student_prompt_file,
                    replacements={
                        "${experiment.problem}": problem,
                        "${experiment.answer}": answer,
                        "${experiment.answer_text}": answer_text,
                    },
                )

    def reply(self, user_input: str, reflection_role: str = None):
        """
        Generate a response with optional role-specific prompt.

        Args:
            user_input: The input message
            with_history: Whether to use conversation history
            reflection_role: Optional role ("tutor" or "student") to use role-specific prompt
        """
        response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reflector-format",
                        "schema": ReflectorFormat.model_json_schema(),
                    }
                }

        if reflection_role == "to_tutor" and self.to_tutor_prompt:
            user_input = self.to_tutor_prompt.replace(
                "${teacher.response}", user_input
            )
            super().__init__(
                role="Reflector",
                model_name=self.tutor_model_name,
                device=self.device,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                system_prompt="",
                response_format=response_format,
                experiment_logger=self.experiment_logger,
                use_vllm=self.tutor_use_vllm,
                vllm_base_url=self.tutor_vllm_base_url,
                seed=self.seed,
            )
        elif reflection_role == "to_student" and self.to_student_prompt:
            user_input = self.to_student_prompt.replace(
                "${student.response}", user_input
            )
            super().__init__(
                role="Reflector",
                model_name=self.student_model_name,
                device=self.device,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                system_prompt="",
                response_format=response_format,
                experiment_logger=self.experiment_logger,
                use_vllm=self.student_use_vllm,
                vllm_base_url=self.student_vllm_base_url,
                seed=self.seed,
            )
        return super().reply(user_input)
    
    def _postprocess_output(self, agent_output):
        if isinstance(agent_output, ReflectorFormat):
            return json.dumps(agent_output.model_dump(), ensure_ascii=False)
        elif isinstance(agent_output, str):
            try:
                json.loads(agent_output) 
                return agent_output
            except json.JSONDecodeError:
                return agent_output + '"}'
        else:
            raise ValueError(f"Unexpected agent output type: {agent_output}")
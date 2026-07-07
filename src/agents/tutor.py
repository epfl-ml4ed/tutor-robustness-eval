"""
Tutor agent implementations (LLM-based and manual).
"""
import json

from src.agents.llm_agent import LLMAgent
from src.agents.manual_agent import ManualAgent
from src.utils.prompt_loader import PromptLoader
from pydantic import BaseModel
from typing import Any, Optional

class GuidancePrompt(BaseModel):
    reason: str
    content: str

class Tutor(LLMAgent):
    """
    Tutor Agent (LLM-based).

    The tutor guides the student step-by-step to understand and solve the problem,
    but must **never** reveal the final numeric answer explicitly.
    """

    def __init__(
        self,
        prompt_file: str,
        model_name: str = "",
        device: str = "auto",
        temperature: float = 1e-5,
        max_new_tokens: int = 500,
        response_format: Optional[Any] = None,
        problem: str = "",
        answer: str = "",
        experiment_logger=None,
        use_vllm: bool = False,
        vllm_base_url: str = None,
        dataset_name: str = None,
        domain: str = None,
        answer_text: str = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Tutor agent.

        Args:
            model_name: Hugging Face model name.
            device: Device identifier, e.g., "cuda:0", "auto".
            system_prompt: Optional custom prompt text.
            prompt_file: Optional path to a .txt file with the system prompt.
            problem: The math problem text.
            answer: The correct numeric answer (used only for reference; never revealed).
            answer_text: The correct answer in text form (used only for reference; never revealed).
            experiment_logger: Optional ExperimentLogger instance.
        """
        self.device = device
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.dataset_name = dataset_name
        self.domain = domain
        self.answer_text = answer_text
        if isinstance(response_format, str):
            if response_format == "GuidancePrompt":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "guidance-prompt",
                        "schema": GuidancePrompt.model_json_schema(),
                    }
                }
        self.response_format = response_format

        # Load prompt using PromptLoader
        if dataset_name == "gsm8k":
            tutor_prompt = PromptLoader.load_prompt(
                prompt_file=prompt_file,
                replacements={
                    "${experiment.problem}": problem,
                    "${experiment.answer}": answer,
                },
            )
        elif dataset_name == "mmlu":
            tutor_prompt = PromptLoader.load_prompt(
                prompt_file=prompt_file,
                replacements={
                    "${experiment.problem}": problem,
                    "${experiment.answer}": answer,
                    "${experiment.answer_text}": answer_text,
                    "${experiment.domain}": domain,
                },
            )

        # Initialize LLMAgent
        super().__init__(
            role="Tutor",
            model_name=model_name,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=tutor_prompt,
            response_format=response_format,
            experiment_logger=experiment_logger,
            use_vllm=use_vllm,
            vllm_base_url=vllm_base_url,
            seed=seed,
        )

        self.logger.info(f"🧑‍🏫 Tutor Prompt: {tutor_prompt}")

    def _preprocess_input(self, user_input) -> str:
        """
        Preprocess student input by extracting content from JSON if present.

        Students may send messages in JSON format with a "content" key,
        either as a JSON string or as a dict object.
        This method extracts the actual message content.

        Args:
            user_input: Raw input from the student (str or dict).

        Returns:
            Extracted content string.
        """
        # Handle dict input directly
        if isinstance(user_input, str):
            try:
                return json.loads(user_input).get("content", user_input)
            except:
                return user_input
        elif isinstance(user_input, dict):
            return user_input.get("content", str(user_input))
    
    def _postprocess_output(self, agent_output) -> list[str]:
        """
        Preprocess a list of student inputs.

        Args:
            user_inputs: List of raw inputs from the student.

        Returns:
            List of extracted content strings.
        """
        if isinstance(agent_output, GuidancePrompt):
            return json.dumps(agent_output.model_dump(), ensure_ascii=False)
        elif isinstance(agent_output, dict):
            return json.dumps(agent_output, ensure_ascii=False)
        elif isinstance(agent_output, str):
            if self.response_format is None:
                return agent_output
            else:
                try:
                    json.loads(agent_output) 
                    return agent_output
                except json.JSONDecodeError:
                    return agent_output + '"}'
        else:
            raise ValueError(f"Unexpected agent output type: {agent_output}")


class ManualTutor(ManualAgent):
    """
    Manual Tutor Agent (human-controlled).

    A human-controlled version of the Tutor agent.
    Useful for debugging or collecting data from real human tutors
    when testing student (AI or human) strategies.
    """

    def __init__(
        self,
        problem: str = "",
        experiment_logger=None,
    ):
        """
        Initialize manual tutor.

        Args:
            problem: Math problem to solve (for context).
            experiment_logger: ExperimentLogger instance.
        """
        self.problem = problem

        super().__init__(
            role="Tutor",
            system_prompt="",
            experiment_logger=experiment_logger,
        )

        self.logger.info(
            "ManualTutor initialized. Please input responses manually."
        )

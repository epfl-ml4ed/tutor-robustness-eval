"""
Student agent implementations (LLM-based and manual).
"""
import json
from pydantic import BaseModel
from src.agents.llm_agent import LLMAgent
from src.agents.manual_agent import ManualAgent
from src.utils.prompt_loader import PromptLoader
from typing import Any, Optional


class AdversarialPrompt(BaseModel):
    """Schema for adversarial prompt with reason."""
    strategy: list[str]
    reason: str
    content: str


class AdversarialPromptWithoutReason(BaseModel):
    """Schema for adversarial prompt without reason."""
    strategy: list[str]
    content: str


class AutoStudent(LLMAgent):
    """
    Automated Student Agent (LLM-based).

    This agent simulates a student attempting to extract the final numeric answer
    from a tutor model by applying various "jailbreak" strategies.

    Attributes:
        json_mode: If True, the student outputs responses in structured JSON format.
    """

    def __init__(
        self,
        prompt_file: str,
        model_name: str = "",
        device: str = "auto",
        temperature: float = 1e-5,
        max_new_tokens: int = 500,
        response_format: Optional[Any] = AdversarialPrompt,
        problem: str = "",
        answer: str = "",
        experiment_logger=None,
        use_vllm: bool = False,
        vllm_base_url: str = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the AutoStudent agent.

        Args:
            prompt_file: Path to the prompt template file.
            model_name: Name of the model to use.
            device: Device to run the model on, e.g. "cuda:0", "auto".
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of tokens to generate.
            response_format: Response format schema (AdversarialPrompt or AdversarialPromptWithoutReason).
            problem: Problem description to include in the context.
            answer: Correct answer to the problem.
            experiment_logger: Optional logger instance.
            use_vllm: Whether to use vLLM for inference.
            vllm_base_url: Base URL for vLLM server.
        """
        self.device = device
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt_file = prompt_file
        if response_format == "AdversarialPromptWithoutReason":
            response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "adversarial-prompt-without-reason",
                        "schema": AdversarialPromptWithoutReason.model_json_schema(),
                    }
                }
        if response_format == "AdversarialPrompt":
            response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "adversarial-prompt",
                        "schema": AdversarialPrompt.model_json_schema(),
                    }
                }
        self.response_format = response_format

        # Load prompt using PromptLoader
        if "sft" in prompt_file:
            student_prompt = ""
        else:
            student_prompt = PromptLoader.load_prompt(
                prompt_file=prompt_file,
                replacements={
                    "${experiment.problem}": problem,
                    "${experiment.answer}": answer,
                },
            )

        # Initialize LLMAgent
        super().__init__(
            role="Student",
            model_name=model_name,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=student_prompt,
            response_format=response_format,
            experiment_logger=experiment_logger,
            use_vllm=use_vllm,
            vllm_base_url=vllm_base_url,
            seed=seed,
        )

    def _postprocess_output(self, agent_output):
        """
        Postprocess the student output.

        Args:
            agent_output: Raw output from the model.

        Returns:
            JSON string or processed string output.

        Raises:
            ValueError: If output type is unexpected.
        """
        if isinstance(agent_output, AdversarialPrompt):
            return json.dumps(agent_output.model_dump(), ensure_ascii=False)
        elif isinstance(agent_output, AdversarialPromptWithoutReason):
            return json.dumps(agent_output.model_dump(), ensure_ascii=False)
        elif isinstance(agent_output, dict):
            return json.dumps(agent_output, ensure_ascii=False)
        elif isinstance(agent_output, str):
            try:
                json.loads(agent_output)
                return agent_output
            except json.JSONDecodeError:
                pass
            try:
                json.loads(agent_output + '"}')
                return agent_output + '"}'
            except json.JSONDecodeError:
                pass
            return agent_output
        else:
            raise ValueError(f"Unexpected agent output type: {agent_output}")

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
        if isinstance(user_input, str):
            try:
                return json.loads(user_input).get("content", user_input)
            except (json.JSONDecodeError, AttributeError):
                return user_input
        elif isinstance(user_input, dict):
            return user_input.get("content", str(user_input))
        return user_input


class ManualStudent(ManualAgent):
    """
    Manual (Human) Student Agent.

    This version allows a human to manually type in student responses.
    Useful for qualitative testing of the tutor model's behavior against
    real human input or adversarial strategies.
    """

    def __init__(
        self,
        problem: str = "",
        experiment_logger=None,
    ):
        """
        Initialize manual student.

        Args:
            problem: Math problem to solve (for context).
            experiment_logger: ExperimentLogger instance.
        """
        self.problem = problem

        super().__init__(
            role="Student",
            system_prompt="",
            experiment_logger=experiment_logger,
        )

        self.logger.info(
            "ManualStudent initialized. Please input responses manually."
        )
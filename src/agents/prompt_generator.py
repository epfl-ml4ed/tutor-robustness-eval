"""
Prompt generator for creating adversarial prompts.
"""
import json
import os
from typing import List

import rootutils
root = rootutils.setup_root(__file__, pythonpath=True)

from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.agents.llm_agent import LLMAgent
from src.utils.prompt_loader import PromptLoader


class AdversarialPrompt(BaseModel):
    """Schema for a single adversarial prompt."""
    strategy: str
    content: str


class AdversarialPromptWrapper(BaseModel):
    """Wrapper for multiple adversarial prompts."""
    results: List[AdversarialPrompt]


class PromptGenerator(LLMAgent):
    """
    Agent for generating adversarial prompts.

    Can either generate new prompts using an LLM or load existing prompts from a file.
    """

    def __init__(
        self,
        prompt_file: str,
        model_name: str = "",
        device: str = "auto",
        temperature: float = 1e-5,
        max_new_tokens: int = 500,
        prompt_number: int = 20,
        response_format: BaseModel = AdversarialPromptWrapper,
        answer: str = None,
        problem: str = "",
        experiment_logger=None,
        use_vllm: bool = False,
        vllm_base_url: str = None,
        using_existing_prompts: bool = False,
        existing_prompts_path: str = None,
    ):
        """
        Initialize the prompt generator.

        Args:
            prompt_file: Path to the prompt template file.
            model_name: Name of the model to use for generation.
            device: Device to run the model on.
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of tokens to generate.
            prompt_number: Number of prompts to generate.
            response_format: Pydantic model for structured output.
            answer: The answer to the problem.
            problem: The problem description.
            experiment_logger: Optional logger instance.
            use_vllm: Whether to use vLLM for inference.
            vllm_base_url: Base URL for vLLM server.
            using_existing_prompts: If True, load existing prompts instead of generating.
            existing_prompts_path: Path to existing prompts JSON file.
        """
        # Store model settings
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.response_format = response_format
        self.using_existing_prompts = using_existing_prompts
        self.existing_prompts_path = existing_prompts_path

        # Load prompt template with replacements if generating new prompts
        self.user_input = PromptLoader.load_prompt(
            prompt_file=prompt_file,
            replacements={
                "${experiment.problem}": problem,
                "${experiment.answer}": answer,
                "${prompt_number}": str(prompt_number),
            },
        ) if not using_existing_prompts else ""

        # Initialize parent class based on mode
        if not using_existing_prompts:
            # Initialize as LLMAgent for generating prompts
            super().__init__(
                role="prompt_generator",
                model_name=model_name,
                device=device,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                system_prompt="",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "adversarial-prompt-wrapper",
                        "schema": AdversarialPromptWrapper.model_json_schema(),
                    }
                },
                experiment_logger=experiment_logger,
                use_vllm=use_vllm,
                vllm_base_url=vllm_base_url,
            )
        else:
            # Initialize as BaseAgent for loading existing prompts
            BaseAgent.__init__(
                self,
                role="prompt_generator",
                logger=experiment_logger,
            )

    def _postprocess_output(self, agent_output) -> List[dict]:
        """
        Postprocess the generated output into a list of prompt dictionaries.

        Args:
            agent_output: Raw output from the model (JSON string).

        Returns:
            List of dictionaries with 'strategy' and 'content' keys.
        """
        agent_output = json.loads(agent_output)
        agent_output = agent_output["results"]
        prompts_dict = [{"strategy": p["strategy"], "content": p["content"]} for p in agent_output]
        return prompts_dict

    def load_existing_prompts(self) -> dict:
        """
        Load existing prompts from a JSON file.

        Returns:
            Dictionary containing the loaded prompts.

        Raises:
            FileNotFoundError: If the prompts file path is invalid or doesn't exist.
        """
        if not self.existing_prompts_path or not os.path.exists(self.existing_prompts_path):
            raise FileNotFoundError(f"Existing prompts path is invalid: {self.existing_prompts_path}")
        with open(self.existing_prompts_path, 'r', encoding='utf-8') as f:
            return json.load(f)
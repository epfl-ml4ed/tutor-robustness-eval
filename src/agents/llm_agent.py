"""
LLM-based agent implementation.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from src.agents.base_agent import BaseAgent
from typing import Optional, Any


class LLMAgent(BaseAgent):
    """
    LLM-based Agent.
    """

    def __init__(
        self,
        role: str,
        model_name: str = "",
        device: Optional[str] = "auto",
        temperature: float = 1e-5,
        max_new_tokens: int = 500,
        system_prompt: str = "",
        response_format: Optional[Any] = None,
        experiment_logger=None,
        use_vllm: bool = False,
        vllm_base_url: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize LLM-based agent.

        Args:
            role: The role of the agent (e.g., "Tutor", "Student", "Judge").
            model_name: Name of the Hugging Face or OpenAI model to use.
            device: Device to run the model on, e.g. "cuda:0", "auto".
            temperature: Sampling temperature for generation randomness.
            max_new_tokens: Maximum number of tokens to generate.
            system_prompt: Initial system prompt to set the context.
            response_format: Optional response format for structured outputs (OpenAI API and vLLM).
            experiment_logger: Optional experiment logger instance.
            use_vllm: Whether to use vLLM for inference via OpenAI-compatible API.
            vllm_base_url: Base URL for vLLM server (e.g., "http://localhost:8000/v1").
        """
        # Initialize base agent (logger, conversation history)
        super().__init__(
            role=role,
            system_prompt=system_prompt,
            logger=experiment_logger,
        )

        # Store model configuration
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.response_format = response_format
        self.use_vllm = use_vllm

        # Use vLLM via OpenAI-compatible API
        if use_vllm:
            if not vllm_base_url:
                raise ValueError("vllm_base_url must be provided when use_vllm=True")
            self.client = OpenAI(
                api_key="EMPTY",  # vLLM doesn't require API key
                base_url=vllm_base_url,
            )
            self.logger.info(f"Using vLLM server at {vllm_base_url} for model: {model_name}")
        # Use OpenAI API
        elif self.model_name.startswith("gpt"):
            self.client = OpenAI()
            self.logger.info(f"Using OpenAI API for model: {model_name}")
        # Load model locally with HuggingFace
        else:
            self.device = device
            self.logger.info(f"Loading model locally: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if device == "auto" else {"": device},
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"Model loaded successfully on device: {self.model.device}")


    def _reply(
        self,
        user_input: str = None,
        **kwargs
    ) -> str:
        """
        Generate a model reply using the conversation history.

        Note: user_input parameter is not used in this implementation as the
        conversation history is already updated by BaseAgent.reply() before
        calling this method.

        Args:
            user_input: User input (unused, kept for interface compatibility).
            **kwargs: Additional generation parameters for HuggingFace models.

        Returns:
            The generated text reply.
        """
        # Use OpenAI-compatible API (vLLM or OpenAI)
        if self.use_vllm or self.model_name.startswith("gpt"):
            if self.response_format:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages_history,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_new_tokens,
                    response_format=self.response_format,
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages_history,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
        
        # Use local HuggingFace model
        else:
            # Step 1. Convert message history into chat template text
            text = self.tokenizer.apply_chat_template(
                    self.messages_history, tokenize=False, add_generation_prompt=True
                )
            # Step 2. Tokenize input text
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            # Step 3. Generate output tokens
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            # Step 4. Remove input tokens from the output to get only the generated part
            outputs = [out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)]
            # Step 5. Decode output tokens to text
            outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[
                0
            ].strip()
            return outputs_text

    def reset_messages(self):
        """
        Reset conversation history to initial state (keep system prompt).

        Useful for agents like Judge that need to reset between evaluations.
        """
        self.conversation.reset(keep_system_prompt=True)
        self.logger.info(f"[{self.role}] Conversation history reset")
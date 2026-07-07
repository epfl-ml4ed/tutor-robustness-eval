"""
Manual (human-controlled) agent implementation.
"""
from src.agents.base_agent import BaseAgent


class ManualAgent(BaseAgent):
    """
    Agent that accepts manual (human) input instead of using a language model.

    This class does NOT load any models, making it lightweight and suitable
    for interactive testing or human-in-the-loop scenarios.
    """

    def __init__(
        self,
        role: str,
        system_prompt: str = "",
        experiment_logger=None,
    ):
        """
        Initialize manual agent.

        Args:
            role: The role of the agent (e.g., "Tutor", "Student").
            system_prompt: Initial system prompt (for context, not used in generation).
            experiment_logger: ExperimentLogger instance.
        """
        # Initialize base agent (logger, conversation history)
        super().__init__(
            role=role,
            system_prompt=system_prompt,
            logger=experiment_logger,
        )

        self.logger.info(
            f"ManualAgent initialized for role: {role}. Waiting for manual input."
        )

    def _reply(self, user_input: str, **kwargs) -> str:
        """
        Prompt the human user to input the agent's reply.

        Note: user_input parameter is not directly used in the prompt as the
        conversation history is already updated by BaseAgent.reply() before
        calling this method. It's logged for context only.

        Args:
            user_input: Latest message from the other agent (logged for context).
            **kwargs: Additional keyword arguments (unused for manual agents).

        Returns:
            The human-provided reply string.
        """
        self.logger.info(f"Other agent said: {user_input}")
        reply = input(f"[{self.role}] Please enter your reply: ")
        return reply
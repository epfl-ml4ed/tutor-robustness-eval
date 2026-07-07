"""
Utility for loading and processing agent prompts.
"""
from pathlib import Path
from typing import Dict, Optional


class PromptLoader:
    """
    Handles loading prompts from files or strings with template variable replacement.
    """

    @staticmethod
    def load_prompt(
        prompt_file: Optional[str] = None,
        system_prompt: Optional[str] = None,
        replacements: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Load a prompt with the following priority:
        1. From prompt_file (if provided and exists)
        2. From system_prompt (if provided)

        Args:
            prompt_file: Path to a text file containing the prompt.
            system_prompt: Direct prompt string.
            replacements: Dict of template variables to replace (e.g., {"${var}": "value"}).

        Returns:
            Processed prompt string with replacements applied.
        """
        # Determine prompt source
        prompt = None

        if prompt_file and Path(prompt_file).exists():
            prompt = Path(prompt_file).read_text(encoding="utf-8")
        elif system_prompt:
            prompt = system_prompt
        else:
            raise ValueError("Either prompt_file or system_prompt must be provided.")

        # Apply template replacements
        if replacements:
            for template_var, value in replacements.items():
                prompt = prompt.replace(template_var, value)

        return prompt

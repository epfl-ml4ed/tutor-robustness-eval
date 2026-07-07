"""
Judge agent implementation for evaluating tutor-student interactions.
"""
import json
from typing import Optional

from src.agents.llm_agent import LLMAgent
from src.utils.prompt_loader import PromptLoader
from pydantic import BaseModel

class JudgeTutorFormat(BaseModel):
    reason: str
    tutor_revealed: bool

class JudgeStudentFormat(BaseModel):
    reason: str
    student_correct: bool

class Judge(LLMAgent):
    """
    LLM-as-a-Judge Agent.

    Evaluates both Tutor's and Student's behavior in each turn:
    - Detects if Tutor leaks the answer.
    - Detects if Student derives the correct answer independently.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        temperature: float = 1e-5,
        max_new_tokens: int = 500,
        to_tutor_prompt_file: str = "",
        to_student_prompt_file: str = "",
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
        Initialize the Judge agent.

        Args:
            model_name: Hugging Face model name.
            device: Device identifier, e.g., "cuda:0", "auto".
            to_tutor_prompt_file: Optional path to a .txt file with the tutor system prompt.
            to_student_prompt_file: Optional path to a .txt file with the student system prompt.
            problem: The math problem text.
            answer: The correct numeric answer.
            experiment_logger: Optional ExperimentLogger instance.
        """
        self.device = device
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.experiment_logger = experiment_logger
        self.use_vllm = use_vllm
        self.vllm_base_url = vllm_base_url
        self.seed = seed

        # Load prompt using PromptLoader
        super().__init__(
            role="Judge",
            model_name=model_name,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt="",
            experiment_logger=experiment_logger,
            use_vllm=use_vllm,
            vllm_base_url=vllm_base_url,
            seed=seed,
        )

        # Load and store role-specific prompts
        self.to_tutor_prompt = None
        self.to_student_prompt = None

        if dataset_name == "mmlu":

            if to_tutor_prompt_file:
                self.to_tutor_prompt = PromptLoader.load_prompt(
                    prompt_file=to_tutor_prompt_file,
                    replacements={
                        "${experiment.problem}": problem,
                        "${experiment.answer}": answer,
                        "${experiment.answer_text}": answer_text,
                        "${experiment.domain}": domain,

                    },
                )

            if to_student_prompt_file:
                self.to_student_prompt = PromptLoader.load_prompt(
                    prompt_file=to_student_prompt_file,
                    replacements={
                        "${experiment.problem}": problem,
                        "${experiment.answer}": answer,
                        "${experiment.answer_text}": answer_text,
                        "${experiment.domain}": domain,
                    },
                )
        
        elif dataset_name == "gsm8k":
            
            if to_tutor_prompt_file:
                self.to_tutor_prompt = PromptLoader.load_prompt(
                    prompt_file=to_tutor_prompt_file,
                    replacements={
                        "${problem}": problem,
                        "${answer}": answer,
                    },
                )

            if to_student_prompt_file:
                self.to_student_prompt = PromptLoader.load_prompt(
                    prompt_file=to_student_prompt_file,
                    replacements={
                        "${problem}": problem,
                        "${answer}": answer,
                    },
                )
    
    def reply(self, user_input: str, judge_role: str = None):
        """
        Generate a reply from the Judge agent.

        Args:
            user_input: The input text to the Judge.
            judge_role: Optional role context for the Judge.
        Returns:
            The generated reply string.
        """
        if judge_role == "to_tutor" and self.to_tutor_prompt:
            user_input = self.to_tutor_prompt.replace(
                "${teacher.response}", user_input
            )
            super().__init__(
                role="Judge",
                model_name=self.model_name,
                device=self.device,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                system_prompt="",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge-tutor-format",
                        "schema": JudgeTutorFormat.model_json_schema(),
                    }
                },
                experiment_logger=self.experiment_logger,
                use_vllm=self.use_vllm,
                vllm_base_url=self.vllm_base_url,
                seed=self.seed,
            )
        elif judge_role == "to_student" and self.to_student_prompt:
            user_input = self.to_student_prompt.replace(
                "${student.response}", user_input
            )
            super().__init__(
                role="Judge",
                model_name=self.model_name,
                device=self.device,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                system_prompt="",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge-student-format",
                        "schema": JudgeStudentFormat.model_json_schema(),
                    }
                },
                experiment_logger=self.experiment_logger,
                use_vllm=self.use_vllm,
                vllm_base_url=self.vllm_base_url,
                seed=self.seed,
            )
        return super().reply(user_input)
    
    def _postprocess_output(self, agent_output):
        if isinstance(agent_output, JudgeTutorFormat) or isinstance(agent_output, JudgeStudentFormat):
            return json.dumps(agent_output.model_dump(), ensure_ascii=False)
        elif isinstance(agent_output, dict):
            return json.dumps(agent_output, ensure_ascii=False)
        elif isinstance(agent_output, str):
            try:
                json.loads(agent_output) 
                return agent_output
            except json.JSONDecodeError:
                return agent_output + '"}'
        else:
            raise ValueError(f"Unexpected agent output type: {agent_output}")



    def evaluate_turn(self, tutor_reply: str, student_reply: str = "") -> dict:
        """
        Evaluate one turn of conversation.

        Args:
            tutor_reply: The latest tutor message.
            student_reply: The following student message (optional).

        Returns:
            Dictionary with evaluation verdict (tutor_leakage, student_correct, reason).
        """
        # Construct conversation snippet
        if not student_reply.strip():
            conversation_snippet = f"Tutor: {tutor_reply}"
        else:
            conversation_snippet = f"Tutor: {tutor_reply}\nStudent: {student_reply}"

        # Reset conversation history for fresh evaluation
        self.reset_messages()

        # Get judge's evaluation
        result = self.reply(conversation_snippet)

        # Parse JSON response
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            self.logger.warning(f"⚠️ Failed to parse Judge output as JSON: {result}")
            parsed = {
                "tutor_leakage": False,
                "student_correct": False,
                "reason": "Invalid JSON from Judge.",
            }

        return parsed

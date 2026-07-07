"""
Unified playground for tutor agent evaluation.

Supports two modes:
1. Student-Tutor interaction (conversational attacker)
2. Prompt Generator-Tutor interaction (pre-generated prompts attacker)

The mode is determined by the attacker_type in the configuration.
"""
import json
from typing import List, Optional
import re
import random

from src.agents.base_agent import BaseAgent
from src.agents.prompt_generator import PromptGenerator
from src.utils.logger import setup_experiment_logger
from src.utils.prompt_loader import PromptLoader


class UnifiedPlayground:
    """
    Unified interactive environment for tutor agent evaluation.

    This class manages:
    - Multi-turn interactions with a tutor agent
    - Support for both student and prompt generator attackers
    - Logging of each interaction (messages, responses, strategies)
    - Optional judge evaluation for answer leakage/correctness detection
    - Optional reflector for agent refinement
    - Conversation data export for analysis

    Attributes:
        attacker_type (str): Type of attacker ("student" or "prompt_generator")
        attacker (BaseAgent): Attacker agent instance
        tutor (BaseAgent): Tutor agent instance under evaluation
        judge (BaseAgent): Optional judge agent for evaluation
        reflector (BaseAgent): Optional reflector agent for refinement
        problem (str): The math problem being discussed
        answer (str): Ground-truth answer
        max_turns (int): Maximum number of interaction rounds
        conversation_log (list[dict]): Stores interaction records
        verdict_log (list[dict]): Stores judge verdicts
        end_reason (str): Reason why the interaction ended
    """

    def __init__(
        self,
        attacker_type: str,
        attacker_agent: BaseAgent,
        tutor_agent: BaseAgent,
        judge_agent: Optional[BaseAgent] = None,
        reflector_agent: Optional[BaseAgent] = None,
        student_reflection: bool = False,
        tutor_reflection: bool = False,
        random_order: bool = True,
        problem: str = "",
        answer: str = "",
        answer_text: str = "",
        max_turns: int = 20,
        experiment_logger=None,
        random_object: random.Random = None,
        dataset_name: str = "gsm8k",
    ):
        """
        Initialize the Unified Playground.

        Args:
            attacker_type: Type of attacker ("student" or "prompt_generator")
            attacker_agent: Instance of the attacker agent
            tutor_agent: Instance of the tutor agent to evaluate
            judge_agent: Optional judge agent for evaluation
            reflector_agent: Optional reflector agent for refinement
            student_reflection: Whether to use reflection for student
            tutor_reflection: Whether to use reflection for tutor
            random_order: Whether to use random order for prompt generation (prompt_generator only)
            problem: Math problem text
            answer: Correct answer (numeric for gsm8k, option letter for mmlu)
            answer_text: Full text of the correct answer option (mmlu only)
            max_turns: Maximum number of interaction rounds
            experiment_logger: Optional ExperimentLogger instance
            dataset_name: Name of the dataset being used
        """
        self.attacker_type = attacker_type
        self.attacker = attacker_agent
        self.tutor = tutor_agent
        self.judge = judge_agent
        self.reflector = reflector_agent
        self.student_reflection = student_reflection
        self.tutor_reflection = tutor_reflection
        self.problem = problem
        self.answer = answer
        self.answer_text = answer_text
        self.max_turns = max_turns
        self.random_order = random_order
        self.dataset_name = dataset_name
        
        # Get main logger from experiment_logger
        if experiment_logger is None:
            experiment_logger = setup_experiment_logger()
        self.logger = experiment_logger.get_logger("main")

        self.conversation_log = []
        self.verdict_log = {}
        self.reflect_log = {}
        self.end_reason = None
        
        self.random_object = random_object

    def run(self, init_message: str = "Hi teacher, can we work on this problem?"):
        """
        Execute the evaluation experiment.

        Runs the appropriate interaction mode based on attacker_type.
        """
        self.logger.info("=== Unified Playground Started ===")
        self.logger.info(f"🎭 Attacker Type: {self.attacker_type}")
        self.logger.info(f"🧩 Problem: {self.problem}")
        self.logger.info(f"🎯 Answer: {self.answer}")
        self.logger.info(f"🔁 Max Turns: {self.max_turns}")

        if self.attacker_type == "prompt_generator" or self.attacker_type == "llm_based_questions" or self.attacker_type == "manual_prompts":
            self._run_prompt_generator_mode()
        elif self.attacker_type == "student" or self.attacker_type == "base_student_adversarial_agent" or self.attacker_type == "student_with_reasoning" or self.attacker_type == "multi_agent_student":
            self._run_student_mode(init_message)
        else:
            raise ValueError(f"Unknown attacker_type: {self.attacker_type}")

        self.logger.info("=== Unified Playground Finished ===")

    def _run_prompt_generator_mode(self):
        """Run prompt generator attack mode."""
        self.logger.info("🤖 Running Prompt Generator Attack Mode")

        # Generate all prompts once at the beginning
        try:
            if self.attacker.using_existing_prompts:
                prompts_list = self.attacker.load_existing_prompts()
            else:
                prompts_list = self.attacker.reply(self.attacker.user_input)
            if self.random_order:
                original_order = [p["strategy"] for p in prompts_list]
                self.logger.info(f"🔹 Original order: {original_order}")
                self.random_object.shuffle(prompts_list)
                shuffled_order = [p["strategy"] for p in prompts_list]
                self.logger.info(f"🔸 Shuffled order: {shuffled_order}")

            self.logger.info(f"✨ Generated {len(prompts_list)} attack prompts")
        except Exception as e:
            self.logger.error(f"❌ Failed to generate prompts: {e}")
            self.end_reason = "generation_error"
            return

        # Use min to handle cases where we have fewer prompts than max_turns
        num_turns = min(len(prompts_list), self.max_turns)

        for turn in range(1, num_turns + 1):
            self.logger.info(f"\n===== ROUND {turn} =====")

            # Use one prompt per turn
            prompt = prompts_list[turn - 1]
            prompt_content = prompt.get("content", prompt)

            if self.attacker.using_existing_prompts:
                while self.contains_answer(prompt_content):
                    prompt = self.random_object.choice(prompts_list)
                    prompt_content = prompt.get("content", "")
            
            else:
                if self.judge is not None and self.contains_answer(prompt_content):
                    verdict_after_student_raw = self.judge.reply(user_input=prompt_content, judge_role="to_student")
                    try:
                        verdict_after_student = json.loads(verdict_after_student_raw)
                    except json.JSONDecodeError:
                        self.logger.warning(f"⚠️ Failed to parse judge output as JSON: {verdict_after_student_raw}")
                        verdict_after_student = {"student_leakage": False, "reason": "Invalid JSON from Judge"}
                    self.judge.messages_history[0]["turn"] = turn
                    self.verdict_log[turn] = (
                        self.judge.messages_history
                    )
                    self.logger.info(
                        f"🔎 Judge(after_student):\n{json.dumps(verdict_after_student, indent=2, ensure_ascii=False)}"
                    )

                    if verdict_after_student.get("student_correct", False):
                        self.tutor.conversation.add_user_message(prompt_content)
                        self.conversation_log.append(
                            {"turn": turn, "student": prompt_content, "tutor": ""}
                        )
                        self.logger.info(f"👩‍🎓 Student: {prompt_content}")
                        self.logger.info("✅ Student independently derived the correct answer. Stopping.")
                        self.end_reason = "student_correct"
                        return
            
            # Tutor replies
            tutor_reply = self.tutor.reply(prompt_content)
            self.logger.info(f"🧑‍🏫 Tutor: {tutor_reply}")
            self.conversation_log.append(
                {"turn": turn, "attacker": prompt, "tutor": tutor_reply}
            )

            try:
                parsed = json.loads(tutor_reply)
                tutor_reply_text = parsed.get("content", tutor_reply)
            except Exception:
                tutor_reply_text = tutor_reply

            if "end_of_conversation" in tutor_reply_text and not self.contains_answer(tutor_reply_text) and self.tutor.model_name == "eth-nlped/TutorRL-7B":
                self.logger.warning("🤚 Tutor finished the conversation. Stopping.")
                self.end_reason = "tutor_finish"
                return
            
            # Judge evaluation after tutor
            if self.judge is not None and self.contains_answer(tutor_reply_text):
                verdict_after_tutor_raw = self.judge.reply(user_input=tutor_reply_text, judge_role="to_tutor")
                try:
                    verdict_after_tutor = json.loads(verdict_after_tutor_raw)
                except json.JSONDecodeError:
                    self.logger.warning(f"⚠️ Failed to parse judge output as JSON: {verdict_after_tutor_raw}")
                    verdict_after_tutor = {"tutor_leakage": False, "reason": "Invalid JSON from Judge"}
                self.judge.messages_history[0]["turn"] = turn
                self.verdict_log[turn] = (
                    self.judge.messages_history
                )
                self.logger.info(
                    f"🔎 Judge(after_tutor):\n{json.dumps(verdict_after_tutor, indent=2, ensure_ascii=False)}"
                )

                if verdict_after_tutor.get("tutor_revealed", False):
                    if self.reflector and self.tutor_reflection:
                        self.logger.info("💡 Reflector reviewing tutor leakage...")
                        reflection = self.reflector.reply(user_input=tutor_reply_text, reflection_role="to_tutor")

                        try:
                            parsed = json.loads(reflection)
                            reflection_text = parsed.get("revised_response", reflection)
                        except Exception:
                            reflection_text = reflection

                        self.conversation_log[-1]["tutor"] = reflection_text
                        self.tutor.conversation.delete_message()
                        self.tutor.conversation.add_assistant_message(reflection_text)

                        self.reflector.messages_history[0]["turn"] = turn
                        self.reflect_log[turn] = (
                            self.reflector.messages_history
                        )
                        self.logger.info(f"🪞 Reflector: {reflection}")

                        # Re-judge after reflection
                        if self.contains_answer(reflection_text):
                            verdict_after_reflection_raw = self.judge.reply(user_input=reflection_text, judge_role="to_tutor")

                            try:
                                verdict_after_reflection = json.loads(verdict_after_reflection_raw)
                            except json.JSONDecodeError:
                                self.logger.warning(f"⚠️ Failed to parse judge output as JSON: {verdict_after_reflection_raw}")
                                verdict_after_reflection = {"tutor_leakage": False, "reason": "Invalid JSON from Judge"}

                            self.verdict_log[turn].append(self.judge.messages_history)
                            self.logger.info(
                                f"🔎 Judge(after_reflection):\n{json.dumps(verdict_after_reflection, indent=2, ensure_ascii=False)}"
                            )

                            if verdict_after_reflection.get("tutor_revealed", False):
                                self.attacker.conversation.add_user_message(reflection_text)
                                self.logger.warning("⛔ Tutor leakage detected after reflection. Stopping.")
                                self.end_reason = "tutor_leakage"
                                return
                        else:
                            tutor_reply = reflection_text
                    else:
                        self.logger.warning("⛔ Tutor leaked the answer. Stopping.")
                        self.end_reason = "tutor_leakage"
                        return
                
                if "end_of_conversation" in tutor_reply_text and self.tutor.model_name == "eth-nlped/TutorRL-7B":
                    self.logger.warning("🤚 Tutor finished the conversation. Stopping.")
                    self.end_reason = "tutor_finish"
                    return
                

        if self.end_reason is None:
            self.end_reason = "timeout"

    def _run_student_mode(self, init_message: str):
        """Run student-tutor conversation mode."""
        self.logger.info("👩‍🎓 Running Student-Tutor Conversation Mode")

        # Student starts
        if self.attacker.response_format is None:
            init_message = PromptLoader.load_prompt(
                prompt_file=self.attacker.prompt_file,
            )
            student_reply = self.attacker.reply_first(init_message)
        else:
            student_reply = init_message
        for turn in range(1, self.max_turns + 1):
            self.logger.info(f"\n===== ROUND {turn} =====")
            self.logger.info(f"👩‍🎓 Student: {student_reply}")

            # Tutor replies
            tutor_reply = self.tutor.reply(student_reply)
            self.logger.info(f"🧑‍🏫 Tutor: {tutor_reply}")
            self.conversation_log.append(
                {"turn": turn, "student": student_reply, "tutor": tutor_reply}
            )

            try:
                parsed = json.loads(tutor_reply)
                tutor_reply_text = parsed.get("content", tutor_reply)
            except Exception:
                tutor_reply_text = tutor_reply

            if "end_of_conversation" in tutor_reply_text and not self.contains_answer(tutor_reply_text) and self.tutor.model_name == "eth-nlped/TutorRL-7B":
                self.logger.warning("🤚 Tutor finished the conversation. Stopping.")
                self.end_reason = "tutor_finish"
                return

            # Judge after Tutor (check leakage)
            if self.judge is not None and self.contains_answer(tutor_reply_text):

                verdict_after_tutor_raw = self.judge.reply(user_input=tutor_reply_text, judge_role="to_tutor")
                try:
                    verdict_after_tutor = json.loads(verdict_after_tutor_raw)
                except json.JSONDecodeError:
                    self.logger.warning(f"⚠️ Failed to parse judge output as JSON: {verdict_after_tutor_raw}")
                    verdict_after_tutor = {"tutor_leakage": False, "reason": "Invalid JSON from Judge"}
                self.verdict_log[turn] = (
                    self.judge.messages_history
                )
                self.logger.info(
                    f"🔎 Judge(after_tutor):\n{json.dumps(verdict_after_tutor, indent=2, ensure_ascii=False)}"
                )

                if verdict_after_tutor.get("tutor_revealed", False):
                    if self.reflector and self.tutor_reflection:
                        self.logger.info("💡 Reflector reviewing tutor leakage...")
                        reflection = self.reflector.reply(user_input=tutor_reply_text, reflection_role="to_tutor")

                        try:
                            parsed = json.loads(reflection)
                            reflection_text = parsed.get("revised_response", reflection)
                        except Exception:
                            reflection_text = reflection

                        self.conversation_log[-1]["tutor"] = reflection_text
                        self.tutor.conversation.delete_message()
                        self.tutor.conversation.add_assistant_message(reflection_text)

                        self.reflector.messages_history[0]["turn"] = turn
                        self.reflect_log[turn] = (
                            self.reflector.messages_history
                        )
                        self.logger.info(f"🪞 Reflector: {reflection}")

                        # Re-judge after reflection
                        if self.contains_answer(reflection_text):
                            verdict_after_reflection_raw = self.judge.reply(user_input=reflection_text, judge_role="to_tutor")
                            try:
                                verdict_after_reflection = json.loads(verdict_after_reflection_raw)
                            except json.JSONDecodeError:
                                self.logger.warning(f"⚠️ Failed to parse judge output as JSON: {verdict_after_reflection_raw}")
                                verdict_after_reflection = {"tutor_leakage": False, "reason": "Invalid JSON from Judge"}
                            self.verdict_log[turn].append(self.judge.messages_history)
                            self.logger.info(
                                f"🔎 Judge(after_reflection):\n{json.dumps(verdict_after_reflection, indent=2, ensure_ascii=False)}"
                            )

                            if verdict_after_reflection.get("tutor_revealed", False):
                                self.attacker.conversation.add_user_message(reflection_text)
                                self.logger.warning("⛔ Tutor leakage detected after reflection. Stopping.")
                                self.end_reason = "tutor_leakage"
                                return
                        else:
                            tutor_reply = reflection_text
                    else:
                        self.attacker.conversation.add_user_message(tutor_reply_text)
                        self.logger.warning("⛔ Tutor leakage detected. Stopping.")
                        self.end_reason = "tutor_leakage"
                        return
                
                if "end_of_conversation" in tutor_reply_text and self.tutor.model_name == "eth-nlped/TutorRL-7B":
                    self.logger.warning("🤚 Tutor finished the conversation. Stopping.")
                    self.end_reason = "tutor_finish"
                    return

            # Student replies
            student_reply = self.attacker.reply(tutor_reply)

            try:
                parsed = json.loads(student_reply)
                student_reply_text = parsed.get("content", student_reply)
            except Exception:
                student_reply_text = student_reply

            # Judge after Student (check correctness)
            if self.judge is not None and self.contains_answer(student_reply_text):
                # Handle potential JSON in student reply
                verdict_after_student_raw = self.judge.reply(user_input=student_reply_text, judge_role="to_student")
                try:
                    verdict_after_student = json.loads(verdict_after_student_raw)
                except json.JSONDecodeError:
                    self.logger.warning(f"⚠️ Failed to parse judge output as JSON: {verdict_after_student_raw}")
                    verdict_after_student = {"student_correct": False, "reason": "Invalid JSON from Judge"}
                self.verdict_log[turn] = (
                    self.judge.messages_history
                )
                self.logger.info(
                    f"🔎 Judge(after_student):\n{json.dumps(verdict_after_student, indent=2, ensure_ascii=False)}"
                )

                if verdict_after_student.get("student_correct", False):
                    if self.reflector and self.student_reflection:
                        self.logger.info("💡 Reflector reviewing student answer correctness...")
                        reflection = self.reflector.reply(user_input=student_reply_text, reflection_role="to_student")

                        try:
                            parsed = json.loads(reflection)
                            reflection_text = parsed.get("revised_response", reflection)
                        except Exception:
                            reflection_text = reflection

                        self.attacker.conversation.delete_message()
                        try:
                            parsed = json.loads(student_reply)
                        except Exception:
                            student_reply_text = student_reply
                        parsed["content"] = reflection_text
                        self.conversation_log[-1]["student"] = json.dumps(parsed)
                        self.attacker.conversation.add_assistant_message(json.dumps(parsed))

                        self.reflector.messages_history[0]["turn"] = turn
                        self.reflect_log[turn] = (
                            self.reflector.messages_history
                        )
                        self.logger.info(f"🪞 Reflector: {reflection}")

                        # Re-judge after reflection
                        if self.contains_answer(reflection_text):
                            verdict_after_reflection_raw = self.judge.reply(user_input=reflection_text, judge_role="to_student")
                            try:
                                verdict_after_reflection = json.loads(verdict_after_reflection_raw)
                            except json.JSONDecodeError:
                                self.logger.warning(f"⚠️ Failed to parse judge output as JSON: {verdict_after_reflection_raw}")
                                verdict_after_reflection = {"student_correct": False, "reason": "Invalid JSON from Judge"}

                            self.verdict_log[turn].append(self.judge.messages_history)
                            self.logger.info(
                                f"🔎 Judge(after_reflection):\n{json.dumps(verdict_after_reflection, indent=2, ensure_ascii=False)}"
                            )

                            if verdict_after_reflection.get("student_correct", False):
                                self.tutor.conversation.add_user_message(reflection_text)
                                self.conversation_log.append(
                                    {"turn": turn, "student": reflection, "tutor": ""}
                                )
                                self.logger.info("✅ Student independently derived the correct answer after reflection. Stopping.")
                                self.end_reason = "student_correct"
                                return
                        else:
                            student_reply = reflection_text
                    else:
                        # Log the student's final correct answer before stopping
                        self.tutor.conversation.add_user_message(student_reply_text)
                        self.conversation_log.append(
                            {"turn": turn, "student": student_reply, "tutor": ""}
                        )
                        self.logger.info(f"👩‍🎓 Student: {student_reply}")
                        self.logger.info("✅ Student independently derived the correct answer. Stopping.")
                        self.end_reason = "student_correct"
                        return

        if self.end_reason is None:
            self.end_reason = "timeout"

    def get_log(self):
        """Return the full interaction log."""
        return self.conversation_log

    def get_verdicts(self):
        """Return all judge verdicts."""
        return self.verdict_log
    
    def get_reflect_log(self):
        return self.reflect_log

    def get_end_reason(self):
        """Return the reason why the interaction ended."""
        return self.end_reason

    def save_log(self, filepath: str):
        """
        Save the interaction log to a JSON file for later analysis.

        Args:
            filepath: Path to save the JSON log
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "attacker_type": self.attacker_type,
                        "problem": self.problem,
                        "answer": self.answer,
                        "end_reason": self.end_reason,
                        "log": self.conversation_log,
                        "verdicts": self.verdict_log,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            self.logger.info(f"🧾 Log saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save log: {e}")

    def contains_answer(self, text: str) -> bool:
        """
        Check if the given text contains the target answer.

        For gsm8k: checks for numeric answer match.
        For mmlu: checks for answer option letter or answer text (case-insensitive).

        Args:
            text (str): Text to check.

        Returns:
            bool: True if target answer is found, False otherwise.
        """
        if not self.answer:
            return False

        if self.dataset_name == "mmlu":
            return self._contains_answer_mmlu(text)
        else:
            return self._contains_answer_gsm8k(text)

    def _contains_answer_gsm8k(self, text: str) -> bool:
        """Check if text contains the target numeric answer (gsm8k)."""
        text = text.replace(",", "")
        answer = str(self.answer).replace(",", "")

        try:
            target = float(answer)
        except ValueError:
            return False

        numbers = [float(num) for num in re.findall(r"\d+(?:\.\d+)?", text)]
        return target in numbers

    def _contains_answer_mmlu(self, text: str) -> bool:
        """Check if text contains the answer option letter or answer text (mmlu)."""
        answer_letter = str(self.answer).strip()

        # Check if text contains the answer option letter as a standalone token
        # Match patterns like "A", "(A)", "A.", "A)", "A:" etc.
        if answer_letter and re.search(
            r'(?<![a-zA-Z])' + re.escape(answer_letter) + r'(?![a-zA-Z])', text
        ):
            return True

        # Check if text contains the answer text (case-insensitive)
        if self.answer_text and self.answer_text.strip():
            if self.answer_text.strip().lower() in text.lower():
                return True

        return False

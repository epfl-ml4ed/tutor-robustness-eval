

"""
Parallel evaluation framework for tutor agent robustness.

This script evaluates tutor agents across multiple problems concurrently,
using vLLM for efficient inference with proper rate limiting and error handling.
"""
import os
import openai
import numpy as np
import asyncio
import json
import random
import re
import datetime
from typing import List, Dict, Any, Tuple, Optional, Literal, Union
from dataclasses import dataclass
from pathlib import Path
import argparse
from datasets import load_dataset
from tqdm.asyncio import tqdm as atqdm
from dotenv import load_dotenv

load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import rootutils
import random

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.agents.judge import Judge
from src.agents.student import AutoStudent
from src.agents.prompt_generator import PromptGenerator
from src.agents.tutor import Tutor
from src.agents.reflector import Reflector
from src.playground.unified_conversation import UnifiedPlayground
from src.utils.logger import setup_experiment_logger

@dataclass
class MathProblem:
    """Structure for math problem"""
    problem_id: int
    problem_text: str
    answer: str
    difficulty: str

@dataclass
class MMLUProblem:
    """Structure for MMLU problem"""
    problem_id: int
    problem_text: str
    answer: str
    answer_text: str
    domain: str


@dataclass
class EvaluationResult:
    """Structure for evaluation result"""
    problem_id: int
    problem: str
    answer: str
    outcome_class: int
    outcome_name: str
    turns: int
    verdicts: List[Dict]
    conversation_log: List[Dict]
    tutor_log: List[Dict]
    attacker_log: List[Dict]
    # judge_log: List[Dict]
    experiment_name: str
    difficulty: str
    tutor_model: str
    attacker_model: str
    attack_type: str
    tutor_type: str
    strategies: Optional[List]
    reflector_log: Optional[List[Dict]] = None


class ParallelEvaluator:
    """Parallel evaluator for tutor agents using vLLM."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize parallel evaluator.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.results = []
        
    def load_dataset(self) -> Union[List[MathProblem], List[MMLUProblem]]:
        """
        Load dataset based on configuration.

        Returns:
            List of MathProblem or MMLUProblem instances
        """
        if self.cfg.dataset.name == "gsm8k" and not self.cfg.dataset.path:
            dataset = load_dataset("gsm8k", "main")[self.cfg.dataset.split]
            samples = dataset.select(range(self.cfg.dataset.start_idx, self.cfg.dataset.end_idx))
            problems = []
            for i, sample in enumerate(samples):
                answer = sample["answer"].split("####")[-1].strip()
                problems.append(MathProblem(
                    problem_id=i,
                    problem_text=sample["question"],
                    answer=answer
                ))
            return problems
        elif self.cfg.dataset.path and self.cfg.dataset.name == "gsm8k":
            with open(self.cfg.dataset.path, 'r') as f:
                samples = json.load(f)
            problems = []
            for i, sample in enumerate(samples[self.cfg.dataset.start_idx:self.cfg.dataset.end_idx]):
                problems.append(MathProblem(
                    problem_id=i,
                    problem_text=sample.get("problem", sample.get("question", "")),
                    answer=sample.get("answer", ""),
                    difficulty=sample.get("difficulty", ""),
                ))
            return problems
        elif self.cfg.dataset.path and self.cfg.dataset.name == "mmlu":
            with open(self.cfg.dataset.path, 'r') as f:
                samples = json.load(f)
            problems = []
            for i, sample in enumerate(samples[self.cfg.dataset.start_idx:self.cfg.dataset.end_idx]):
                answer_text = ""
                options = sample.get("options", [])
                answer_val = sample.get("answer", "")
                if options and len(answer_val) == 1 and answer_val.isalpha():
                    idx = ord(answer_val.upper()) - ord('A')
                    if 0 <= idx < len(options):
                        answer_text = options[idx]
                # Combine question and options into problem_text
                question = sample.get("problem", sample.get("question", ""))
                if options:
                    options_str = "\n".join(
                        f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)
                    )
                    problem_text = f"{question}\n\n{options_str}"
                else:
                    problem_text = question

                problems.append(MMLUProblem(
                    problem_id=i,
                    problem_text=problem_text,
                    answer=answer_val,
                    answer_text=answer_text,
                    domain=sample.get("category", sample.get("domain", "")),
                ))
            return problems
        else:
            raise ValueError("Either dataset.name='gsm8k' or dataset.path must be specified")
    
    def create_attacker_agent(self, problem: str, answer: str, exp_logger, dataset_name: str = None, domain: str = None):
        """Create attacker agent based on configuration."""
        if self.cfg.attacker.type == "student" or self.cfg.attacker.type == "base_student_adversarial_agent" or self.cfg.attacker.type == "student_with_reasoning" or self.cfg.attacker.type == "multi_agent_student":
            return AutoStudent(
                model_name=self.cfg.agent.student.model_name,
                device=self.cfg.agent.student.get("device"),
                prompt_file=str(root / self.cfg.agent.student.prompt_file),
                temperature=self.cfg.agent.student.temperature,
                max_new_tokens=self.cfg.agent.student.max_new_tokens,
                response_format=self.cfg.agent.student.response_format,
                problem=problem,
                answer=answer,
                experiment_logger=exp_logger,
                use_vllm=self.cfg.agent.student.get("use_vllm", False),
                vllm_base_url=self.cfg.agent.student.get("base_url"),
                seed=self.cfg.evaluation.get("seed", None),
            )
        elif self.cfg.attacker.type == "prompt_generator" or self.cfg.attacker.type == "llm_based_questions" or self.cfg.attacker.type == "manual_prompts":
            return PromptGenerator(
                model_name=self.cfg.agent.prompt_generator.model_name,
                device=self.cfg.agent.prompt_generator.get("device"),
                prompt_file=str(root / self.cfg.agent.prompt_generator.prompt_file),
                prompt_number=self.cfg.agent.prompt_generator.prompt_number,
                temperature=self.cfg.agent.prompt_generator.temperature,
                max_new_tokens=self.cfg.agent.prompt_generator.max_new_tokens,
                experiment_logger=exp_logger,
                answer=answer,
                problem=problem,
                use_vllm=self.cfg.agent.prompt_generator.get("use_vllm", False),
                vllm_base_url=self.cfg.agent.prompt_generator.get("base_url"),
                using_existing_prompts=True if self.cfg.attacker.type == "manual_prompts" else False,
                existing_prompts_path=self.cfg.agent.prompt_generator.get("existing_prompts_path", None)
            )
        else:
            raise ValueError(f"Unknown attacker type: {self.cfg.attacker.type}")
    
    def get_outcome_from_end_reason(self, end_reason: str) -> Tuple[int, str]:
        """Determine outcome class and name from end reason."""
        if end_reason == "tutor_leakage":
            return 0, "tutor_leakage"
        elif end_reason == "student_correct":
            return 1, "student_correct"
        # elif end_reason == "all_prompts_contain_answer":
        #     return 1, "all_prompts_contain_answer"
        elif end_reason == "timeout":
            return 2, "timeout"
        elif self.cfg.agent.tutor.model_name == "eth-nlped/TutorRL-7B" and end_reason == "tutor_finish":
            return 3, "tutor_finish"
        # else:
        #     return 3, "abnormal_end"
    
    async def evaluate_problem(
        self, 
        problem: Union[MathProblem, MMLUProblem],
        semaphore: asyncio.Semaphore,
        seed_id: int
    ) -> EvaluationResult:
        """
        Evaluate a single problem asynchronously.
        
        Args:
            problem: Problem to evaluate
            semaphore: Semaphore for rate limiting
            
        Returns:
            EvaluationResult instance
        """
        async with semaphore:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._evaluate_problem_sync, 
                problem,
                seed_id
            )
            return result
    
    def _evaluate_problem_sync(self, problem: Union[MathProblem, MMLUProblem], seed_id: int) -> EvaluationResult:
        """
        Synchronous evaluation of a single problem with retry logic.

        Args:
            problem: Problem to evaluate

        Returns:
            EvaluationResult instance
        """
        
        # Get max retries from config, default to 3
        max_retries = self.cfg.evaluation.get("max_retries", 5)

        # Create logger for this specific problem
        exp_logger = setup_experiment_logger(
            experiment_name=self.cfg.wandb.name,
            problem_id=problem.problem_id,
            base_log_dir=f"{root}/logs/{self.cfg.wandb.project}",
        )

        main_logger = exp_logger.get_logger("main")
        main_logger.info(f"\n================= 🧮 Problem {problem.problem_id+1} =================")
        main_logger.info(f"❓ Question: {problem.problem_text}")
        main_logger.info(f"✅ Ground truth answer: {problem.answer}")

        # Retry loop
        last_exception = None
        attacker = None  # Initialize to avoid UnboundLocalError
        for attempt in range(max_retries):
            #Make seed
            cfg_seed = self.cfg.evaluation.get("seed", None)
            random_object = random.Random()
            random_object.seed(cfg_seed if cfg_seed is not None else seed_id)
            
            if attempt > 0:
                main_logger.info(f"🔄 Retry attempt {attempt + 1}/{max_retries}")

            try:
                # Create fresh agents for each experiment
                attacker = self.create_attacker_agent(
                    problem.problem_text,
                    problem.answer,
                    exp_logger
                )
                print(attacker)

                tutor = Tutor(
                    model_name=self.cfg.agent.tutor.model_name,
                    device=self.cfg.agent.tutor.get("device"),
                    temperature=self.cfg.agent.tutor.temperature,
                    max_new_tokens=self.cfg.agent.tutor.max_new_tokens,
                    prompt_file=str(root / self.cfg.agent.tutor.prompt_file),
                    response_format=self.cfg.agent.tutor.get("response_format"),
                    problem=problem.problem_text,
                    answer=problem.answer,
                    experiment_logger=exp_logger,
                    use_vllm=self.cfg.agent.tutor.get("use_vllm", False),
                    vllm_base_url=self.cfg.agent.tutor.get("base_url"),
                    dataset_name=self.cfg.dataset.get("name", None),
                    domain=getattr(problem, 'domain', None),
                    answer_text=getattr(problem, 'answer_text', None),
                    seed=self.cfg.evaluation.get("seed", None),
                )

                judge = Judge(
                    model_name=self.cfg.agent.judge.model_name,
                    device=self.cfg.agent.judge.get("device"),
                    temperature=self.cfg.agent.judge.temperature,
                    max_new_tokens=self.cfg.agent.judge.max_new_tokens,
                    to_tutor_prompt_file=str(root / self.cfg.agent.judge.to_tutor_prompt_file),
                    to_student_prompt_file=str(root / self.cfg.agent.judge.to_student_prompt_file),
                    problem=problem.problem_text,
                    answer=problem.answer,
                    experiment_logger=exp_logger,
                    use_vllm=self.cfg.agent.judge.get("use_vllm", False),
                    vllm_base_url=self.cfg.agent.judge.get("base_url"),
                    dataset_name=self.cfg.dataset.get("name", None),
                    domain=getattr(problem, 'domain', None),
                    answer_text=getattr(problem, 'answer_text', None),
                    seed=self.cfg.evaluation.get("seed", None),
                )

                if self.cfg.playground.student_reflection or self.cfg.playground.tutor_reflection:
                    reflector = Reflector(
                        student_model_name=self.cfg.agent.reflector.student_model_name,
                        student_use_vllm=self.cfg.agent.reflector.get("student_use_vllm", False),
                        student_vllm_base_url=self.cfg.agent.reflector.get("student_base_url"),
                        tutor_model_name=self.cfg.agent.reflector.tutor_model_name,
                        tutor_use_vllm=self.cfg.agent.reflector.get("tutor_use_vllm", False),
                        tutor_vllm_base_url=self.cfg.agent.reflector.get("tutor_base_url"),
                        device=self.cfg.agent.reflector.get("device"),
                        temperature=self.cfg.agent.reflector.temperature,
                        max_new_tokens=self.cfg.agent.reflector.max_new_tokens,
                        to_tutor_prompt_file=str(root / self.cfg.agent.reflector.to_tutor_prompt_file),
                        to_student_prompt_file=str(root / self.cfg.agent.reflector.to_student_prompt_file),
                        problem=problem.problem_text,
                        answer=problem.answer,
                        experiment_logger=exp_logger,
                        dataset_name=self.cfg.dataset.get("name", None),
                        domain=getattr(problem, 'domain', None),
                        answer_text=getattr(problem, 'answer_text', None),
                        seed=self.cfg.evaluation.get("seed", None),
                    )
                else: 
                    reflector = None

                # Run Unified Playground
                playground = UnifiedPlayground(
                    attacker_type=self.cfg.attacker.type,
                    attacker_agent=attacker,
                    tutor_agent=tutor,
                    judge_agent=judge,
                    reflector_agent=reflector,
                    student_reflection=self.cfg.playground.student_reflection,
                    tutor_reflection=self.cfg.playground.tutor_reflection,
                    random_order=self.cfg.attacker.get("random_order", True),
                    problem=problem.problem_text,
                    answer=problem.answer,
                    answer_text=getattr(problem, 'answer_text', ''),
                    max_turns=self.cfg.evaluation.max_turns,
                    experiment_logger=exp_logger,
                    random_object=random_object,
                    dataset_name=self.cfg.dataset.get("name", "gsm8k"),
                )
                playground.run()

                # Collect result summary
                verdicts = playground.get_verdicts()
                reflect_log = playground.get_reflect_log()
                main_logger.info(f"📝 Verdicts: {verdicts}")
                end_reason = playground.get_end_reason()
                turns = len(playground.get_log())

                # Determine outcome class and name
                outcome_class, outcome_name = self.get_outcome_from_end_reason(end_reason)
                main_logger.info(f"🎯 Outcome: {outcome_name}")

                # Success! Return the result
                return EvaluationResult(
                    problem_id=problem.problem_id,
                    problem=problem.problem_text,
                    answer=problem.answer,
                    outcome_class=outcome_class,
                    outcome_name=outcome_name,
                    turns=turns,
                    verdicts=verdicts,
                    conversation_log=playground.conversation_log,
                    tutor_log=tutor.messages_history,
                    attacker_log=attacker.messages_history,
                    # judge_log=judge.messages_history,
                    experiment_name=self.cfg.wandb.name,
                    difficulty=getattr(problem, 'difficulty', getattr(problem, 'domain', '')),
                    tutor_model=self.cfg.agent.tutor.model_name,
                    attacker_model=attacker.model_name if self.cfg.attacker.type != "manual_prompts" else "manual_prompts",
                    attack_type=self.cfg.attacker.type,
                    tutor_type=self.cfg.tutor.type,
                    strategies=OmegaConf.to_container(self.cfg.evaluation.strategies, resolve=True) if self.cfg.evaluation.strategies else None,
                    reflector_log=reflect_log if reflector else None,
                )

            except Exception as e:
                last_exception = e
                main_logger.error(f"❌ Error evaluating problem {problem.problem_id} (attempt {attempt + 1}/{max_retries}): {e}")
                import traceback
                main_logger.error(traceback.format_exc())

                # If this was the last attempt, fall through to return error result
                if attempt < max_retries - 1:
                    main_logger.info(f"⏳ Retrying after error...")
                    continue

        # All retries exhausted, return error result
        main_logger.error(f"❌ Failed after {max_retries} attempts. Last error: {last_exception}")
        return EvaluationResult(
            problem_id=problem.problem_id,
            problem=problem.problem_text,
            answer=problem.answer,
            outcome_class=3,
            outcome_name="error",
            turns=0,
            verdicts=[],
            conversation_log=[],
            tutor_log=[],
            attacker_log=[],
            # judge_log=[],
            experiment_name=self.cfg.wandb.name,
            difficulty=getattr(problem, 'difficulty', getattr(problem, 'domain', '')),
            tutor_model=self.cfg.agent.tutor.model_name,
            attacker_model=attacker.model_name if attacker and self.cfg.attacker.type != "manual_prompts" else "manual_prompts" if self.cfg.attacker.type == "manual_prompts" else "unknown",
            attack_type=self.cfg.attacker.type,
            tutor_type=self.cfg.tutor.type,
            strategies=OmegaConf.to_container(self.cfg.evaluation.strategies, resolve=True) if self.cfg.evaluation.strategies else None,
            reflector_log=None,
        )
    
    async def evaluate_problems_batch(
        self, 
        problems: Union[List[MathProblem], List[MMLUProblem]],
        max_concurrent: int,
        batch_id: int
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of problems concurrently.
        
        Args:
            problems: List of problems to evaluate
            max_concurrent: Maximum number of concurrent evaluations
            
        Returns:
            List of EvaluationResult instances
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all problems in the batch
        tasks = [
            self.evaluate_problem(problems[i], semaphore, batch_id + i) 
            for i in range(len(problems))
        ]
        
        # Execute all tasks concurrently with progress bar
        results = []
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating problems"):
            result = await coro
            results.append(result)
        
        # Sort results by problem_id to maintain order
        results.sort(key=lambda x: x.problem_id)
        
        return results
    
    async def evaluate_dataset(self) -> Dict[str, Any]:
        """
        Evaluate entire dataset with batched parallel processing.
        
        Returns:
            Dictionary with evaluation statistics and results
        """
        problems = self.load_dataset()
        print(f"📊 Loaded {len(problems)} problems")

        # Set global seed if specified
        cfg_seed = self.cfg.evaluation.get("seed", None)
        if cfg_seed is not None:
            random.seed(cfg_seed)
            np.random.seed(cfg_seed)
            print(f"🌱 Global seed set to {cfg_seed}")
        
        batch_size = self.cfg.evaluation.batch_size
        max_concurrent_problems = self.cfg.evaluation.max_concurrent_problems
        
        all_results = []
        
        # Initialize WandB if enabled
        if self.cfg.use_wandb:
            wandb.init(
                project=self.cfg.wandb.project,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                name=self.cfg.wandb.name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
            
            columns = [
                "id", "problem", "answer", "outcome_class", "outcome_name", 
                "turns", "judge_log", "tutor_log", "attacker_log", "conversation_log", "experiment_name", "difficulty", "tutor_model", "attacker_model", "attack_type", "tutor_type", "strategies"
            ]
            if self.cfg.playground.student_reflection or self.cfg.playground.tutor_reflection:
                columns.append("reflector_log")
            results_table = wandb.Table(columns=columns)
        
        # Initialize total counts
        total_counts = {
            "tutor_leakage": 0,
            "student_correct": 0,
            # "all_prompts_contain_answer": 0,
            "timeout": 0,
            "tutor_finish": 0,
            # "abnormal_end": 0,
            "error": 0,
            "turns": 0,
        }
        
        # Process problems in batches
        total_batches = (len(problems) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(problems), batch_size):
            batch_end = min(batch_idx + batch_size, len(problems))
            current_batch = problems[batch_idx:batch_end]
            current_batch_num = batch_idx // batch_size + 1
            
            print(f"\n{'='*60}")
            print(f"Processing Batch {current_batch_num}/{total_batches}")
            print(f"Problems {batch_idx + 1}-{batch_end} of {len(problems)}")
            print(f"{'='*60}")
            
            # Process current batch concurrently
            try:
                batch_results = await self.evaluate_problems_batch(
                    current_batch, max_concurrent_problems, batch_id = batch_idx
                )
                
                # Process results from this batch
                for result in batch_results:
                    all_results.append(result)
                    
                    # Update stats
                    total_counts[result.outcome_name] += 1
                    total_counts["turns"] += result.turns
                    
                    # Log to WandB
                    if self.cfg.use_wandb:
                        denom = len(all_results)
                        outcome_ratios = {k: v / denom for k, v in total_counts.items() if k != "turns"}
                        avg_turns = total_counts["turns"] / denom
                        
                        row = [
                            result.problem_id,
                            result.problem,
                            result.answer,
                            result.outcome_class,
                            result.outcome_name,
                            result.turns,
                            json.dumps(result.verdicts, ensure_ascii=False, indent=2),
                            json.dumps(result.tutor_log, ensure_ascii=False, indent=2),
                            json.dumps(result.attacker_log, ensure_ascii=False, indent=2),
                            json.dumps(result.conversation_log, ensure_ascii=False, indent=2),
                            result.experiment_name,
                            result.difficulty,
                            result.tutor_model,
                            result.attacker_model,
                            result.attack_type,
                            result.tutor_type,
                            result.strategies
                        ]
                        if self.cfg.playground.student_reflection or self.cfg.playground.tutor_reflection:
                            row.append(json.dumps(result.reflector_log, ensure_ascii=False, indent=2) if result.reflector_log else "")
                        results_table.add_data(*row)
                        
                        # Real-time logging
                        wandb.log(
                            {
                                "problem_index": result.problem_id,
                                "outcome_class": result.outcome_class,
                                "outcome_name": result.outcome_name,
                                "avg_turns_so_far": avg_turns,
                                **{f"{k}_rate_so_far": v for k, v in outcome_ratios.items()},
                            },
                            step=result.problem_id,
                        )
                
            except Exception as e:
                print(f"❌ Error processing batch {current_batch_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate final stats
        total_problems = len(all_results)
        final_stats = {
            k + "_rate": v / total_problems for k, v in total_counts.items() if k != "turns"
        }
        final_stats["avg_turns"] = total_counts["turns"] / total_problems if total_problems > 0 else 0
        
        if self.cfg.use_wandb:
            wandb.log(final_stats)
            wandb.log({"results_table": results_table})
            wandb.finish()
        
        return {
            "statistics": final_stats,
            "total_counts": total_counts,
            "results": all_results,
        }


async def async_main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # show config summary

    evaluator = ParallelEvaluator(cfg)
    results = await evaluator.evaluate_dataset()

    # Print final summary
    print("\n" + "=" * 60)
    print("✅ All experiments completed!")
    print("=" * 60)
    for k, v in results["statistics"].items():
        print(f"{k}: {v:.2%}" if "rate" in k else f"{k}: {v:.2f}")



@hydra.main(config_path=f"{root}/src/configs",
            config_name="parallel_evaluation",
            version_base=None)
def main(cfg: DictConfig):
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(async_main(cfg))   # run async code correctly


if __name__ == "__main__":
    import sys
    main()   # DO NOT call asyncio.run() here
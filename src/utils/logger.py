import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from rich.console import Console
from rich.theme import Theme

# Rich console for beautiful output
console = Console(theme=Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
}))


class AgentFilter(logging.Filter):
    """Filter that only allows records from a specific agent."""

    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, 'agent', None) == self.agent_name


class ExperimentLogger:
    """
    Thread-safe logger for tutor-student experiments using standard logging.

    Features:
    - Separate log files for each agent (tutor, student, judge)
    - Problem-specific folders
    - Thread-safe file handlers
    - Per-problem logger instances (no global state)
    """

    # Class variables to share timestamp across instances (thread-safe)
    _shared_timestamp: Optional[str] = None
    _shared_experiment_dirs: Dict = {}

    def __init__(
        self,
        experiment_name: str = "experiment",
        problem_id: Optional[int] = None,
        base_log_dir: str = "logs",
        use_timestamp_dir: bool = True,
    ):
        """
        Initialize the experiment logger.

        Args:
            experiment_name: Name of the experiment
            problem_id: ID of the current problem (for folder organization)
            base_log_dir: Base directory for all logs
            use_timestamp_dir: If True, create a timestamped experiment directory
        """
        self.experiment_name = experiment_name
        self.problem_id = problem_id
        self.base_log_dir = Path(base_log_dir)
        self.use_timestamp_dir = use_timestamp_dir

        # Use shared timestamp for all instances in the same run
        if ExperimentLogger._shared_timestamp is None:
            ExperimentLogger._shared_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.timestamp = ExperimentLogger._shared_timestamp

        # Create experiment folder structure
        # Structure: logs/{experiment_name}_{timestamp}/problem_xxxx/
        if use_timestamp_dir:
            # Use or create shared experiment directory for this experiment name
            dir_key = (str(self.base_log_dir), experiment_name)
            if dir_key not in ExperimentLogger._shared_experiment_dirs:
                ExperimentLogger._shared_experiment_dirs[dir_key] = (
                    self.base_log_dir / f"{experiment_name}_{self.timestamp}"
                )
            self.experiment_dir = ExperimentLogger._shared_experiment_dirs[dir_key]

            if problem_id is not None:
                self.log_folder = self.experiment_dir / f"problem_{problem_id:04d}"
            else:
                self.log_folder = self.experiment_dir
        else:
            # Legacy behavior for backward compatibility
            if problem_id is not None:
                self.log_folder = self.base_log_dir / f"problem_{problem_id:04d}"
            else:
                self.log_folder = self.base_log_dir / self.timestamp

        self.log_folder.mkdir(parents=True, exist_ok=True)

        # Create unique logger name for this problem
        # This ensures each problem gets its own logger instance
        logger_name = f"{experiment_name}.problem_{problem_id:04d}" if problem_id is not None else experiment_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Don't propagate to root logger

        # Store handlers for cleanup
        self.handlers = []

        # Track which agents have been set up
        self._setup_agents = set()

        # Track if error log handler has been set up (lazy initialization)
        self._error_handler_setup = False

    def setup_agent_logger(self, agent_name: str, lazy: bool = False):
        """
        Setup a separate log file for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., "tutor", "student", "judge", "main")
            lazy: If True, delay file creation until first write
        """
        if agent_name in self._setup_agents:
            return  # Already set up

        log_file = self.log_folder / f"{agent_name}.log"

        # Create file handler
        handler = logging.FileHandler(log_file, mode='a', delay=lazy)
        handler.setLevel(logging.DEBUG)

        # Simple formatter without timestamps
        formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        handler.setFormatter(formatter)

        # Add filter to only log messages for this agent
        handler.addFilter(AgentFilter(agent_name))

        # Add handler to logger
        self.logger.addHandler(handler)
        self.handlers.append(handler)
        self._setup_agents.add(agent_name)

        # Setup error logger when setting up the first agent (usually "main")
        # This ensures error.log is available for all agents
        if not self._error_handler_setup:
            self.setup_error_logger()

    def setup_error_logger(self):
        """
        Setup error log file that only captures ERROR and CRITICAL level messages.
        This is called when setting up the first agent logger.
        The file is only created when the first error is actually logged (delay=True).
        """
        if self._error_handler_setup:
            return  # Already set up

        error_log_file = self.log_folder / "error.log"

        # Create error file handler with delay=True - file only created on first error
        error_handler = logging.FileHandler(error_log_file, mode='a', delay=True)
        error_handler.setLevel(logging.ERROR)  # Only ERROR and CRITICAL levels

        # Formatter with more details for errors
        formatter = logging.Formatter('%(levelname)-8s | [%(name)s] %(message)s')
        error_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(error_handler)
        self.handlers.append(error_handler)
        self._error_handler_setup = True

    def setup_console_logger(self):
        """Setup console output without timestamps."""
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.handlers.append(console_handler)

    def get_logger(self, agent_name: str = "main") -> logging.LoggerAdapter:
        """
        Get a logger adapter for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            LoggerAdapter instance that adds agent context
        """
        # Ensure agent logger is set up
        if agent_name not in self._setup_agents:
            self.setup_agent_logger(agent_name, lazy=True)

        # Return adapter that adds 'agent' to extra
        return logging.LoggerAdapter(self.logger, {'agent': agent_name})

    def info(self, message: str, agent: str = "main"):
        """Log info message."""
        self.get_logger(agent).info(message)

    def warning(self, message: str, agent: str = "main"):
        """Log warning message."""
        self.get_logger(agent).warning(message)

    def error(self, message: str, agent: str = "main"):
        """Log error message."""
        self.get_logger(agent).error(message)

    def debug(self, message: str, agent: str = "main"):
        """Log debug message."""
        self.get_logger(agent).debug(message)

    def success(self, message: str, agent: str = "main"):
        """Log success message (mapped to info level)."""
        self.get_logger(agent).info(message)

    def cleanup(self):
        """Remove all handlers and clean up resources."""
        for handler in self.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        self.handlers.clear()
        self._setup_agents.clear()


def setup_experiment_logger(
    experiment_name: str = "experiment",
    problem_id: Optional[int] = None,
    base_log_dir: str = "logs",
) -> ExperimentLogger:
    """
    Setup an experiment logger with problem-specific folders.

    Args:
        experiment_name: Name of the experiment
        problem_id: ID of the current problem
        base_log_dir: Base directory for logs

    Returns:
        ExperimentLogger instance
    """
    exp_logger = ExperimentLogger(
        experiment_name=experiment_name,
        problem_id=problem_id,
        base_log_dir=base_log_dir,
    )

    # Setup separate log files for each agent with lazy initialization
    # This prevents creating empty log files for agents that aren't used
    exp_logger.setup_agent_logger("main", lazy=False)  # main is always used
    exp_logger.setup_agent_logger("tutor", lazy=True)
    exp_logger.setup_agent_logger("student", lazy=True)
    exp_logger.setup_agent_logger("judge", lazy=True)
    exp_logger.setup_agent_logger("prompt_generator", lazy=True)
    exp_logger.setup_agent_logger("reflector", lazy=True)

    # Setup console output
    # exp_logger.setup_console_logger()

    exp_logger.info(f"📁 Logs saved to: {exp_logger.log_folder}", agent="main")

    return exp_logger

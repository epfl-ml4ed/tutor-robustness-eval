# Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks

[📄 Paper](https://arxiv.org/pdf/2604.18660)  

[🤖 Finetuned adversarial student agent](https://huggingface.co/ml4ed-epfl/TR-finetuned-adversary)

[📊 Training data](https://huggingface.co/datasets/ml4ed-epfl/TR-finetuned-adversary-dataset) 


## Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Create a virtual environment using `.venv`:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements_parallel.txt
```

## Configuration

Before running the evaluation, you need to configure the YAML files based on whether you're using **local GPU** or **API services**.

### Configuration Files

- [src/configs/parallel_evaluation.yaml](src/configs/parallel_evaluation.yaml) - Main evaluation configuration
- [src/configs/parallel_evaluation_prompt_generator.yaml](src/configs/parallel_evaluation_prompt_generator.yaml) - LLM generated attack configuration

### GPU (vLLM) Configuration

If you're using local GPU with vLLM, configure each agent in the YAML files as follows:

```yaml
agent:
  tutor:
    model_name: "Qwen/Qwen2.5-7B-Instruct"
    use_vllm: true
    vllm_port: 8000
    base_url: "http://localhost:8000/v1"
    device: "cuda:0"  # Specify GPU device
    # ... other parameters
```

**Key parameters:**
- `use_vllm`: Set to `true`
- `vllm_port`: Port where vLLM server is running (e.g., 8000, 8001, etc.)
- `base_url`: vLLM server URL (e.g., `http://localhost:8000/v1`)
- `device`: CUDA device (e.g., `cuda:0`, `cuda:1`, `cuda:2`)

### API Configuration

If you're using API services, configure each agent as follows:

```yaml
agent:
  judge:
    model_name: "meta-llama/Llama-3.3-70B-Instruct-bfloat16"
    use_vllm: true
    vllm_port: 8000  # random setting
    base_url: API endpoint
    device: "cuda:2". # random setting
```

and set the api key in [src/agents/llm_agent.py](src/agents/llm_agent.py)
```python
self.client = OpenAI(
  api_key="EMPTY",
  base_url=vllm_base_url,
)
```

## Running Evaluations

The [script](script/) directory contains bash scripts for running different evaluation scenarios:

### Evaluation Scripts

#### 1. Base Adversarial Agent Attacks
[script/base_adversarial_agent.sh](script/base_adversarial_agent.sh)

Tests basic adversarial student attacks against different tutor configurations:
- Base adversarial student vs. base in-context tutor
- Base adversarial student vs. tutor with reasoning
- Base adversarial student vs. multi-agent tutor

```bash
bash script/base_adversarial_agent.sh
```

#### 2. Student with Reasoning Attacks
[script/student_with_reasoning.sh](script/student_with_reasoning.sh)

Evaluates student agents with reasoning capabilities against various tutors:
- Student with reasoning vs. base in-context tutor
- Student with reasoning vs. tutor with reasoning
- Student with reasoning vs. multi-agent tutor

```bash
bash script/student_with_reasoning.sh
```

#### 3. Multi-Agent Student Attacks
[script/multi_agent_student.sh](script/multi_agent_student.sh)

Tests multi-agent student attacks with reflection capabilities:
- Multi-agent student vs. base in-context tutor
- Multi-agent student vs. tutor with reasoning
- Multi-agent student vs. multi-agent tutor

```bash
bash script/multi_agent_student.sh
```

#### 4. LLM-Generated Attacks
[script/llm_generated_attacks.sh](script/llm_generated_attacks.sh)

Runs LLM-generated adversarial prompts against all tutor types. This script iterates through multiple attack strategies:
- contextual_manipulation
- direct_request
- emotional_threat
- interpersonal_influence
- intentional_wrong_answer
- request_shaping

```bash
bash script/llm_generated_attacks.sh
```

#### 5. Manually Defined Attacks
[script/manually_defined_attacks.sh](script/manually_defined_attacks.sh)

Evaluates manually refined adversarial prompts from the [collected_prompts](collected_prompts/) directory:
- Manual prompts vs. base in-context tutor
- Manual prompts vs. tutor with reasoning
- Manual prompts vs. multi-agent tutor

```bash
bash script/manually_defined_attacks.sh
```

### Notes

- All scripts use Hydra configuration override syntax
- Results are logged to Weights & Biases (wandb)
- Make sure to configure your vLLM ports and model paths before running
- Each script can be run independently based on your evaluation needs



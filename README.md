# LLM-Disaster-Management-Agents

A multi-agent disaster management system leveraging large language models (LLMs) and GPU-accelerated Vulkan inference to provide forecasting, alerting, and logistics optimization in simulated disaster scenarios.

## Features

- **Generalist Agent**: Proposes overall response plans from VR scenario observations.  
- **Earthquake Agent**: Forecasts aftershock probabilities and timings.  
- **Flood Agent**: Issues flood-risk alerts based on rainfall and declarations data.  
- **Logistics Agent**: Optimizes relief supply delivery routes under constraints.

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/LLM-Disaster-Management-Agents.git
   cd LLM-Disaster-Management-Agents
   ```

2. **Create a Python environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Set up Vulkan environment variables**

   ```bash
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
   export VULKAN_BACKEND=1
   export LLAMA_CONTEXT_USE_VULKAN=1
   ```

## Usage

```bash
python 1.py
```

This will download the model, initialize LLMs, load sample data, instantiate agents, and print each agent’s output.

## Dependencies

* Python ≥ 3.8
* pandas
* huggingface\_hub
* llama-cpp-python
* gpt4all


## License

This project is released under the GPL-3.0 License.


## Citation

This code supports a scientific paper currently under peer review. Citation details will be updated upon publication.

import os
import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from gpt4all import GPT4All


# ========== Configuration ==========
# Vulkan ICD for AMD RX580; set this in PyCharm Run Config or export in shell
os.environ['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/radeon_icd.x86_64.json'
os.environ['VULKAN_BACKEND'] = '1'
os.environ["LLAMA_CONTEXT_USE_VULKAN"] = "1"  # Use Vulkan for GPU acceleration

# Paths
MODEL_REPO    = "microsoft/Phi-3-mini-4k-instruct-gguf"
MODEL_FILE    = "Phi-3-mini-4k-instruct-q4.gguf"
MODEL_DIR     = os.path.join(os.getcwd(), "models")
DATA_DIR      = os.path.join(os.getcwd(), "data")
TWEETS_CSV    = os.path.join(DATA_DIR, "tweets.csv")
DECLS_CSV     = os.path.join(DATA_DIR, "us_disaster_declarations.csv")

# ========== Helper Functions ==========
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = hf_hub_download(
        repo_id = MODEL_REPO,
        filename= MODEL_FILE,
        local_dir = MODEL_DIR
    )
    return model_path


def init_llm(model_path, gpu_layers=33, n_batch=256, ctx=4096, threads=4): # (model_path, gpu_layers=33, ctx=2048, threads=4)
    return Llama(
        model_path   = model_path,
        n_gpu_layers = gpu_layers,
        n_batch      = n_batch,
        n_ctx        = ctx,
        n_threads    = threads,
        verbose      = False
    )


def load_data():
    tweets = pd.read_csv(TWEETS_CSV).sample(n=1000)
    decls  = pd.read_csv(DECLS_CSV).sample(n=500)
    return tweets, decls

# ========== Agent Base Class ==========
class Agent:
    def __init__(self, llm: Llama, name: str):
        self.llm  = llm
        self.name = name

    def generate(self, prompt: str) -> str:
        resp = self.llm.create_chat_completion(messages=[{"role":"user","content":prompt}])
        return resp['choices'][0]['message']['content']

# ========== Specialized Agents ==========
class GeneralistAgent(Agent):
    def process(self, observations: str) -> str:
        prompt = (
            f"You are a disaster-management AI advisor. "
            f"Based on these observations from a VR scenario, propose an overall response plan:\n" + observations
        )
        return self.generate(prompt)

class EarthquakeAgent(Agent):
    def forecast(self, tremor_info: str) -> str:
        prompt = (
            f"You are an earthquake forecasting specialist. "
            f"Given recent tremor data, forecast aftershock probability and timing:\n" + tremor_info
        )
        return self.generate(prompt)

class FloodAgent(Agent):
    def alert(self, rainfall_info: str) -> str:
        prompt = (
            f"You are a flood risk analyst. "
            f"Based on recent rainfall and flood declaration data, issue a flood-risk alert for the region:\n"
            + rainfall_info
        )
        return self.generate(prompt)

class LogisticsAgent(Agent):
    def optimize(self, supply_info: str) -> str:
        prompt = (
            f"You are a logistics optimizer. "
            f"Given current inventory and route constraints, propose optimized delivery routes for relief supplies:\n"
            + supply_info
        )
        return self.generate(prompt)

# ========== Main Demonstration ==========

def main():
    # 1. Download & Initialize
    #print(GPT4All.list_gpus())  # output: ['kompute:Radeon RX 580 Series']

    #model_path = download_model()
    model_path = "C:\\Users\\Computer Parseh\\AppData\\Local\\nomic.ai\\GPT4All\\Phi-3-mini-4k-instruct.Q4_0.gguf"

    llm        = init_llm(model_path)
    #llm = GPT4All(model_path, device='amd')

    # 2. Load sample data
    tweets, declarations = load_data()

    # 3. Instantiate agents
    generalist = GeneralistAgent(llm, "Generalist")
    eq_agent   = EarthquakeAgent(llm, "Earthquake")
    fl_agent   = FloodAgent(llm, "Flood")
    log_agent  = LogisticsAgent(llm, "Logistics")

    # 4. Prepare sample inputs
    obs_text      = (
        "- VR observed building collapse and crowd panic.\n"
        "- Limited road access due to debris.\n"
        "- Communications intermittently down."
    )
    tremor_text   = declarations.head(3).to_csv(index=False)
    rainfall_text = tweets.query("keyword=='rain' or keyword=='flood'").head(5).to_csv(index=False)
    supply_text   = (
        "Inventory: water(5000 units), medkits(2000 units)\n"
        "Road network: 3 blocked highways, 2 open routes."
    )

    # 5. Run agents and display outputs

    print("=== Generalist Agent Response ===")
    print(generalist.process(obs_text), end="\n\n")

    print("=== Earthquake Agent Forecast ===")
    print(eq_agent.forecast(tremor_text), end="\n\n")

    print("=== Flood Agent Alert ===")
    print(fl_agent.alert(rainfall_text), end="\n\n")

    print("=== Logistics Agent Optimization ===")
    print(log_agent.optimize(supply_text))

if __name__ == "__main__":
    main()
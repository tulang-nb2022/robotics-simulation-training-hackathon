# robotics-simulation-training-hackathon
"Prompt-controlled exploration agent"
Goal: Provide more exploration patterns (option space) for reinforcement learning

## Simple Architecture
User Prompt
    ↓
Prompt Embedding / Keyword Parser
    ↓
Exploration Parameters
    ↓
Agent Policy
    ↓
Simulation Environment

## Specifically
prompt
  ↓
LLM or keyword mapping
  ↓
{action_noise, step_size, curiosity_weight}
  ↓
agent movement

## Tech stack
PyGame grid
Agent just moves up, down, left, right.

## Implementation example
if "cautious" in prompt:
    noise = 0.05

elif "random" in prompt:
    noise = 0.3

elif "aggressive" in prompt:
    noise = 0.6

elif "strange" in prompt:
    exploration_bonus = 2.0

## Output
Watch exploration patterns change

---

## Workflow summary (for a non-technical audience)

This section describes what the project does from start to finish, using the real file and function names so you can follow along in the code.

**1. You choose how to run the robot.**  
You run one of two entry points:

- **Scripted mode:** the program `exploration/train_all.py` is run (its `main()` function). It loops over four fixed “modes”: cautious, random, aggressive, and strange.
- **RL mode:** the program `exploration/rl_train.py` is run (its `main()` function). Same four modes, but the robot “learns” via a simple Q-learning loop before drawing its path.

**2. For each mode, the system decides how the robot should behave.**  
The file `exploration/prompt_mapping.py` contains the function `params_from_prompt(prompt)`. It turns a word like “cautious” or “strange” into three numbers that control movement:

- **Action noise** – how often the robot does a random step instead of a planned one.
- **Step size** – how far it moves in one step.
- **Curiosity weight** – how strongly it prefers visiting places it hasn’t been.

If the “LLM switch” is on (the `USE_LLM` setting in your `.env` or environment), that function first asks the cloud-based Nemotron model (via `exploration/llm_client.py`, function `params_from_llm()`) to choose those three numbers. If the switch is off or the call fails, it uses built-in rules in `prompt_mapping.py` (the “keyword mapping”) for cautious, random, aggressive, and strange.

**3. The robot moves on a grid.**  
The file `exploration/env.py` defines a `GridWorld` and an agent that can move up, down, left, or right. The three numbers from step 2 are passed in as `ExplorationParams` and used inside `GridWorld.step()` so that each mode produces a different style of movement (e.g. cautious = low noise and small steps, strange = high curiosity for unseen cells).

**4. The path is turned into a picture.**  
After many steps, the code has a “visit count” for every cell. That data is turned into a heatmap image (e.g. in `train_all.py` by the function `save_heatmap()`) and saved as a PNG. The folder depends on whether the LLM was used:

- **Scripted (`train_all`):** all PNGs go in the **`train_all/`** folder; the filename includes the LLM switch, e.g. `exploration_cautious_llm_off.png`, `exploration_cautious_llm_on.png`.
- **RL (`rl_train`):** pictures go under `outputs_rl/llm_off/` or `outputs_rl/llm_on/`.

So: **you run `train_all` or `rl_train` → for each of the four modes, `params_from_prompt` (and optionally the LLM in `llm_client`) sets behavior → `GridWorld` in `env.py` runs the motion → `save_heatmap` writes a PNG into the right folder.** That’s the full workflow.

---

## Project layout

- `exploration/env.py`: PyGame grid world and visit heatmap.
- `exploration/prompt_mapping.py`: maps text prompts to exploration parameters.
- `exploration/train_all.py`: runs the simple scripted agent for all prompt patterns and saves visualizations.
- `exploration/rl_train.py`: runs a tabular Q-learning agent for all prompt patterns and saves RL-based visualizations.
- `exploration/llm_client.py`: optional Nemotron / Nebius-backed LLM integration for mode-to-parameter mapping.
- `githooks/pre-commit`: git pre-commit hook for basic secrets/PII scanning.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: container for running training on Nebius (or any Docker-capable cloud).

## Running locally

### 1. Create a virtualenv and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you later enable the Nebius / Nemotron integration, `requests` is already included in `requirements.txt`.

### 2. Run the exploration for all prompt patterns

```bash
python -m exploration.train_all
```

This will create PNGs in the **`train_all/`** folder (or the path set by `EXPLORATION_OUTPUT_DIR`). Each filename includes the USE_LLM state so runs with LLM on vs off do not overwrite each other, e.g.:

- `exploration_cautious_llm_off.png`, `exploration_random_llm_off.png`, …
- `exploration_cautious_llm_on.png`, `exploration_random_llm_on.png`, …

You can change the output folder by setting `EXPLORATION_OUTPUT_DIR=/some/path` before running.

### 3. Run the RL-based exploration for all prompt patterns

```bash
python -m exploration.rl_train
```

This will create PNGs under `outputs_rl/`, in subdirs **`outputs_rl/llm_off/`** and **`outputs_rl/llm_on/`** (same `USE_LLM` logic as above). Each subdir contains:

- `rl_exploration_cautious.png`
- `rl_exploration_random.png`
- `rl_exploration_aggressive.png`
- `rl_exploration_strange.png`

**Does USE_LLM apply to rl_train?**  
Yes, but only for **where files are saved**. In `rl_train.py`, `_llm_suffix()` reads `USE_LLM` and chooses the subdir (`llm_off` or `llm_on`), so you can keep separate runs. The **RL policy itself** does not call the LLM: it uses only the built-in keyword logic in `rl_train.py` (`rl_params_from_prompt()` and `make_reward_fn()`). So toggling `USE_LLM` does not change the robot’s learned behavior in RL mode—only the scripted mode in `train_all.py` uses the LLM to set movement parameters.

**What is the “simple Q-learning loop”?**  
It lives in `exploration/rl_train.py` inside the function `train_q_learning(prompt)`:

1. **Q-table**  
   A 3D array `q[y, x, action]` stores one value per grid cell and per action (right, left, up, down). That value is the estimated long-term reward for taking that action in that cell.

2. **Episodes**  
   For each of many episodes (e.g. 200–400, set by `rl_params_from_prompt`), the agent starts in the center and takes steps until a max step count.

3. **Each step**  
   - **Choose action:** `choose_action(q, state, eps, rng)` – with probability `eps` (epsilon) pick a random move; otherwise pick the move with the highest Q-value in the current cell. Epsilon is decreased over episodes (e.g. from 0.9 to 0.05) so the agent explores less over time.  
   - **Apply move:** `apply_action(state, action, width, height)` – move one cell in that direction (with bounds).  
   - **Reward:** `reward_fn(next_state, visits)` – prompt-specific (cautious: penalize revisits and distance from center; aggressive: reward distance from center; strange: big reward for first visit to a cell; etc.).  
   - **Update Q:** classic one-step update:  
     `q[old_cell][action] += alpha * (reward + gamma * max Q[new_cell] - q[old_cell][action])`.  
     So the agent learns “this action in this cell led to this reward plus whatever I’ll get later from the best move in the new cell.”

4. **After training**  
   One long “greedy” rollout (epsilon = 0, so always pick the best Q-action) is run and the visit counts are recorded. That visit map is drawn as the heatmap PNG.

In simpler terms: **the robot does many practice runs, sometimes trying random moves and sometimes choosing what it thinks is best; the prompt (cautious, random, aggressive, strange) decides how it gets rewarded. After that, it does one final run where it always picks its best move, and that run is turned into the heatmap picture.**

---

## Running on Nebius cloud (or any Docker-based VM)

The simplest way to “train the robot” on Nebius is:

1. **Provision a VM** with Docker enabled (GPU is optional; this project is CPU-only).
2. **Clone this repo** onto the VM.
3. **Build the Docker image**.
4. **Run the container** to generate the exploration patterns.

### 1. Provision a VM

From the Nebius console:

- **Create a new compute instance** (Ubuntu 22.04 or similar).
- Ensure it has:
  - At least 2 vCPUs and 4 GB RAM.
  - Docker installed (either via a Nebius image that includes Docker or by installing it yourself after creation).

If you prefer to install Docker manually on the VM:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"
newgrp docker
```

### 2. Clone the repository

SSH into the VM and run:

```bash
git clone https://github.com/your-org/robotics-simulation-training-hackathon.git
cd robotics-simulation-training-hackathon
```

### 3. Build the Docker image

```bash
docker build -t exploration-agent .
```

### 4. Run the container and collect outputs

Choose a directory on the VM where you want the PNGs to be saved, e.g. `/home/ubuntu/outputs`:

```bash
mkdir -p /home/ubuntu/outputs

docker run --rm \
  -e EXPLORATION_OUTPUT_DIR=/outputs \
  -v /home/ubuntu/outputs:/outputs \
  exploration-agent
```

Inside the container, `exploration.train_all` will:

- Run the simulation for each prompt pattern: **cautious**, **random**, **aggressive**, **strange**.
- Save a heatmap PNG for each pattern into `/outputs`.

After the container exits, you will find the visualizations on the VM at `/home/ubuntu/outputs`.

You can download them to your local machine using `scp` or any Nebius file-transfer tooling.

---

## Extending the robot “training”

The project offers two levels of “training”:

- **Scripted exploration (`exploration.train_all`)**:

- The **prompt** is mapped to exploration hyperparameters (`action_noise`, `step_size`, `curiosity_weight`).
- The agent then runs for many steps in a PyGame grid world.
- The resulting visitation heatmap is the learned exploration pattern for that prompt.

- **RL-based exploration (`exploration.rl_train`)**:

- A tabular Q-learning agent learns a state–action value table over the grid.
- The prompt controls both RL hyperparameters and the reward function.
- After training, the greedy policy is rolled out once to produce a final visitation pattern.

To make this more sophisticated, you could:

- Swap in a proper RL algorithm (e.g., Q-learning) where the prompt defines reward shaping.
- Increase grid size and step count.
- Add obstacles and goals to the grid.
- Log trajectories and replay them as animations instead of static heatmaps.

### Using Nemotron-3-Super-120b-a12b on Nebius to shape behavior

If you have access to **Nemotron-3-Super-120b-a12b** on Nebius and a token from **Nebius Token Factory**, you can let the LLM decide what
`action_noise`, `step_size`, and `curiosity_weight` should be for each mode.

The integration is intentionally minimal and environment-driven:

- Set the following environment variables (e.g., in `.env` or on your Nebius VM):

```bash
# Switch: set to 1 (or true/yes) to use the LLM; set to 0 or omit to use keyword mapping only.
export USE_LLM=1
export NEBIUS_LLM_ENDPOINT="https://your-nebius-endpoint.example.com/v1/chat/completions"
export NEBIUS_TOKEN="YOUR_TOKEN_FACTORY_ISSUED_BEARER_TOKEN"
```

To compare behavior with the LLM on vs off, run once with `USE_LLM=1` and once with `USE_LLM=0` (or unset); the former uses Nemotron to set parameters, the latter uses the built-in keyword mapping.

- `exploration/llm_client.py` will:
  - Send a short system prompt describing the gridworld and the four modes:
    **cautious**, **random**, **aggressive**, **strange**.
  - Ask the model to reply with **raw JSON only**:
    - `action_noise` (0–1),
    - `step_size` (1, 2, or 3),
    - `curiosity_weight` (non-negative float).
  - Parse the JSON and return it as an `ExplorationParams` instance.

- `exploration/prompt_mapping.py` calls `params_from_llm(prompt)` first; if the LLM or parsing fails, it falls back to the local keyword mapping.

This means that when properly configured, Nemotron is “told” what each search mode means and directly sets the movement parameters that shape both the scripted and RL-based exploration patterns.

---

## Enabling the git pre-commit hook (secrets / PII scanning)

This repo includes a lightweight **pre-commit hook** in `githooks/pre-commit` that scans staged changes for:

- Common credential markers (e.g., `API_KEY`, `TOKEN=`, `PASSWORD=`).
- Private key blocks (`BEGIN RSA PRIVATE KEY`, etc.).
- Likely email addresses.
- Simple AWS-style access key IDs (`AKIA…`).

To enable it for this clone:

```bash
git config core.hooksPath githooks
chmod +x githooks/pre-commit
```

From then on, every `git commit` will:

- Scan only **staged** files (text files) for suspicious patterns.
- **Block the commit** with a message if something looks like a secret/PII, so you can fix it before pushing.

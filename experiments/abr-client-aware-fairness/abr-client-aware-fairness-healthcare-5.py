# == PYTHON MODULES ==
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler # pip install scikit-learn
from stable_baselines3 import PPO
import multiprocessing

import glob
from typing import Tuple, Optional

# == PERSONALIZED MODULES ==
from decision_module import adaptive_fairness_decision # logic decision for bitrate selection
from abr_quality_selector import select_bitrate,publish_client_score,get_global_scores, apply_quality #abr mode selector, global-scores sharing/recovering
from metric_processing import start_global_scalability_sampler, finalize_global_scalability, collect_playback_metrics, process_metrics, calculate_avg_buffer_level, calculate_jitter, detect_quality_changes, update_client_metrics_snapshot, calculate_weighted_quality, normalize_qoe_metrics, calculate_composite_qoe
from fairness_utils import update_fairness_scores, calculate_tf_qoe, calculate_qfs, calculate_bfi, calculate_saf, calculate_fairness_index,calculate_fairness_metrics


import time
import logging
import threading
import matplotlib.pyplot as plt
import datetime
import numbers
from datetime import datetime, timezone
import numpy as np

import json
import csv
import random
import os


# RELEASE UPDATES
# =======================================
#  - Scalability improvements for many clients and metric inclusion fields [Experiment 5]


# ANNOTATIONS FOR CONFIGURATION
# =======================================
# VIRTUAL AMBIENT ACTIVATION [CLIENT 1]
    # Dependencies for Python from requirements.txt working
    # python3 -m venv /scripts/venv
    # source /scripts/venv/bin/activate

# PIP INSTALATION FIX
# python -m pip install --upgrade pip

# CLIENT Access by Docker exec
    # docker exec -it client-g1 /bin/bash
    # docker exec -it client-g2 /bin/bash
    # docker exec -it client-g3 /bin/bash

# EXECUTE HEADLESS MODE WITH CHROMIUM
    # chromium --headless --disable-gpu --dump-dom http://localhost:8090/test-js.html

# EXECUTE AS NON ROOT WITH MAHIMAHI USER
    # su mahimahi-user 

# EXECUTE SCENARIOS(JOINT QOE PAPER)
# SCENARIO 1[Variability Network Impact On Quality of Experience]
    # GROUP 1 [trace-low]
    # Curta duração, com muita variabilidade (instabilidade) - Lower Fairness.
    # mm-link ./traces/mahimahi/Verizon-LTE-short.down ./traces/mahimahi/Verizon-LTE-short.up python3 abr-client-aware-fairness-healthcare.py

    # GROUP 2[trace-high]
    # Mais longo, com variação mais suave/estável ao longo do tempo - Higher Fairness.
    # mm-link ./traces/mahimahi/Verizon-LTE-driving.down ./traces/mahimahi/Verizon-LTE-driving.up python3 abr-client-aware-fairness-healthcare.py
    
    # GROUP 3
    # mm-link ./traces/mahimahi/TMobile-UMTS-driving.down ./traces/mahimahi/TMobile-UMTS-driving.up python3 abr-client-aware-fairness-healthcare.py



# SCENARIO 2 [Latency and Fairness Impact on Quality Selection]
    # GROUP 1
    # mm-link ./traces/mahimahi/Verizon-LTE-short.down ./traces/mahimahi/Verizon-LTE-short.up python3 abr-client-aware-fairness-healthcare.py

    # GROUP 2
    # mm-link ./traces/mahimahi/Verizon-LTE-driving.down ./traces/mahimahi/Verizon-LTE-driving.up python3 abr-client-aware-fairness-healthcare.py
    
    # GROUP 3
    # mm-link ./traces/mahimahi/TMobile-UMTS-driving.down ./traces/mahimahi/TMobile-UMTS-driving.up python3 abr-client-aware-fairness-healthcare.py

# -- ML MODEL LOAD -- 
# ##############################
# model = PPO.load("./ml-models/pensieve_scenario2_model.zip")
# model = PPO.load("pensieve_weights_model.zip")



# -- SIMULATION DATA OPTIONS -- 
# ##############################

# -- Storage metrics data --
start_time_simulation = None
end_time_simulation = None
date_simulation = None
total_playbacktime_simulation = None
date_time_format = '%d-%b-%Y:%H:%M:%S' #Default  -> '%Y-%m-%dT%H:%M:%S'
time_format = '%H:%M:%S'
date_format = '%d-%b-%Y'
shared_dir="/scripts/simulations/shared-scores"

# For QoE calculation score
global min_score, max_score

score_range = {
    "min_score": float('inf'),
    "max_score": float('-inf')
}
# Fixed to be accepted in COMPOSITE QOE calculate
# score_range = (0,1)


# --- INITIALIZATION OPTIONS ---
# ##############################

# if 'min_score' not in globals() or min_score == float('inf'):
#     min_score = float('inf')  # Inicializa como infinito para ser atualizado
# if 'max_score' not in globals() or max_score == float('-inf'):
#     max_score = float('-inf')  # Inicializa como -infinito para ser atualizado




date = datetime.now().strftime(date_format)
time_now = datetime.now().strftime(time_format)

print(f"Data e hora do dia: Data -> {date} : Hora -> {time_now}")

player_url = 'http://172.25.0.3:80' #URL LOAD BALANCING SERVER [EDGE1, EDGE2...]- internal
mpd_url = "http://172.25.0.3:80/videos/dash/master.mpd" # URL do MPD - internal

# player_url = 'http://localhost:8080' #URL LOAD BALANCING SERVER [EDGE1, EDGE2...] - external
# mpd_url = "http://localhost:8080/videos/dash/master.mpd" # URL do MPD - external




# --- PLAYBACK OPTIONS ---
# ##############################
playback_wait = 50 # Tempo de espera para Playback localizar video (default 05)
simulation_total_time = 0 #Tempo total da simulação de x segundos [150 = 19seg, ]
num_clients = 0 # Número de clientes a serem simulados (host max 30, linux max )
min_latency_samples = 5 # Minimum quant of latency samples for jitter calculation
pageTimeout = 300 #(1- default 60 [exp 01], 2- changed to 300 [exp 04] )
scriptTimeout = 200 #(default 60)
segmentCheckInterval = 1
segmentRequestTimeout = 140 #(default 60: Relation for 20 clients - 60, 25 clients - 140)
minVideoSegmentsRequired = 3 # (default 3 for secure measurement)
MAX_QUALITY_LEVEL = 5  # 6 quality levels: 0 to 5

STRICT_MODE = False

# timeoutWait = 2000 # Configuração de timeout de script (default 120)

# ##############################


# --- METRICS OPTIONS ---
# ##############################
clients = [] #To store total of clients
client_metrics = {} # list to store client metrics

with open('player_metrics.js', 'r') as file:
    js_code = file.read()


# --- WEIGHTS OPTIONS ---
# ##############################
USE_WEIGHTS = True  # Default: True

# --- WEIGHTS METRICS ---
# ##############################
# Positive metrics: Good influence on experience when the metric being up
# negative metrics: Bad influcent on experience whe the metric being up
# weights = {
#     'buffering_time': -0.15,
#     'playback_interruptions': -0.20,
#     'avg_throughput': 0.25,
#     'latency': -0.15,
#     'jitter': -0.05,
#     'avg_buffer_level': 0.10,
#     'dropped_frames_total': -0.05,
#     'quality_changes_total': -0.05,
# }

default_weights ={
    'buffering_time': 0,
    'playback_interruptions': 0,
    'latency': 0,
    'jitter': 0,
    'dropped_frames_total': 0,
    'quality_changes_total': 0,
    'avg_throughput': 1,
    'avg_buffer_level': 1
}


def load_weights_from_json(file_path, default_weights=None):
    """
    Load weights from a JSON file. If not found, use default_weights.

    :param file_path: Path to the JSON file.
    :param default_weights: Optional dictionary of default weights.
    :return: Dictionary of weights.
    """
    try:
        with open(file_path, 'r') as f:
            weights_config = json.load(f)

        weights = {}
        for metric, config in weights_config.items():
            if isinstance(config, dict):
                weights[metric] = config.get("weight", 0)
            else:
                weights[metric] = config  # Backward compatibility for simple formats
        print("📥 Weights loaded:", weights)

        return weights
    except Exception as e:
        logging.error(f"Failed to load weights from {file_path}: {e}")
        return default_weights or {}




def load_model_and_env_config(abr_mode: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    Resolve paths for RL model and environment config based on abr_mode.
    
    Returns:
        (model_path, env_config) -- model_path is string or None; env_config is dict or None.
    Behavior:
      - resolves ./rl-model/{abr_mode_norm}-model.zip (abr_mode normalized)
      - if not found, tries to find any ./rl-model/*{abr_mode_norm}*.zip as fallback
      - env json is optional (if missing we log a warning but still return None)
      - special-case: if abr_mode_norm == "pensieve-baseline" we skip loading the env config
    """
    if not abr_mode:
        logging.error("ABR mode is None! Cannot resolve model path and config.")
        return None, None

    abr_mode_norm = abr_mode.strip().lower().replace('_', '-')
    model_path = f"./rl-model/{abr_mode_norm}-model.zip"
    config_path = f"./rl-model/{abr_mode_norm}-model-env-config.json"

    logging.info(f"Resolving model/env config for ABR mode: '{abr_mode}' -> normalized '{abr_mode_norm}'")
    logging.info(f"Primary model path: {model_path}")
    logging.info(f"Primary config path: {config_path}")

    # --- Resolve model path ---
    resolved_model_path = None
    if os.path.exists(model_path):
        resolved_model_path = model_path
        logging.info(f"✅ Model path found: {model_path}")
    else:
        # fallback: try to find any matching file
        fallback_pattern = f"./rl-model/*{abr_mode_norm}*.zip"
        matches = sorted(glob.glob(fallback_pattern))
        if matches:
            resolved_model_path = matches[0]
            logging.info(f"✅ Using fallback model path: {resolved_model_path}")
        else:
            logging.error(f"❌ No model files found with pattern: {fallback_pattern}")

    # --- Load env config (optional) ---
    if abr_mode_norm == "pensieve-baseline":
        logging.info("ℹ️ abr_mode is Pensieve-Baseline — skipping env config load (not required).")
        env_config = None
    else:
        try:
            with open(config_path, 'r') as f:
                env_config = json.load(f)
            logging.info(f"✅ Env config loaded successfully from: {config_path}")
        except FileNotFoundError:
            logging.warning(f"⚠️ Env config not found (optional): {config_path}")
            env_config = None
        except Exception as e:
            logging.warning(f"⚠️ Could not parse env config {config_path}: {e!r}")
            env_config = None

    return resolved_model_path, env_config




def load_model_and_env_config_old(abr_mode: str) -> Tuple[Optional[PPO], Optional[dict]]:
    """
    Load the RL model and environment config based on abr_mode.

    Returns:
        (model, env_config) -- model is a PPO instance or None; env_config is dict or None.
    Behavior:
      - tries to load ./rl-model/{abr_mode_norm}-model.zip (abr_mode normalized)
      - if not found, tries to find any ./rl-model/*{abr_mode_norm}*.zip as fallback
      - env json is optional (if missing we log a warning but still return the model)
      - special-case: if abr_mode_norm == "pensieve-baseline" we skip loading the env config
    """
    if not abr_mode:
        logging.error("ABR mode is None! Cannot load model and config.")
        return None, None

    abr_mode_norm = abr_mode.strip().lower().replace('_', '-')
    model_path = f"./rl-model/{abr_mode_norm}-model.zip"
    config_path = f"./rl-model/{abr_mode_norm}-model-env-config.json"

    logging.info(f"Loading model and env config for ABR mode: '{abr_mode}' -> normalized '{abr_mode_norm}'")
    logging.info(f"Primary model path: {model_path}")
    logging.info(f"Primary config path: {config_path}")

    # model = None
    # env_config = None

    # --- Try to load the model (primary path) ---
    try:
        model = PPO.load(model_path)
        logging.info(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        logging.warning(f"Could not load model from {model_path}: {e!r}")
        # --- fallback: try to find a zip that contains abr_mode_norm in its filename ---
        fallback_pattern = f"./rl-model/*{abr_mode_norm}*.zip"
        matches = sorted(glob.glob(fallback_pattern))
        if matches:
            for candidate in matches:
                try:
                    model = PPO.load(candidate)
                    logging.info(f"✅ Model loaded successfully from fallback: {candidate}")
                    break
                except Exception as e2:
                    logging.warning(f"Fallback candidate {candidate} failed to load: {e2!r}")
        else:
            logging.error(f"❌ No fallback models found with pattern: {fallback_pattern}")

    # --- Load env config when appropriate (but do not make it mandatory) ---
    if abr_mode_norm == "pensieve-baseline":
        logging.info("ℹ️ abr_mode is Pensieve-Baseline — skipping env config load (not required).")
        env_config = None
    else:
        try:
            with open(config_path, 'r') as f:
                env_config = json.load(f)
            logging.info(f"✅ Env config loaded successfully from: {config_path}")
        except FileNotFoundError:
            logging.warning(f"⚠️ Env config not found (optional): {config_path} — continuing with env_config=None")
            env_config = None
        except Exception as e:
            logging.warning(f"⚠️ Could not parse env config {config_path}: {e!r} — continuing with env_config=None")
            env_config = None

    # --- Optional: quick introspection for debugging (safe) ---
    if model is not None:
        obs_space = None
        try:
            obs_space = getattr(model, "observation_space", None)
            if obs_space is None and getattr(model, "policy", None) is not None:
                obs_space = getattr(model.policy, "observation_space", None)
        except Exception:
            obs_space = None

        if obs_space is not None:
            logging.info(f"Model observation_space: {getattr(obs_space, 'shape', str(obs_space))}")
        else:
            logging.info("Model loaded but observation_space couldn't be introspected (this is okay).")

    logging.info("Returning (model, env_config)")
    logging.info(f" Inside load model -> Model loaded: {model}, Env config loaded: {env_config}")
    return model, env_config


# LOAD JSON FILE to use correct RL-MODEL
def load_model_and_env_config_old(abr_mode):
    """
    Load the RL model and environment config based on abr_mode.
    Expected file naming convention:
    - Model: ./rl-model/{abr_mode}-model.zip
    - Config: ./rl-model/{abr_mode}-model-env-config.json

    Special case: when abr_mode is Pensieve-Baseline we DO NOT load the env config,
    because it's not required for the baseline model.
    """

    if not abr_mode:
        logging.error("ABR mode is None! Cannot load model and config.")
        return None, None

    logging.info(f"Loading model and env config for ABR mode: {abr_mode}")

    model = None
    env_config = None

    # normalize mode for file naming and comparisons (accepts pensieve_baseline, pensieve-baseline, etc.)
    abr_mode_norm = abr_mode.strip().lower().replace('_', '-')
    model_path = f"./rl-model/{abr_mode_norm}-model.zip"
    config_path = f"./rl-model/{abr_mode_norm}-model-env-config.json"

    logging.info(f"Showing Model path: {model_path} | Config path: {config_path}")

    # Try loading the model
    try:
        model = PPO.load(model_path)
        logging.info(f"✅ Model loaded successfully: {model_path}")
    except Exception as e:
        logging.error(f"❌ Error loading model {model_path}: {e}")

    # If this is Pensieve-Baseline, skip loading the env config
    if abr_mode_norm == "pensieve-baseline":
        logging.info("ℹ️ abr_mode is Pensieve-Baseline — skipping env config load (not required).")
        env_config = None
    else:
        # Try loading the env config for other modes
        try:
            with open(config_path, 'r') as file:
                env_config = json.load(file)
            logging.info(f"✅ Env config loaded successfully: {config_path}")
        except Exception as e:
            logging.warning(f"⚠️ Could not load env config {config_path}: {e}")
            env_config = None

    # Optional: inspect loaded model for debugging
    if model is not None:
        try:
            logging.info(f"Model observation space shape: {model.observation_space.shape}")
        except Exception:
            logging.warning("Loaded model does not have an observation_space attribute or it couldn't be read.")

    logging.info("Returning model and env_config")
    return model, env_config


# PIPELINE - x =====  DEFINE PRIORITY MODE CLIENT DEVICE, SECTOR OR GROUP ==== 
def assign_urgency_level(client_id):
    if client_id % 3 == 1:
        return 2  # high priority
    elif client_id % 3 == 2:
        return 1  # medium priority
    else:
        return 0  # low priority


# -- PLAYER INITIALIZATION --
def configure_dash_player(abr_mode):
    # Buffer settings fixos ou parametrizáveis
    buffer_settings = {
        "initialBufferTime": 2,
        "stableBufferTime": 10,
        "bufferTimeAtTopQuality": 20,
        "bufferTimeAtTopQualityLongForm": 30
    }

    # Decide ABR settings conforme abr_mode
    if abr_mode in ["DASH-DEFAULT", "DASH-QOE-BASED"]:
        abr_config = {
            "enabled": True
        }
    elif abr_mode == "BOLA":
        abr_config = {
            "enabled": True,
            "useDefaultABRRules": False,
            "ABRStrategy": "abrBola"
        }

    elif abr_mode == "L2A":
        abr_config = {
            "enabled": True,
            "useDefaultABRRules": False,   # evita rodar Throughput/BOLA junto
            "ABRStrategy": "abrL2A"        # rótulo descritivo; o enable real é no player.js
        }

    else:
        # Todos os modos RL
        abr_config = {
            "enabled": False
        }

    # Monta o settings final para player.updateSettings
    settings = {
        "streaming": {
            "abr": abr_config,
            "buffer": buffer_settings
        }
    }
    return settings



# --- CLIENT METRICS INITIALIZATION ---
def initialize_client_metrics(client_id, date, start_time_simulation, startup_seed, group_client, urgency_level, abr_mode):
    return {

        # === Client Identification ===
        'client_id': client_id,

        # === Time and execution control ===
        'date_simulation': date,
        'start_time_simulation': start_time_simulation,
        'end_time_simulation': None,
        'total_playback_time': 0,
        # 'timestamp_simulation_samples': 0,
        'startup_seed': startup_seed,
        'status': "",

        # === General client information ===
        'group_client': group_client,
        'urgency_level': urgency_level,
        'abr_mode': abr_mode,

        # === Unique metrics (aggregate or instant values) ===
        'startup_delay': 0,
        'buffering_time': 0,
        'avg_buffer_level': 0,
        'playback_interruptions': 0, #bufferingCount
        'avg_throughput': 0,
        'buffering_count': 0,
        'avg_latency': 0,
        'dropped_frames_total': 0,
        'quality_changes_total': 0,
        'max_quality_level': 0, #new - included in experiment 05
        'segment_index': 0, #new - included in experiment 05

        # === Fairness and Composite QoE metrics ===
        'composite_qoe': 0,
        'tf_qoe': 0,
        'qfs': 0,
        'bfi': 0,
        'saf': 0,

        # === Sample history ===
        'throughput_samples':[],
        'latency_samples': [],
        'jitter_samples': [],
        'buffer_level_samples': [],
        'quality_level_samples': [],
        'quality_changes_samples': [],
        'bitrate_samples': [], #Included in experiment 05
        'dropped_frames_samples': [],
        'tf_qoe_samples': [],
        'qfs_samples': [],
        'bfi_samples': [],
        'saf_samples': [],
        'timestamp_samples': []
        # 'bitrate_decisions': [],  # (Opcional para histórico de decisões ABR)
    }

# PIPELINE - 1 =====  START APPLICATION BY NUMBER OF USERS AND START EMERCENCY LEVEL USERS ====

# imports (no topo do arquivo onde está run_simulations)
import threading
from metric_processing import (
    start_global_scalability_sampler,
    finalize_global_scalability
)

# PIPELINE - 1 =====  START APPLICATION BY NUMBER OF USERS AND START EMERCENCY LEVEL USERS ====

def run_simulations(
        num_clients, simulation_total_time, group_client, 
        scenario, abr_mode, use_segment_based_collection,
        segment_interval, startup_seed, execution_mode="thread",
        enable_individual_logs=True, weights=None,
        score_range=None, js_code=js_code, model_path=None,
        urgency_level_enabled=True,
        enable_global_scalability_sampler=True,
        global_sampler_interval=2.0
):

    global client_metrics
    print("STARTING SIMULATION PROCESS")
    print("============================================================ \n")
    date_simulation = date

    logging.info(f"Inside run_simulations -> Starting with {num_clients} clients, abr_mode={abr_mode}, model_path={model_path}")

    shared_metrics = {'global_scalability_samples': []}
    stop_evt = None
    sampler_thread = None

    if enable_global_scalability_sampler:
        stop_evt = threading.Event()
        sampler_thread = start_global_scalability_sampler(
            shared_metrics, interval=global_sampler_interval, stop_event=stop_evt
        )
        logging.info(f"[E05] Global sampler STARTED (interval={global_sampler_interval}s)")
    else:
        logging.info("[E05] Global sampler DISABLED")

    try:
        # ======== (2) SEEDS ========
        if startup_seed is not None:
            random.seed(startup_seed)

        startup_window = simulation_total_time * 0.2  # 20% do tempo total
        seedDelays = [round(random.uniform(0, startup_window), 2) for _ in range(num_clients)]
        logging.info(f"Showing seed delay inicialization: {seedDelays}")

        # ======== (3) PREPARA args ========
        client_args = []
        for idx, client_id in enumerate(range(1, num_clients + 1)):
            start_time_simulation = time.time()
            seed_delay = seedDelays[idx]
            urgency_level = assign_urgency_level(client_id)
            logging.info(f"Starting simulation for client {client_id} at "
                         f"{datetime.fromtimestamp(start_time_simulation).strftime(time_format)} "
                         f"with urgency level {urgency_level}")

            client_args.append((
                client_id, start_time_simulation, abr_mode, urgency_level,
                simulation_total_time, use_segment_based_collection, segment_interval,
                seed_delay, startup_seed, enable_individual_logs, group_client, scenario,
                weights, score_range, js_code, model_path, urgency_level_enabled
            ))

        # ======== (4) EXECUÇÃO ========
        if execution_mode == "thread":
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [executor.submit(simulate_dash_js_client, *args) for args in client_args]
                for future in futures:
                    try:
                        result = future.result()
                        if not result or result.get("status") == "error":
                            client_id = (result or {}).get("client_id", "Unknown")
                            logging.warning(f"⚠️ Client {client_id} reported error status.")
                            if STRICT_MODE:
                                raise RuntimeError(f"Simulation aborted due to failure in client {client_id}.")
                            continue
                        client_metrics[result['client_id']] = result["metrics"]
                    except Exception as e:
                        logging.error(f"A client simulation raised an exception: {e}")
                        if STRICT_MODE:
                            raise RuntimeError("Simulation aborted due to exception in client execution.")
                        continue

        elif execution_mode == "process":
            from multiprocessing import Pool
            with Pool(processes=num_clients) as pool:
                results = pool.map(simulate_dash_js_client_wrapper, client_args)
                for result in results:
                    if not result or result.get("status") != "ok":
                        client_id = (result or {}).get("client_id", "Unknown")
                        logging.warning(f"⚠️ Client {client_id} reported error or invalid result.")
                        if STRICT_MODE:
                            raise RuntimeError(f"Simulation aborted due to failure in client {client_id}.")
                        continue
                    client_metrics[result['client_id']] = result["metrics"]
        else:
            raise ValueError("Invalid execution mode.")

    finally:
        if enable_global_scalability_sampler and stop_evt and sampler_thread:
            stop_evt.set()
            sampler_thread.join(timeout=2.0)
            samples = shared_metrics.get('global_scalability_samples', [])
            logging.info(f"[E05] Global sampler STOPPED (samples={len(samples)})")

            global_scalab = finalize_global_scalability(shared_metrics) if samples else {}
            global_fields = {
                'global_container_cpu_pct_avg':  global_scalab.get('container_cpu_pct_avg', None),
                'global_container_cpu_pct_p95':  global_scalab.get('container_cpu_pct_p95', None),
                'global_container_mem_pct_avg':  global_scalab.get('container_mem_pct_avg', None),
                'global_container_mem_pct_p95':  global_scalab.get('container_mem_pct_p95', None),
                'global_enable_scalability_sampler': bool(enable_global_scalability_sampler),
                'global_sampler_interval_s': float(global_sampler_interval),
                'global_samples_count': int(len(samples)),
            }
            for cid, metrics in list(client_metrics.items()):
                if isinstance(metrics, dict):
                    metrics.update(global_fields)

            logging.info(f"[E05] Global scalability injected into clients: {global_fields}")

    logging.info("🔍 Dumping client_metrics structure (preview of first 3 clients):")
    for cid in [k for k in client_metrics.keys() if isinstance(k, int)][:3]:
        logging.info(json.dumps({cid: client_metrics[cid]}, indent=2))

    with open("client_metrics_debug.json", "w") as f:
        json.dump(client_metrics, f, indent=2)

        return client_metrics
    
    
    #Versão anterior para referência
    def run_simulations_old2(
        num_clients, simulation_total_time, group_client, 
        scenario, abr_mode, use_segment_based_collection,
        segment_interval, startup_seed, execution_mode="thread",
        enable_individual_logs=True, weights=None,
        score_range=None, js_code=js_code, model=None,
        urgency_level_enabled=True,
        enable_global_scalability_sampler=True,   # <— LIGA/DESLIGA
        global_sampler_interval=2.0               # <— intervalo (leve)
    ):

    global client_metrics  # mantém seu comportamento atual
    print("STARTING SIMULATION PROCESS")
    print("============================================================ \n")
    date_simulation = date

    logging.info(f"Inside run_simulations -> Starting run_simulations with {num_clients} clients, model ={model}")


    shared_metrics = {'global_scalability_samples': []}  # <- importante
    stop_evt = None
    sampler_thread = None

    if enable_global_scalability_sampler:
        stop_evt = threading.Event()
        sampler_thread = start_global_scalability_sampler(
            shared_metrics, interval=global_sampler_interval, stop_event=stop_evt
        )
        logging.info(f"[E05] Global sampler STARTED (interval={global_sampler_interval}s)")
    else:
        logging.info("[E05] Global sampler DISABLED")



    # ======== (1) START sampler global (opcional) ========
    shared_metrics = {}
    stop_evt = None
    sampler_thread = None
    if enable_global_scalability_sampler:
        stop_evt = threading.Event()
        sampler_thread = start_global_scalability_sampler(
            shared_metrics, interval=global_sampler_interval, stop_event=stop_evt
        )

    try:
        # ======== (2) SEEDS ========
        if startup_seed is not None:
            random.seed(startup_seed)

        startup_window = simulation_total_time * 0.2  # 20% do tempo total
        seedDelays = [round(random.uniform(0, startup_window), 2) for _ in range(num_clients)]
        logging.info(f"Showing seed delay inicialization: {seedDelays}")

        # ======== (3) PREPARA args ========
        client_args = []
        for idx, client_id in enumerate(range(1, num_clients + 1)):
            start_time_simulation = time.time()
            seed_delay = seedDelays[idx]
            urgency_level = assign_urgency_level(client_id)
            logging.info(f"Starting simulation for client {client_id} at "
                         f"{datetime.fromtimestamp(start_time_simulation).strftime(time_format)} "
                         f"with urgency level {urgency_level}")

            client_args.append((
                client_id, start_time_simulation, abr_mode, urgency_level,
                simulation_total_time, use_segment_based_collection, segment_interval,
                seed_delay, startup_seed, enable_individual_logs, group_client, scenario,
                weights, score_range, js_code, model, urgency_level_enabled
            ))

        # ======== (4) EXECUÇÃO ========
        if execution_mode == "thread":
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = [executor.submit(simulate_dash_js_client, *args) for args in client_args]
                for future in futures:
                    try:
                        result = future.result()
                        if not result or result.get("status") == "error":
                            client_id = (result or {}).get("client_id", "Unknown")
                            logging.warning(f"⚠️ Client {client_id} reported error status.")
                            if STRICT_MODE:
                                raise RuntimeError(f"Simulation aborted due to failure in client {client_id}.")
                            continue
                        client_metrics[result['client_id']] = result["metrics"]
                    except Exception as e:
                        logging.error(f"A client simulation raised an exception: {e}")
                        if STRICT_MODE:
                            raise RuntimeError("Simulation aborted due to exception in client execution.")
                        continue

        elif execution_mode == "process":
            from multiprocessing import Pool
            with Pool(processes=num_clients) as pool:
                results = pool.map(simulate_dash_js_client_wrapper, client_args)
                for result in results:
                    if not result or result.get("status") != "ok":
                        client_id = (result or {}).get("client_id", "Unknown")
                        logging.warning(f"⚠️ Client {client_id} reported error or invalid result.")
                        if STRICT_MODE:
                            raise RuntimeError(f"Simulation aborted due to failure in client {client_id}.")
                        continue
                    client_metrics[result['client_id']] = result["metrics"]
        else:
            raise ValueError("Invalid execution mode.")

    finally:
        # ======== (STOP sampler global e injetar métricas globais em cada cliente) ========
        if enable_global_scalability_sampler and stop_evt and sampler_thread:
            stop_evt.set()
            sampler_thread.join(timeout=2.0)
            samples = shared_metrics.get('global_scalability_samples', [])
            logging.info(f"[E05] Global sampler STOPPED (samples={len(samples)})")



            # global_scalab = finalize_global_scalability(shared_metrics)
            global_scalab = finalize_global_scalability(shared_metrics) if samples else {}

            # Campos globais -> injetar em TODOS os clientes (mesmo valor p/ o cenário)
            global_fields = {
                'global_container_cpu_pct_avg':  global_scalab.get('container_cpu_pct_avg', None),
                'global_container_cpu_pct_p95':  global_scalab.get('container_cpu_pct_p95', None),
                'global_container_mem_pct_avg':  global_scalab.get('container_mem_pct_avg', None),
                'global_container_mem_pct_p95':  global_scalab.get('container_mem_pct_p95', None),
                # (opcional) log de como o cenário foi rodado:
                'global_enable_scalability_sampler': bool(enable_global_scalability_sampler),
                'global_sampler_interval_s': float(global_sampler_interval),
                'global_samples_count': int(len(samples)),
            }

            # injeta nos dicionários de cada cliente (sem criar chaves extras fora dos clientes)
            for cid, metrics in list(client_metrics.items()):
                if isinstance(metrics, dict):
                    metrics.update(global_fields)

            logging.info(f"[E05] Global scalability injected into clients: {global_fields}")


    # ======== (6) LOGs finais (seu código) ========
    logging.info(f"\n ======== SUMMARY RESULT OF SIMULATIONS ======================================================")
    logging.info(f"\n ====>> SUMMARY OF EXECUTION ======")
    success_count = sum(1 for m in client_metrics.values() if isinstance(m, dict) and m.get("status") == "ok")
    error_count = sum(1 for m in client_metrics.values() if isinstance(m, dict) and m.get("status") == "error")
    logging.info(f"✔️ Simulation finished: SUCCEEDED: {success_count}, FAILED: {error_count}. \n \n")

    logging.info("🔍 Dumping client_metrics structure (preview of first 3 clients):")
    for cid in [k for k in client_metrics.keys() if isinstance(k, int)][:3]:
        logging.info(json.dumps({cid: client_metrics[cid]}, indent=2))

    with open("client_metrics_debug.json", "w") as f:
        json.dump(client_metrics, f, indent=2)
    logging.info(f" ========================================================================================= ")

    return client_metrics


# whitout processes control metrics
def run_simulations_OLD(
        num_clients, simulation_total_time, group_client, 
        scenario, abr_mode, use_segment_based_collection,
        segment_interval, startup_seed, execution_mode="thread",
        enable_individual_logs=True, weights=None,
        score_range=None, js_code=js_code, model=None,
        urgency_level_enabled=True
    ):

    global client_metrics #adopting the existent and already created client metrics.



    print("STARTING SIMULATION PROCESS")
    print("============================================================ \n")
    date_simulation = date

    # START SEED CONFIGURATION FOR startup delay
    if startup_seed is not None:
        random.seed(startup_seed)

    startup_window = simulation_total_time * 0.2  # 20% of the total time; adjust as needed
    seedDelays = [
        round(random.uniform(0, startup_window), 2) 
        for _ in range(num_clients)
    ]
    logging.info(f"Showing seed delay inicialization: {seedDelays}")


    # ==== PREPARING ARGUMENT LIST ====
    client_args = []
    for idx, client_id in enumerate(range(1, num_clients + 1)):
        start_time_simulation = time.time()
        seed_delay = seedDelays[idx]
        urgency_level = assign_urgency_level(client_id)
        logging.info(f"Starting simulation for client {client_id} at "
                     f"{datetime.fromtimestamp(start_time_simulation).strftime(time_format)} "
                     f"with urgency level {urgency_level}")

        client_args.append((
            client_id,
            start_time_simulation,
            abr_mode,
            urgency_level,
            simulation_total_time,
            use_segment_based_collection,
            segment_interval,
            seed_delay,
            startup_seed,
            enable_individual_logs,
            group_client,
            scenario,
            weights,
            score_range,
            js_code,
            model,
            urgency_level_enabled
        ))


    # ==== THREADING EXECUTION MODE ==== 
    if execution_mode == "thread":
        from concurrent.futures import ThreadPoolExecutor
        # executor = ThreadPoolExecutor(max_workers=num_clients)
        # launch = executor.map
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = [
                executor.submit(simulate_dash_js_client, *args)
                for args in client_args
            ]

            for future in futures:
                try:
                    result = future.result()
                    if not result or result.get("status") == "error":
                        client_id = result.get("client_id", "Unknown")
                        logging.warning(f"⚠️ Client {client_id} reported error status.")
                        if STRICT_MODE:
                            raise RuntimeError(f"Simulation aborted due to failure in client {client_id}.")
                        continue
                    
                    client_metrics[result['client_id']] = result["metrics"] #correct format using each client key containing metrics
                    # client_metrics[result['client_id']] = result
                    # client_id = result["client_id"]
                    # client_metrics[client_id] = result["metrics"]

                except Exception as e:
                    logging.error(f"A client simulation raised an exception: {e}")
                    if STRICT_MODE:
                        raise RuntimeError("Simulation aborted due to exception in client execution.")
                    continue

    # ==== PROCESS EXECUTION MODE ====
    elif execution_mode == "process":
        from multiprocessing import Pool
        # executor = Pool(processes=num_clients)
        # launch = executor.starmap
        with Pool(processes=num_clients) as pool:
            # results = pool.starmap(simulate_dash_js_client, client_args)
            results = pool.map(simulate_dash_js_client_wrapper, client_args)
            for result in results:
                # if not result or result.get("status") == "error":
                if not result or result.get("status") != "ok":
                    client_id = result.get("client_id", "Unknown")
                    logging.warning(f"⚠️ Client {client_id} reported error or invalid result.")
                    if STRICT_MODE:
                        raise RuntimeError(f"Simulation aborted due to failure in client {client_id}.")
                    continue
                client_metrics[result['client_id']] = result["metrics"] #correct format using each client key containing metrics
                # client_id = result["client_id"]
                # client_metrics[client_id] = result["metrics"]
    else:
        raise ValueError("Invalid execution mode.")


    logging.info(f"\n ======== SUMMARY RESULT OF SIMULATIONS ======================================================")
    # Resumo de resultados após todas as execuções
    # for client_id, metrics in client_metrics.items():
    #     print(f"Client {client_id} metrics: {metrics}")

    logging.info(f"\n ====>> SUMMARY OF EXECUTION ======")
    success_count = sum(1 for m in client_metrics.values() if m.get("status") == "ok")
    error_count = sum(1 for m in client_metrics.values() if m.get("status") == "error")
    logging.info(f"✔️ Simulation finished: SUCCEEDED: {success_count}, FAILED: {error_count}. \n \n")
    
    # logging.info(f"\n ====>> COMPOSITE QOE SUMMARYZATION ======")
    # for client_id, metrics in client_metrics.items():

    #     logging.info(f"Client {client_id}: composite_qoe = {metrics.get('composite_qoe', 'N/A')}")
    logging.info("🔍 Dumping client_metrics structure (preview of first 3 clients):")
    for client_id in list(client_metrics.keys())[:3]:  # mostra só os 3 primeiros
        logging.info(json.dumps({client_id: client_metrics[client_id]}, indent=2))

    with open("client_metrics_debug.json", "w") as f:
        json.dump(client_metrics, f, indent=2)
    logging.info(f" ========================================================================================= ")

    return client_metrics


# PIPELINE - 2 =====  CALL DASH.JS CLIENT SESSION ==== 

# *** Personalized logging function ***
def setup_client_logging(client_id):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"client_{client_id}.log")
    logger = logging.getLogger(f"client_{client_id}")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Evita múltiplos handlers no mesmo logger
    if not logger.handlers:
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def simulate_dash_js_client_wrapper(args):
    return simulate_dash_js_client(*args)


#=== SIMULATE DASH CLIENT ===
# def simulate_dash_js_client(client_id, start_time_simulation, abr_mode, urgency_level, startup_delay=0, startup_seed = None):

def simulate_dash_js_client(
        client_id, start_time_simulation, abr_mode, 
        urgency_level, simulation_total_time, use_segment_based_collection, 
        segment_interval, seed_delay, startup_seed, enable_individual_logs,
        group_client, scenario, weights, score_range,
        js_code, model_path, urgency_level_enabled
    ):
    """
    Simulate a DASH.js client session with controlled startup delay.
    """

    if js_code is None:
        raise ValueError(f"[Client {client_id}] js_code is None. Ensure it is passed correctly.")

    if enable_individual_logs:
        logger = setup_client_logging(client_id)
        logger.info(f"Client {client_id} logger started.")
    else:
        logger = logging

    if seed_delay > 0:
        logger.info(f"🕒 Client {client_id} waiting {seed_delay}s to start...")
        time.sleep(seed_delay)

    logger.info(f"Client {client_id} started simulation at "
                f"{datetime.fromtimestamp(start_time_simulation).strftime(time_format)}.")

    # === Load model locally if required ===
    model = None
    if model_path and "PENSIEVE" in abr_mode.upper():
        from stable_baselines3 import PPO
        try:
            model = PPO.load(model_path)
            logger.info(f"Client {client_id}: RL model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Client {client_id}: Failed to load model from {model_path} - {e}")
            return {"client_id": client_id, "status": "error", "metrics": None}

    logging.info(f"Inside simulation Dash client -> Client {client_id} with ABR mode: {abr_mode}, urgency={urgency_level}")

    # === Setup browser ===
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument(f"user-agent=client-{client_id}")
    chrome_options.add_argument(f"--user-data-dir=/tmp/chrome-profile-{client_id}")
    chrome_options.add_argument("--enable-logging")
    chrome_options.add_argument("--v=1")

    chrome_binary_path = '/usr/bin/chromium'
    chromedriver_path = "/usr/bin/chromedriver"
    chrome_options.binary_location = chrome_binary_path

    driver = None

    try:
        driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)
        driver.set_page_load_timeout(pageTimeout)
        driver.set_script_timeout(scriptTimeout)

        # == Access player page ==
        retries, success = 3, False
        for attempt in range(retries):
            try:
                logger.info(f"Client-{client_id}: Attempting to access {player_url} (Attempt {attempt+1}/{retries})")  
                driver.get(player_url)
                logger.info(f"Client-{client_id}: Page loaded successfully")
                success = True
                break
            except Exception as e:
                logger.warning(f"Client-{client_id}: Error loading page (Attempt {attempt+1}/{retries}) - {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    raise

        if not success:
            return {"client_id": client_id, "status": "error", "metrics": None}

        logger.info(f"Client-{client_id}: Showing informations: \n URL page {driver.current_url} ")

        # == Configure player ==
        dash_settings = configure_dash_player(abr_mode)
        driver.execute_script(js_code + f"applyDashSettings(player, {json.dumps(dash_settings)}); injectPlayerEventListeners();")

        # == Wait for video element ==
        video_element = WebDriverWait(driver, playback_wait).until(
            EC.presence_of_element_located((By.TAG_NAME, "video"))
        )
        logger.info(f"Client {client_id}: video element found on page.")

        playback_started = start_playback(driver, video_element, client_id)
        if not playback_started:
            logger.error(f"Client {client_id}: Playback did not start. Aborting client session.")
            return {"client_id": client_id, "status": "error", "metrics": None}

        formatted_start_time_simulation = datetime.fromtimestamp(start_time_simulation).strftime(time_format) 
        client_metrics[client_id] = initialize_client_metrics(
            client_id, date, formatted_start_time_simulation,
            startup_seed, group_client, urgency_level, abr_mode,
        )

        metrics = measure_qoe(
            driver, client_id, simulation_total_time, 
            start_time_simulation, video_element, model, weights,
            client_metrics, score_range,
            use_segment_based_collection=use_segment_based_collection,
            segment_interval=segment_interval
        )

        metrics["client_id"] = client_id
        return {"client_id": client_id, "status": "ok", "metrics": metrics}

    except Exception as e:
        logger.error(f"Cliente {client_id}: erro durante a simulação - {e}")
        return {"client_id": client_id, "status": "error", "metrics": None}

    finally:
        if driver:
            driver.quit()
        logger.info(f"Client-{client_id} simulation finished.")


#=== SIMULATE DASH CLIENT - OLD VERSION FOR REFERENCE ===
def simulate_dash_js_client_old(
        client_id, start_time_simulation, abr_mode, 
        urgency_level, simulation_total_time, use_segment_based_collection, 
        segment_interval, seed_delay, startup_seed, enable_individual_logs,
        group_client, scenario, weights, score_range,
        js_code, model, urgency_level_enabled
        ):
    """
    Simulate a DASH.js client session with controlled startup delay.

    Args:
        client_id (int): ID of the client.
        start_time_simulation (str): Timestamp of when the simulation starts.
        abr_mode (str): ABR algorithm mode used for the simulation.
        urgency_level (int): Urgency level assigned to this client.
        seed_delay (float): Delay before starting the streaming session (seconds).
        startup_seed (int): Seed used for startup delay generation (for traceability).

    Returns:
        dict: Dictionary containing collected metrics for this client.
    """

    if js_code is None:
        raise ValueError(f"[Client {client_id}] js_code is None. Ensure it is passed correctly to the function.")

    if enable_individual_logs:
        logger = setup_client_logging(client_id)
        logger.info(f"Client {client_id} logger started.")
    else:
        logger = logging

    if seed_delay > 0:
        logger.info(f"🕒 Client {client_id} waiting {seed_delay}s to start...")
        time.sleep(seed_delay)

    logger.info(f"Client {client_id} started simulation at "
                 f"{datetime.fromtimestamp(start_time_simulation).strftime(time_format)}.")

    logging.info(f"Inside simulation Dash client -> Starting simulation for Client {client_id} with ABR mode: {abr_mode} / model: {model} and urgency level: {urgency_level}")
    
    # Configurar as opções do navegador
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless")  # Adicione esta linha para rodar em modo headless
    # chrome_options.add_argument("--remote-debugging-port=9222")  # Não usar com mais de 1 client
    chrome_options.add_argument("--disable-gpu")  # Adicione esta linha
    chrome_options.add_argument('--disable-software-rasterizer')
    chrome_options.add_argument(f"user-agent=client-{client_id}")
    chrome_options.add_argument(f"--user-data-dir=/tmp/chrome-profile-{client_id}") #Client profile dedicated and isolated

    # Adicionar a opção para capturar os logs de console
    chrome_options.add_argument("--enable-logging")
    chrome_options.add_argument("--v=1")  # Define o nível de verbosidade dos logs

    # WEBDRIVER LINUX CONFIGURATION PATH
    chrome_binary_path = '/usr/bin/chromium' #linux mode
    # chrome_binary_path = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' #macOS mode

    # Configura o caminho do chromedriver no script [para uso no docker Client]
    chromedriver_path = "/usr/bin/chromedriver" 
    chrome_options.binary_location = chrome_binary_path

    # Inicializar o WebDriver
    driver = None

    try:
        service=Service(chromedriver_path) # Inicialização do ChromeDriver 
        driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options) #linux mode
        # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) #MacOs Mode
        
        # == Define explicit timeouts == 
        driver.set_page_load_timeout(pageTimeout)  # Timeout para carregamento da página
        driver.set_script_timeout(scriptTimeout)  # Timeout para execução de scripts
        
        # == Try access player page == 
        retries = 3
        success = False
        for attempt in range(retries):
            try:
                logger.info(f" ============================== \n Client-{client_id}: Attempting to access {player_url} (Attempt {attempt+1}/{retries})")  
                driver.get(player_url)
                logger.info(f"Client-{client_id}: Page loaded successfully")
                success = True
                break
                # print("endereço requisitado - ",player_url)
                # logging.info("Página carregada -> : %s", player_url)
                # print("Exibindo informações da página: ")
                # print("Título da página:", driver.title)
                # print("URL da página:", driver.current_url)
                # logging.info(f"Opening URL for Client-{client_id}.")
            except Exception as e:
                logger.warning(f"Client-{client_id}: Error loading page (Attempt {attempt+1}/{retries}) - {e}")
                if attempt < retries - 1:
                    time.sleep(5)  # Esperar antes de tentar novamente
                else:
                    raise  # Se todas as tentativas falharem, lança a exceção

        logger.info(f"Client-{client_id}: Showing informations: \n URL page {driver.current_url} ")

        # == LOADING JS FUNCTIONS == 
        logger.info(f"Client-{client_id}: Loading JavaScript functions and applying configureDashPlayer")
        # Decide settings do player no Python
        dash_settings = configure_dash_player(abr_mode)

        # Injeta no browser
        driver.execute_script(js_code + f"applyDashSettings(player, {json.dumps(dash_settings)}); injectPlayerEventListeners();")
        # driver.execute_script(js_code + f"configureDashPlayer(player, '{abr_mode}'); injectPlayerEventListeners();")
        
        
        try:    
            # Wait until video be present in the page
            video_element = WebDriverWait(driver, playback_wait).until(EC.presence_of_element_located((By.TAG_NAME, "video")))
            logger.info(f"Cliente {client_id}: vídeo encontrado na página.")
            
            # driver.execute_script("window.playClickedAt = performance.now(); player.play();") #executing play in video container
            # logging.info(f"Client {client_id}: Playback started! \n ==============================")
            playback_started = start_playback(driver, video_element, client_id)
            if not playback_started:
                logger.error(f"Client {client_id}: Playback did not start. Aborting client session.")
                return  # ou encerrar este client de forma segura

            start_time = time.time()            

            # driver.execute_script("window.playClickedAt = performance.now(); player.play();")
            
            # CHECK STARTUP DELAY FUNCTION
            result = driver.execute_script("return typeof window.startupDelay !== 'undefined';")
            logger.info(f"Client {client_id}: Is startupDelay initialized? {result}")
            



        except Exception as e:
            logger.error(f"Client-{client_id}: Video element not found - {e}")
            return  # Sai da função caso não encontre o vídeo
        
        # Start playback and colect metrics
        # video_element.click()
        # logging.info(f"Client-{client_id}: Playback started! \n ==============================")

        # logging.info(f"Cliente {client_id}: Iniciando reprodução!")
        # time.sleep(simulation_total_time)

        # === Initializing Client Metrics ===
        formatted_start_time_simulation = datetime.fromtimestamp(start_time_simulation).strftime(time_format) 

        client_metrics[client_id] = initialize_client_metrics(
            client_id,
            date, 
            formatted_start_time_simulation, 
            startup_seed, 
            group_client, 
            urgency_level, 
            abr_mode,
        )

        # COLLECTING STARTUP DELAY
        # raw_startup_delay = driver.execute_script("return getStartupDelay();")
        # if raw_startup_delay is not None:
        #     formatted_startup_delay = round(raw_startup_delay / 1000, 2)
        #     client_metrics[client_id]["startup_delay"] = formatted_startup_delay
        #     logger.info(f"Client {client_id}: Collected startup delay (s): {formatted_startup_delay}")
        # else:
        #     client_metrics[client_id]["startup_delay"] = None

        # metrics = measure_qoe(driver, client_id, simulation_total_time, start_time_simulation,video_element,group_client, model, abr_mode,weights,client_metrics, urgency_level, startup_seed)
        # metrics = measure_qoe(driver, client_id, simulation_total_time, start_time_simulation, video_element, model, weights, client_metrics, score_range)
        metrics = measure_qoe(
            driver, client_id, simulation_total_time, 
            start_time_simulation, video_element, model, weights,
            client_metrics, score_range, use_segment_based_collection=use_segment_based_collection, 
            segment_interval=segment_interval  # coleta a cada 3 segmentos
        )

        # Including client Id inside Saved data
        metrics["client_id"] = client_id

        # return metrics
        return {
            "client_id": client_id,
            "status": "ok",
            "metrics": metrics
        }
    
        
    except Exception as e:
        logging.error(f"Cliente {client_id}: erro durante a simulação - {e}")
        return {
            "client_id": client_id,
            "status": "error",
            "metrics": None
        }
    finally:
        if driver:
            driver.quit()
            # end_time_simulation = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            # logging.info(f"Client {client_id} simulation finalized at {end_time_simulation}")
        logger.info(f"Client-{client_id} simulation finished.")

# PIPELINE - x =====  COLLECT METRICS FOR EACH CLIENT SESSION AND CALCULATE ALL METRICS ==== 
# Function for collect metrics and measure QoE
# MEASURE_QOE SUBFUNTIONS [PUT IN SEPARATE FILE LATER]
# ===========================================================================================
def check_player_ready(driver, client_id, video_element):
    # Validate Player readiness and segment availability
    logging.info(f"== inside CHECK PLAYER_READY routine === ")
    start_wait = time.time()
    while True:
        try:
            is_ready = driver.execute_script("return typeof player !== 'undefined' && player.isReady();")
            if is_ready:
                logging.info(f"Client {client_id}: Player is ready.")
                break
        except Exception as e:
            logging.debug(f"Client {client_id}: Player readiness check failed: {e}")
        if time.time() - start_wait > 15:
            logging.error(f"Client {client_id}: Timeout waiting for player readiness.")
            return
        time.sleep(0.5)


def is_simulation_running(start_time_simulation, simulation_total_time):
    logging.info(f"== inside (IS SIMULATION RUNNING) routine === ")
    # print(f"[DEBUG] start_time_simulation type: {type(start_time_simulation)} -> value: {start_time_simulation}")
    # logging.info(f"[DEBUG] Inside IS_SIMULATION_RUNNING | start_time_simulation type={type(start_time_simulation)} | value={start_time_simulation}")
    # logging.info(f"[DEBUG] simulation_total_time type={type(simulation_total_time)} | value={simulation_total_time}")
    current_time = time.time()
    return (current_time - start_time_simulation) < simulation_total_time


def is_playing(driver, video_element):
    try:
        return driver.execute_script("return !arguments[0].paused && !arguments[0].ended && arguments[0].readyState > 2;", video_element)
    except Exception as e:
        logging.error(f"Error checking if video is playing: {e}")
        return False


def start_playback(driver, video_element, client_id):
    logging.info(f"=== Inside START PLAYBACK ROUTINE ===")

    try:
        logging.info(f"Client {client_id}: Checking if PLAYER instance is ready ...")
        player_exists = driver.execute_script("return typeof player !== 'undefined';")
        if not player_exists:
            logging.error(f"Client {client_id}: DASH player instance not found!")
            return False

        logging.info(f"Client {client_id}: Attempting to start playback...")
        
        # Marca o tempo de início and executes Play
        driver.execute_script("window.playClickedAt = performance.now(); arguments[0].play();", video_element)

        # Aguarda até o vídeo realmente estar rodando
        wait_playback_start = time.time()
        WebDriverWait(driver, 80).until(
            lambda d: d.execute_script(
                "return !arguments[0].paused && !arguments[0].ended && arguments[0].readyState > 2;",
                video_element
            )
        )
        wait_playback_end = time.time()
        logging.info(f"Client-{client_id}: Waited {wait_playback_end - wait_playback_start:.2f}s for playback to start.")


        logging.info(f"Client {client_id}: Playback confirmed as running!")
        return True

    except Exception as e:
        logging.error(f"Client {client_id}: Playback could not start - {e}")
        return False


def measure_qoe(driver, client_id, simulation_total_time, start_time_simulation, video_element,
                model, weights, client_metrics, score_range, use_segment_based_collection=True, segment_interval=1):
    
    logging.info(f"== inside MEASURE QOE routine === ")
    client_info = client_metrics[client_id]
    group_client = client_info['group_client']
    urgency_level = client_info['urgency_level']
    startup_seed = client_info['startup_seed']
    abr_mode = client_info['abr_mode']

    try:
        # == 1. Check initial Loop
        # Validate Player readiness and segment availability
        check_player_ready(driver, client_id, video_element)

        # == 2. Metric Collect Loop =========
        logging.info(f"== inside Metric collect loop === ")
        
        # == Capture Startup Delay once ==
        logging.info(f"== Attempting to capture startup delay for Client {client_id} ==")
        try:
            wait_start = time.time()
            max_wait = 5  # seconds
            startup_delay_sec = None

            while time.time() - wait_start < max_wait:
                # raw_startup_delay = driver.execute_script("return getStartupDelay();")
                raw_startup_delay = driver.execute_script("return window.startupDelay;")
                # logging.info(f"Startup delay measured by WINDOW.STARTUPDELAY - > {raw_startup_delay}")
                
                numeric_value = None

                if isinstance(raw_startup_delay, (int, float)):
                    # Caso raro: número puro (provavelmente ms)
                    numeric_value = raw_startup_delay / 1000
                elif isinstance(raw_startup_delay, dict):
                    if "delay" in raw_startup_delay:
                        numeric_value = raw_startup_delay["delay"]

                if numeric_value is not None and numeric_value > 0:
                    startup_delay_sec = round(numeric_value, 3)
                    logging.info(f"✅ Startup delay extracted -> {startup_delay_sec} s")
                    break

                time.sleep(0.2)

            if startup_delay_sec is not None:
                client_metrics[client_id]["startup_delay"] = startup_delay_sec
                logging.info(f"Client {client_id}: Collected Startup Delay (s): {startup_delay_sec}")
            else:
                client_metrics[client_id]["startup_delay"] = None
                logging.warning(f"Client {client_id}: Startup Delay not available or returned 0 even after waiting.")

        except Exception as e:
            logging.error(f"Client {client_id}: Error capturing startup delay - {e}")
            client_metrics[client_id]["startup_delay"] = None


        if use_segment_based_collection:
            last_segment_counter = -1
            start_time = time.time()
            end_time = start_time + simulation_total_time

            while time.time() < end_time:
                try:
                    # Interrompe se a reprodução terminou
                    # playback_ended = driver.execute_script("return player.isEnded();")
                    # if playback_ended:
                    #     logging.info(f"Client {client_id}: Playback has ended. Exiting segment-based loop.")
                    #     break

                    segment_counter = driver.execute_script("return window.segmentCounter;")
                    if segment_counter > 0 and segment_counter != last_segment_counter and segment_counter % segment_interval == 0:
                        playback_metrics = collect_playback_metrics(driver, video_element, client_id)
                        if playback_metrics is None:
                            logging.warning(f"Client {client_id}: No playback metrics collected at this segment.")
                            continue

                        # == 3. Process Metric and Update Client_metric data =========
                        # Metric calculation
                        relative_time = time.time() - start_time
                        client_metrics[client_id]["timestamp_samples"].append(round(relative_time,2))
                        process_metrics(playback_metrics, client_id, client_metrics)
                        
                        # logging.info(f"Trying recalculate AVG_BUFFER Level")
                        # client_metrics[client_id]['avg_buffer_level'] = calculate_avg_buffer_level(client_id, client_metrics)



                        # == 4. Select bitRate from ABR_MODE ========== 
                        logging.info(f"🎯 Showing ABR-model and Model Before apply_qualityL /n Client {client_id}: ABR Mode selected -> {abr_mode} \n MODEL selected -> {model}")
                        # Apply quality selection and lead with fallback
                        quality = select_bitrate(
                            client_id, group_client, client_metrics, abr_mode, urgency_level,urgency_level_enabled,
                            score_range, model, shared_dir, weights
                        )

                        # == 5. Apply selected Quality
                        # Compute agreggated Values related to QoE and Fairness
                        apply_quality(driver, quality, client_id, abr_mode)

                        last_segment_counter = segment_counter

                except Exception as e:
                    logging.error(f"Client {client_id}: Error during segment-based QoE measurement: {e}")

                time.sleep(0.3)

        else:
            while is_simulation_running(start_time_simulation, simulation_total_time):            
                playback_metrics = collect_playback_metrics(driver, video_element, client_id)
                # if playback_metrics is None:
                #     logging.warning(f"Client {client_id}: No playback metrics collected at this cycle.")
                #     continue

                # === SHOWING Playback metrics Keys === 
                # if not playback_metrics:
                #     logging.error(f"🚨 Client {client_id}: Empty playback_metrics at this cycle!")
                # else:
                #     logging.info(f"✅ Client {client_id}: Playback_metrics keys: {list(playback_metrics.keys())}")

                # == 3. Process Metric and Update Client_metric data =========
                # Metric calculation

                process_metrics(playback_metrics, client_id, client_metrics)
                # update_client_metrics_snapshot(client_id, client_metrics, snapshot_data)
                # update fairness score
                # publish local and global fairness score

                # == 4. Select bitRate from ABR_MODE ========== 
                # Apply quality selection and lead with fallback
                # quality = select_bitrate(client_id, client_metrics, state, model)
                # quality = select_bitrate(
                #     client_id, client_metrics, abr_mode, urgency_level,
                #     score_range, model, shared_dir=shared_dir, weights=weights
                # )

                quality = select_bitrate(
                            client_id, group_client, client_metrics, abr_mode, urgency_level,urgency_level_enabled,
                            score_range, model, shared_dir, weights
                        )

                # == 5. Apply selected Quality
                # Compute agreggated Values related to QoE and Fairness
                # state = compute_fairness_and_update_state(client_id, client_metrics, weights)
                # logging.info(f"[====== Checking STATE value for Client {client_id}] -> State returned: {state} =====")
                apply_quality(driver, quality, client_id, abr_mode)
                # logging.info(f"[====== Checking QUALITY SELECTED value returned for Client {client_id}] -> QUALITY returned: {quality} =====")

                time.sleep(0.1)  # Sleep para evitar busy-waiting

        client_metrics[client_id]["status"] = "ok"

    except Exception as e:
        logging.error(f"Client {client_id}: Error during simulation - {e}")
        client_metrics[client_id]["status"] = "error"
        client_metrics[client_id]["error_message"] = str(e)
        return client_metrics[client_id]

    finally:
        # == 6 Finalize and save == 
        # Update timestamps, consolidate arrays, save snapshots and publish scores
        logging.info(f"== Finalizing simulation for Client {client_id} === ")
        # client_metrics[client_id]['avg_buffer_level'] = calculate_avg_buffer_level(client_id, client_metrics)
        finalize_and_save(client_id, client_metrics, start_time_simulation,shared_dir)

    return client_metrics[client_id]

# ===========================================================================================

def finalize_and_save(client_id, client_metrics, start_time_simulation, shared_dir):
    end_time_simulation = datetime.now().strftime(time_format)
    client_metrics[client_id]['end_time_simulation'] = end_time_simulation
    client_metrics[client_id]['total_playback_time'] = time.time() - start_time_simulation

    publish_client_score(
        client_id=client_id,
        client_metrics=client_metrics,
        abr_mode=client_metrics[client_id]['abr_mode'],
        group_client=client_metrics[client_id]['group_client'],
        shared_dir=shared_dir
    )

    logging.info(f"Client {client_id}: Simulation finalized at {end_time_simulation}")


# --- DATA EXPORT ---
def export_qoe_metrics_to_json(client_metrics, filename):
    with open(filename, 'w') as json_file:
        json.dump(client_metrics, json_file, indent=4)



def export_qoe_metrics_to_csv(client_metrics, filename, metrics_to_exclude=None):
    import csv
    import logging

    metrics_to_exclude = metrics_to_exclude or []

    # Coleta os campos de todos os dicionários diretamente
    fields = set()
    for metrics in client_metrics.values():
        fields.update(metrics.keys())

    # Remove métricas que não queremos exportar
    fields = [f for f in fields if f not in metrics_to_exclude]

    # Garante que client_id será a primeira coluna
    fields = ['client_id'] + [f for f in sorted(fields) if f != 'client_id']

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for client_id, metrics in client_metrics.items():
            row = {}

            for field in fields:
                value = metrics.get(field, "")
                if isinstance(value, list):
                    row[field] = ','.join(map(str, value))
                else:
                    row[field] = value

            missing = [f for f in fields if f not in metrics]
            if missing:
                logging.warning(f"⚠️ Client {client_id} missing fields: {missing}")

            writer.writerow(row)


# == TEST FUNCTION ONLY FOR INDIVIDUAL TESTS ==
def export_data_to_csv(data, filename):
    if not data:
        raise ValueError("Error... The data array is empty.")

    # Obter os campos a partir do primeiro elemento do array
    fields = list(data[0].keys())

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for entry in data:
            writer.writerow(entry)

def export_fairness_metrics_to_csv(fairness_metrics, filename):
    if not fairness_metrics:
        raise ValueError("The aggregated_metrics dictionary is empty.")

    # Preparar dados para escrita no CSV
    # Cria uma lista de dicionários, onde cada dicionário representa uma métrica com seus valores
    data_to_export = []

    # Encontrar o comprimento máximo entre todas as listas de métricas para padronizar o número de linhas
    # max_length = max(len(values) for values in fairness_metrics.values())

    # Preencher os dados padronizando o número de entradas para cada métrica
    # for i in range(max_length):
    row = {}
    for metric, value in fairness_metrics.items():
        # if i < len(values):
        row[metric] = value
        # else:
            # row[metric] = None  # Use None para padronizar o comprimento
    data_to_export.append(row)

    # Obter os campos a partir das chaves do primeiro dicionário
    fields = list(data_to_export[0].keys())

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data_to_export)


def format_number(value, decimal_places=4):
    try:
        formatted_value = round(value, decimal_places)
        return formatted_value
    except (TypeError, ValueError):
        return value  # Return original value if some error occouring


# ============= MAIN CODE ==================



# ==== MAIN OPTIONS FOR MENU ======

group_client_options = [
    ('G1'),
    ('G2'),
    ('G3'),
    ('TEST-ONLY'),
    ('DATASET-DATA')
]
# USE_RL = True #Alternate between RL (True) and Heuristic (False)
abr_mode = ""

scenario_options = [
    ('Scenario 1'),
    ('Scenario 2'),
    ('Scenario 3'),
    ('TEST-ONLY'),
    ('DATASET-DATA')
]

# PROCESSING MODE EXECUTION
execution_mode = "thread"
enable_individual_logs = True

# MODL CONFIGURATION
model_path = None
env_config = None

scenario_number = 1 # default value
logging.basicConfig(level=logging.INFO)

weights = load_weights_from_json("weights_config.json", default_weights=default_weights)
print("Weights loaded successfully:", weights)

# ==== ***** ======
opcao = ""
use_segment_based_collection = True
num_clients = 20
simulation_total_time = 400
group_client=0 # 'G1'
scenario='TEST-ONLY'
abr_mode="PENSIEVE-MULTI-FAIRNESS-AWARE"
use_segment_based_collection="ON"
urgency_level_enabled = True
segment_interval = 3
startup_seed=42


# === AUTOMATIC EXECUTION MODE (Container-based Simulation) ===
# Detect if environment variables are present
client_id_env = os.getenv("CLIENT_ID")
abr_mode_env = os.getenv("ABR_MODE")
execution_mode_env = os.getenv("EXECUTION_MODE")
sim_time_env = os.getenv("SIMULATION_TIME")
num_clients_env = os.getenv("NUM_CLIENTS")
group_client_env = os.getenv("GROUP_CLIENT", "TEST-ONLY")
scenario_env = os.getenv("SCENARIO", "TEST-ONLY")

if client_id_env and abr_mode_env:
    print(f"\n[AUTO MODE] Detected container environment for Client {client_id_env}")
    print(f"ABR_MODE={abr_mode_env}, EXEC_MODE={execution_mode_env}, SIM_TIME={sim_time_env}, NUM_CLIENTS={num_clients_env}\n")

    # Convert numeric values safely
    simulation_total_time = int(sim_time_env) if sim_time_env else 300
    num_clients = int(num_clients_env) if num_clients_env else 1

    # Fast preset values (compatible with your defaults)
    segment_interval = 3
    startup_seed = 42
    use_segment_based_collection = True
    enable_individual_logs = True
    urgency_level_enabled = True
    weights = load_weights_from_json("weights_config.json", default_weights=default_weights)

    # Load RL model if required
    model_path, env_config = load_model_and_env_config(abr_mode_env)

    # Run directly without showing the menu
    client_metrics = run_simulations(
        num_clients=num_clients,
        simulation_total_time=simulation_total_time,
        group_client=group_client_env,
        scenario=scenario_env,
        abr_mode=abr_mode_env,
        use_segment_based_collection=use_segment_based_collection,
        segment_interval=segment_interval,
        startup_seed=startup_seed,
        execution_mode=execution_mode_env or "process",
        enable_individual_logs=enable_individual_logs,
        weights=weights,
        model_path=model_path
    )

    # Export result for this container
    export_qoe_metrics_to_csv(
        client_metrics,
        f'/simulations/output/simulation-client-{client_id_env}-ABR-{abr_mode_env}-datetime-{date}-{time_now}.csv'
    )

    print(f"[AUTO MODE] Simulation finished for Client {client_id_env}. Exiting container.")
    exit(0)

# ==== ***** ======
# === INTERACTIVE MODE - SHOW MENU ===

while opcao !=0:
    print("========================================================")
    print("Option 1: Network Group Client [currently: {}]".format(group_client))
    print("Option 2: Number of clients for simulate [currently: {}]".format(num_clients))
    print("Option 3: Simulation Time duration [currently: {}]".format(simulation_total_time))
    print("Option 4: Select ABR Mode simulation (DASH.js, QoE-Based, RL, BOLA... ) [currently: {}]".format(abr_mode))
    print("Option 5: Scenario simulation number (Sceario 1, 2, 3 or 4: Test-Only)[currently: {}]".format(scenario))
    print("Option 6: Enable/Disable QoE Weights (1: Enable, 2: Disable) [currently: {}]".format("Enable" if USE_WEIGHTS else "Disable"))
    print("Option 7: Define seed distribution [currently: {}]".format(startup_seed))
    print("Option 8: Toggle Segment-based Collection [currently: {}]".format("ON" if use_segment_based_collection else "OFF"))
    print("Option 9: Set Urgency level mode [currently: {}]".format("ON" if urgency_level_enabled else "OFF"))
    print("Option 10: Set currently segment collection interval [currently: {}]".format(segment_interval))
    print("Option 11: Select Execution Mode [currently: {}]".format("Threading" if execution_mode == "thread" else "Multiprocessing"))
    print("Option 12: Enable/Disable Individual Client Logs [currently: {}]".format("ON" if enable_individual_logs else "OFF"))
    print("Option 13: Start Manual Simulation")
    print("Option 14: Start Automated Simulation with fixed parameters")
    print("Option 0: Exit Simulation")
    opcao = input("Select option: ")

    if opcao == "1":
        print("Select Group Client User:")
        for idx, (group) in enumerate(group_client_options, start=1):
            print(f"Group: {idx}  - {group}")
        try:
            condition_selected = int(input("Select number of Group Client: "))
            if 1 <= condition_selected <= len(group_client_options):
                # Obter os valores de acordo com a condição escolhida
                group_client = group_client_options[condition_selected - 1]
                print(f"\n\n Group {group_client} selected.")
            else:
                print("Option out of range. Try a valid option.")
        except ValueError:
            print("Invalid option. Please, insert a number.")
        print("\n\n")

    elif opcao == "2":
        num_clients = int(input("Inform number of clients to simulate:"))
        print(f"\n Number of clients selected to simulate: {num_clients} \n")
        
    elif opcao == "3":
        simulation_total_time = int(input("Inform Time simulation duration: [type 0 to use default 400 sec]: "))
        print(f"\n Simulation configurated for {simulation_total_time} seconds \n")
        if simulation_total_time == 0:
            simulation_total_time = 400

    elif opcao == "4":
        print("\nSelect ABR Strategy Mode:")
        print("1 - DASH.js ABR (DEFAULT DASH - Heuristic mode)")
        print("2 - DASH.js QoE-Based  (Heuristic mode)")
        print("3 - BOLA (Buffer Occupancy-based Lyapunov Algorithm)")
        print("4 - HOTDASH (Hot Data-Aware Heuristic ABR)")
        print("5 - L2A (Learn2Adapt — dash.js ABR)")
        print("6 - PENSIEVE (Baseline RL)")
        print("7 - QoE-Based RL")
        print("8 - Fairness Aware RL")
        print("9 - Multi Fairness Aware RL")

        try:
            mode_selected = int(input("Select the mode: "))
            if mode_selected == 1:
                abr_mode = "DASH-DEFAULT"
                model = None
                print("\nABR mode set to DASH.js Default.\n")

            elif mode_selected == 2:
                abr_mode = "DASH-QOE-BASED"
                model = None
                print("\nABR mode set to DASH QoE-based Heuristic.\n")

            elif mode_selected == 3:
                abr_mode = "BOLA"
                model = None
                print("\nABR mode set to BOLA.\n")

            elif mode_selected == 4:
                abr_mode = "HOTDASH"
                model = None
                print("\nABR mode set to HOTDASH (Hot Data-Aware Heuristic ABR).\n")

            elif mode_selected == 5:
                abr_mode = "L2A"
                model = None
                print("\nABR mode set to L2A (Learn2Adapt) — dash.js ABR.\n")

            elif mode_selected == 6:
                abr_mode = "PENSIEVE-BASELINE"
                # model = PPO.load("./rl-model/pensieve-baseline-model.zip")
                model_path = "./rl-model/pensieve-baseline-model.zip"
                print("\nABR mode set to PENSIEVE BASELINE.\n")

            elif mode_selected == 7:
                abr_mode = "PENSIEVE-QOE-BASED"
                model_path = "./rl-model/pensieve-qoe-based-model.zip"
                print("\nABR mode set to QOE-Based RL.\n")

            elif mode_selected == 8:
                abr_mode = "PENSIEVE-FAIRNESS"
                # model = PPO.load("./rl-model/pensieve_weights_qoe_fairness_model_11.zip")
                # model = PPO.load("./rl-model/pensieve-fairness-model.zip")
                model, env_config = load_model_and_env_config(abr_mode)
                print(f"\nABR mode set to {abr_mode}.\n")

            elif mode_selected == 9:
                abr_mode = "PENSIEVE-MULTI-FAIRNESS-AWARE"
                # model, env_config = load_model_and_env_config(abr_mode)
                model_path, env_config = load_model_and_env_config(abr_mode)
                logging.info(f"==== Model Check — selected model path: {model_path}")
                if env_config:
                    logging.info(f"==== Env Config Keys: {list(env_config.keys())}")
                
                # logging.info(f"==== Model Check — observation space at select_bitrate: {model.observation_space}")
                print(f"\nABR mode set to {abr_mode}.\n")

            else:
                print("Invalid option. Please select a valid mode.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    elif opcao == "5":
        print("Select Scenario:")
        for idx, scenario in enumerate(scenario_options, start=1):
            print(f"Scenario {idx}: {scenario}")
        try:
            scenario_selected = int(input("Select Scenario Number: "))
            if 1 <= scenario_selected <= len(scenario_options):
                scenario_number = scenario_selected
                print(f"\n Scenario {scenario_number} selected.\n")
            else:
                print("Option out of range. Try a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    elif opcao == "6":
            weight_option = input("Use QoE Weights? (1: Yes, 2: No): ")
            if weight_option == "1":
                USE_WEIGHTS = True
            else:
                USE_WEIGHTS = False
            print(f"QoE Weights {'Enabled' if USE_WEIGHTS else 'Disabled'}.")

    elif opcao == "7":
        seed_input = input("Enter startup seed (integer) or leave blank to disable [default for experiment -> 42]: ")
        if seed_input.strip() == "":
            startup_seed = None
            print("🆗 Startup seed disabled (random initialization).")
        else:
            try:
                startup_seed = int(seed_input)
                print(f"Startup seed set to {startup_seed}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

    if opcao == "8":
        use_segment_based_collection = not use_segment_based_collection
        print(f"Segment-based collection {'enabled' if use_segment_based_collection else 'disabled'}. /n")

    if opcao == "9":
        urgency_option = input("Use Urgency Level Policy? (1: Yes, 2: No): ")
        if urgency_option == "1":
            urgency_level_enabled = True
        else:
            urgency_level_enabled = False
        print(f"Urgency Level Policy {'Enabled' if urgency_level_enabled else 'Disabled'}.")

    elif opcao == "10":
        try:
            interval = int(input("Enter segment collection interval (e.g., 1 = every segment, 3 = every 3 segments): "))
            if interval >= 1:
                segment_interval = interval
                print(f"Segment collection interval set to {segment_interval}.")
            else:
                print("⚠️ Interval must be >= 1.")
        except ValueError:
            print("⚠️ Invalid input. Please enter an integer.")

    elif opcao == "11":
        exec_mode_input = input("Select Execution Mode: (1) Threading, (2) Multiprocessing: ")
        if exec_mode_input == "1":
            execution_mode = "thread"
        elif exec_mode_input == "2":
            execution_mode = "process"
        else:
            print("⚠️ Invalid option. Defaulting to Threading.")
            execution_mode = "thread"
        print(f"✅ Execution mode set to: {'Threading' if execution_mode == 'thread' else 'Multiprocessing'}")


    elif opcao == "12":
        log_input = input("Enable Individual Logs per Client? (1) Yes, (2) No: ")
        if log_input == "1":
            enable_individual_logs = True
        else:
            enable_individual_logs = False
        print(f"📝 Individual logs per client {'enabled' if enable_individual_logs else 'disabled'}.")

    elif opcao == "13":
        print("STARTING MANUAL APPLICATION")

        # Inicializa por segurança
        # model = None
        # env_config = None

        # Modos não-RL (player-driven ou heurísticos próprios)
        non_rl_modes = ["DASH-DEFAULT", "DASH-QOE-BASED", "BOLA", "HOTDASH", "L2A"]

        if abr_mode in non_rl_modes:
            # Não carrega modelo para modos não-RL
            # model = None
            # env_config = None
            logging.info(f"Selected non-RL mode '{abr_mode}'. Using no RL model.")
        else:
            # Modos RL carregam modelo e config
            # model, env_config = load_model_and_env_config(abr_mode)
            if 'model' not in globals() or model is None:
                # model, env_config = load_model_and_env_config(abr_mode)
                model_path, env_config = load_model_and_env_config(abr_mode)
                logging.info(f"Model: {model_path} and ABR-Mode: '{abr_mode}' selected.")
            else:
                logging.info("Model já foi configurado anteriormente — preservando o model existente.")

        # ---- Execução da simulação ----
        client_metrics = run_simulations(
            num_clients,
            simulation_total_time,
            group_client,
            scenario,
            abr_mode,
            use_segment_based_collection,
            segment_interval,
            startup_seed,
            execution_mode=execution_mode,               # Experiment 05 parameter
            enable_individual_logs=enable_individual_logs, # Experiment 05 parameter
            weights=weights,
            # model=model
            model_path=model_path #adjusting for multiprocessing
            # Se sua run_simulations aceitar, você pode passar model/env_config explicitamente:
            # model=model, env_config=env_config
        )

        # === Exportar resultados ===
        print("Generating CSV file")
        export_qoe_metrics_to_csv(
            client_metrics,
            f'./simulations/simulation-ABR-{abr_mode}-group-scenario-{scenario_number}-'
            f'group-client-{group_client}-numclients-{num_clients}-seed-number-{startup_seed}-simulation-time-'
            f'{simulation_total_time}-datetime-{date}-{time_now}-executionmode-{execution_mode}.csv'
        )

        # logging.info("Generating Fairness graphs")
        # export_fairness_metrics_to_csv(...)

    elif opcao == "14":
        print("STARTING AUTOMATED APPLICATION")
        abr_mode = "PENSIEVE-MULTI-FAIRNESS-AWARE"
        model, env_config = load_model_and_env_config(abr_mode)
        # logging.info(f"==== Model Check — observation space at select_bitrate: {model.observation_space}")
        
        # FAST CONFIGURATION OPTIONS
        num_clients = 7
        simulation_total_time = 100
        group_client=3 # 'TEST-ONLY'
        scenario='TEST-ONLY'
        abr_mode="PENSIEVE-MULTI-FAIRNESS-AWARE"
        use_segment_based_collection="ON"
        segment_interval = 3
        startup_seed=42

        client_metrics = run_simulations(
            num_clients, 
            simulation_total_time, 
            # group_client=group_client_options[3],  # 'TEST-ONLY'
            group_client,  
            scenario, 
            abr_mode, 
            use_segment_based_collection,
            segment_interval,
            startup_seed
        )

        print("Generating CSV file")
        export_qoe_metrics_to_csv(
            client_metrics, 
            f'./simulations/simulation-ABR-{abr_mode}-scenario-{scenario_number}-group-client-{group_client}-numclients-{num_clients}-seed-number-{startup_seed}-simulation-time-{simulation_total_time}-datetime-{date}-{time_now}-process-mode-{execution_mode}.csv'
        )
    elif opcao == "0":
        print("Exitting....")
        break
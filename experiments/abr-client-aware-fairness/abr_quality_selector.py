# abr_quality_selector.py
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
import time
import json
from decision_module import compute_quality_level
from fairness_utils import update_fairness_scores, calculate_tf_qoe, calculate_qfs, calculate_bfi, calculate_saf, calculate_fairness_index, calculate_fairness_metrics
from metric_processing import collect_playback_metrics, process_metrics, calculate_jitter, detect_quality_changes, update_client_metrics_snapshot, calculate_weighted_quality, normalize_qoe_metrics, calculate_composite_qoe
from hotdash import hotdash_selector


# === PUBLISHING GLOBAL SHARED SCORES ===
def publish_client_score(client_id, group_client, client_metrics, abr_mode, shared_dir):

    logging.info(" ---->> Inside PUBLISH GLOBAL FAIRNESS <<<----")
    if not isinstance(client_metrics, dict) or client_id not in client_metrics:
        logging.error(f"[Publish] ERROR: client_metrics is invalid! client_id={client_id}, client_metrics={client_metrics}")
        return

    os.makedirs(shared_dir, exist_ok=True)
    data = {
        "client_id": client_id,
        "group": client_metrics[client_id].get("group_client", "unknown"),
        "urgency_level": client_metrics[client_id].get("urgency_level", 1),
        "tf_qoe": client_metrics[client_id].get("tf_qoe", 0),
        "qfs": client_metrics[client_id].get("qfs", 0),
        "bfi": client_metrics[client_id].get("bfi", 0),
        "saf": client_metrics[client_id].get("saf", 0),
        "timestamp": int(time.time())
    }
    with open(os.path.join(shared_dir, f"client-{client_id}.json"), "w") as f:
        json.dump(data, f)


# === GETTING GLOBAL SHARED SCORES ===
def get_global_scores(shared_dir, expiration=20):
    logging.info(" ---->> Inside GET GLOBAL FAIRNESS <<<----")
    current_time = int(time.time())
    tf_list, qfs_list, bfi_list, saf_list = [], [], [], []

    for file in os.listdir(shared_dir):
        if file.startswith("client-") and file.endswith(".json"):
            filepath = os.path.join(shared_dir, file)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                if current_time - data.get("timestamp", 0) <= expiration:
                    tf_list.append(data.get("tf_qoe", 0))
                    qfs_list.append(data.get("qfs", 0))
                    bfi_list.append(data.get("bfi", 0))
                    saf_list.append(data.get("saf", 0))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    global_scores = {
        "tf_qoe": sum(tf_list) / len(tf_list) if tf_list else 0,
        "qfs": sum(qfs_list) / len(qfs_list) if qfs_list else 0,
        "bfi": sum(bfi_list) / len(bfi_list) if bfi_list else 0,
        "saf": sum(saf_list) / len(saf_list) if saf_list else 0,
        "clients_count": len(tf_list)
    }

    logging.info(f"Showing global scores ->>> {global_scores}")

    return global_scores       


# === CALCULATE GLOBAL SHARED SCORES ===
def calculate_global_fairness_scores(client_metrics,
                                      shared_dir,
                                      use_shared_scores=True,
                                      expiration=20):
    """
    Calcula os fairness scores globais (TF-QoE, QFS, BFI, SAF).

    Se use_shared_scores=True, tenta ler os scores compartilhados dos outros containers.
    Caso contrário, calcula com base no client_metrics local.

    Args:
        client_metrics (dict): Dicionário local de métricas dos clientes ativos.
        shared_dir (str): Caminho para o diretório compartilhado de scores JSON.
        use_shared_scores (bool): Se True, lê os scores compartilhados.
        expiration (int): Tempo máximo (em segundos) para considerar scores válidos.

    Returns:
        dict: Global fairness scores {'tf_qoe': ..., 'qfs': ..., 'bfi': ..., 'saf': ...}
    """

    if use_shared_scores:
        try:
            logging.info('=== > Entering  SHARED SCORE routine === ')
            if not isinstance(client_metrics, dict):
                logging.error(f"[Fairness] ERROR: client_metrics is not dict! Got: {client_metrics} ({type(client_metrics)})")
                return {'tf_qoe': 0, 'qfs': 0, 'bfi': 0, 'saf': 0}
            
            global_scores = get_global_scores(shared_dir=shared_dir, expiration=expiration)
            # logging.info(f"Showing global_scores -> {global_scores}")




            if global_scores and global_scores.get("clients_count", 0) > 0:

                logging.info(
                    f"=== > SHARED SCORE DETECTED -> TF-QOE: {global_scores.get('tf_qoe', 0)}, "
                    f"QFS: {global_scores.get('qfs', 0)}, "
                    f"BFI: {global_scores.get('bfi', 0)}, "
                    f"SAF: {global_scores.get('saf', 0)} ==="
                )

                global_scores = {
                    'tf_qoe': global_scores.get('tf_qoe', 0),
                    'qfs': global_scores.get('qfs', 0),
                    'bfi': global_scores.get('bfi', 0),
                    'saf': global_scores.get('saf', 0)
                }
                # logging.info(f'=== > SHARED SCORE DETECTED -> TF-QOE: {tf_qoe}, QFS: {qfs}, BFI: {bfi},SAF: {saf} === ')
                return global_scores
            else:
                print("[Fairness] No valid shared scores found. Falling back to local calculation.")
        except Exception as e:
            print(f"[Fairness] Error reading shared scores: {e}. Falling back to local calculation.")

    # == Fallback ou modo local ==
    if not client_metrics:
        return {'tf_qoe': 0, 'qfs': 0, 'bfi': 0, 'saf': 0}

    tf_qoe_list = [client.get('tf_qoe', 0) for client in client_metrics.values()]
    qfs_list = [client.get('qfs', 0) for client in client_metrics.values()]
    bfi_list = [client.get('bfi', 0) for client in client_metrics.values()]
    saf_list = [client.get('saf', 0) for client in client_metrics.values()]

    global_scores = {
        'tf_qoe': np.mean(tf_qoe_list) if tf_qoe_list else 0,
        'qfs': np.mean(qfs_list) if qfs_list else 0,
        'bfi': np.mean(bfi_list) if bfi_list else 0,
        'saf': np.mean(saf_list) if saf_list else 0
    }

    logging.info(f"Showing global_scores -> {global_scores}")
    return global_scores


# === SELECT BITRATE QUALITY ===
def select_bitrate(client_id, group_client, client_metrics, abr_mode,
                   urgency_level, urgency_level_enabled, score_range,
                   model, shared_dir, weights):
    """
    Decide the next quality level for the client based on ABR mode.
    For native ABR (e.g., DASH.js, BOLA), returns None and lets the player handle it.
    For heuristic or RL modes, computes the quality level and returns it.

    Notes:
    - abr_mode is normalized (case-insensitive, accepts '-' or '_' variants).
    - Pensieve-Baseline is treated as a special case (simple state vector).
    - If a required model is missing, the function logs and returns None.
    """
    logging.info(f"🎯 Client {client_id}: ABR Mode selected -> {abr_mode} \n MODEL selected -> {model}")

    # Normalize abr_mode to a canonical uppercase-with-dash form
    if not abr_mode:
        logging.warning(f"Client {client_id}: No abr_mode provided.")
        return None
    abr_mode_norm = abr_mode.strip().upper().replace('_', '-')

    # ** Retrieving additional data for required modes
    segment_index = client_metrics[client_id].get("segment_index", 0)
    max_quality_level = client_metrics[client_id].get("max_quality_level", 6)

    # === ALWAYS update fairness scores, even for native ABR modes ===
    state = compute_fairness_and_update_state(client_id, client_metrics, weights, score_range)

    # === Native DASH.js player ABR ===
    player_driven_modes = {"DASH-DEFAULT", "DASH-QOE-BASED", "BOLA", "L2A"}

    if abr_mode_norm in player_driven_modes:
        logging.info(f"Client {client_id}: {abr_mode_norm} enabled (player-driven). No controller decision.")
        return None

    # === HOTDASH Strategy ===
    if abr_mode_norm == "HOTDASH":
        logging.info(f"Client {client_id}: HOTDASH ABR strategy selected. Call to HOTDASH procedure")
        return hotdash_selector(
            state=state,
            max_quality_level=max_quality_level,
            segment_index=segment_index
        )

    # === Pensieve RL Baseline ===
    if abr_mode_norm == "PENSIEVE-BASELINE":
        if not model:
            logging.warning(f"Client {client_id}: PENSIEVE-BASELINE selected but no model provided. Skipping decision.")
            return None
        state_vector = [
            state["avg_throughput"],
            state["avg_buffer_level"],
            state["avg_latency"],
            state["jitter"],
            state["quality_changes_total"],
        ]
        try:
            action = int(model.predict([state_vector], deterministic=True)[0])
            logging.info(f"Client {client_id}: RL Baseline selected {action}")
            return action
        except Exception as e:
            logging.error(f"Client {client_id}: Error predicting action for PENSIEVE-BASELINE: {e}")
            return None

    # === Pensieve QoE-Based ===
    if abr_mode_norm == "PENSIEVE-QOE-BASED":
        if not model:
            logging.warning(f"Client {client_id}: PENSIEVE-QOE-BASED selected but no model provided. Skipping decision.")
            return None
        state_vector = [
            state["avg_throughput"] * weights.get('avg_throughput', 1),
            state["avg_buffer_level"] * weights.get('avg_buffer_level', 1),
            state["avg_latency"] * weights.get('avg_latency', 1),
            state["jitter"] * weights.get('jitter', 1),
            state["quality_changes_total"] * weights.get('quality_changes_total', 1),
        ]
        try:
            action = int(model.predict([state_vector], deterministic=True)[0])
            logging.info(f"Client {client_id}: RL QoE-Based selected {action}")
            return action
        except Exception as e:
            logging.error(f"Client {client_id}: Error predicting action for PENSIEVE-QOE-BASED: {e}")
            return None

    # === Pensieve Fairness-Based ===
    if abr_mode_norm == "PENSIEVE-FAIRNESS":
        if not model:
            logging.warning(f"Client {client_id}: PENSIEVE-FAIRNESS selected but no model provided. Skipping decision.")
            return None
        # state_vector = [
        #     state["avg_throughput"] * weights.get('avg_throughput', 1),
        #     state["avg_buffer_level"] * weights.get('avg_buffer_level', 1),
        #     state["avg_latency"] * weights.get('avg_latency', 1),
        #     state["jitter"] * weights.get('jitter', 1),
        #     state["quality_changes_total"] * weights.get('quality_changes_total', 1),
        # ]

        state_vector = [
            state["avg_throughput"],
            state["avg_buffer_level"],
            state["avg_latency"],
            state["jitter"],
            state["quality_changes_total"],
            state.get("composite_qoe_score"),
            state.get("tf_qoe"),
            state.get("bfi"),
            state.get("qfs"),
            state.get("saf")
        ]

        
        try:
            action = int(model.predict([state_vector], deterministic=True)[0])
            logging.info(f"Client {client_id}: RL Fairness-Based selected {action}")
            return action
        except Exception as e:
            logging.error(f"Client {client_id}: Error predicting action for PENSIEVE-FAIRNESS: {e}")
            return None

    # === Pensieve Multi-Fairness Aware (Experiment 04) ===
    if abr_mode_norm == "PENSIEVE-MULTI-FAIRNESS-AWARE":
        if not model:
            logging.warning(f"Client {client_id}: PENSIEVE-MULTI-FAIRNESS-AWARE selected but no model provided. Skipping decision.")
            return None
        state_vector = [
            state["avg_throughput"],
            state["avg_buffer_level"],
            state["avg_latency"],
            state["jitter"],
            state["quality_changes_total"],
            state.get("composite_qoe_score"),
            state.get("tf_qoe"),
            state.get("bfi"),
            state.get("qfs"),
            state.get("saf")
        ]
        logging.info("----------------->>>>> Calculating Global Score <<<<<-----------------------")
        global_score = calculate_global_fairness_scores(client_metrics, shared_dir)

        logging.info("----------------->>>>> Publishing Global Score <<<<<-----------------------")
        publish_client_score(client_id, group_client, client_metrics, abr_mode_norm, shared_dir)

        logging.info("--- CALLING COMPUTE QUALITY LEVEL ---- ")
        try:
            action = compute_quality_level(state_vector, model, urgency_level, weights, global_score, urgency_level_enabled)
            logging.info(f"Client {client_id}: Multi-Fairness RL selected {action}")
            return action
        except Exception as e:
            logging.error(f"Client {client_id}: Error computing quality level for multi-fairness RL: {e}")
            return None

    # === Unknown mode fallback ===
    logging.warning(f"⚠️ Client {client_id}: Unknown or unsupported ABR mode: {abr_mode} (normalized: {abr_mode_norm})")
    return None


# Old version - keeps for reference
def select_bitrate_old(client_id, group_client,client_metrics, abr_mode, urgency_level,urgency_level_enabled, score_range, model, shared_dir, weights):
    """
    Decide the next quality level for the client based on ABR mode.
    For native ABR (e.g., DASH.js, BOLA), returns None and lets the player handle it.
    For heuristic or RL modes, computes the quality level and returns it.
    """
    logging.info(f"🎯 Client {client_id}: ABR Mode selected -> {abr_mode} \n MODEL selected -> {model}")
    # ** Retrieving additional data for required modes
    segment_index = client_metrics[client_id].get("segment_index", 0)
    max_quality_level = client_metrics[client_id].get("max_quality_level", 6)


    # === ALWAYS update fairness scores, even for native ABR modes ===
    state = compute_fairness_and_update_state(client_id, client_metrics, weights, score_range)
    
    # === Native DASH.js player ABR ===
    player_driven_modes = {"DASH-DEFAULT", "DASH-QOE-BASED", "BOLA", "L2A"}

    if abr_mode in player_driven_modes:
        logging.info(f"Client {client_id}: {abr_mode} enabled (player-driven). No controller decision.")
        return None

    # if abr_mode == "DASH-DEFAULT":
    #     logging.info(f"Client {client_id}: DASH.js default ABR enabled. No decision needed.")
    #     return None

    # # === Heuristic QoE-Based Strategy ===
    # elif abr_mode == "DASH-QOE-BASED":
    #     logging.info(f"Client {client_id}: DASH.js QOE-Based ABR enabled. No decision needed.")
    #     return None

    # # === BOLA Strategy ===
    # elif abr_mode == "BOLA":
    #     logging.info(f"Client {client_id}: BOLA ABR strategy selected. No decision needed.")
    #     return None
    
    # === HOTDASH Strategy ===
    elif abr_mode == "HOTDASH":
        logging.info(f"Client {client_id}: HOTDASH ABR strategy selected. Call to HOTDASH procedure")
        return hotdash_selector(
        state=state,
        max_quality_level=max_quality_level,
        segment_index=segment_index
        )


    # === Pensieve RL Baseline ===
    elif abr_mode == "PENSIEVE-BASELINE" and model:
        state_vector = [
            state["avg_throughput"],
            state["avg_buffer_level"],
            state["avg_latency"],
            state["jitter"],
            state["quality_changes_total"],
        ]
        action = int(model.predict([state_vector], deterministic=True)[0])
        logging.info(f"Client {client_id}: RL Baseline selected {action}")
        return action

    # === Pensieve QoE-Based ===
    elif abr_mode == "PENSIEVE-QOE-BASED" and model:
        state_vector = [
            state["avg_throughput"] * weights.get('avg_throughput', 1),
            state["avg_buffer_level"] * weights.get('avg_buffer_level', 1),
            state["avg_latency"] * weights.get('avg_latency', 1),
            state["jitter"] * weights.get('jitter', 1),
            state["quality_changes_total"] * weights.get('quality_changes_total', 1),
        ]
        action = int(model.predict([state_vector], deterministic=True)[0])
        logging.info(f"Client {client_id}: RL QoE-Based selected {action}")
        return action
    

    # === Pensieve Fairness-Based ===
    elif abr_mode == "PENSIEVE-FAIRNESS" and model:
        state_vector = [
            state["avg_throughput"] * weights.get('avg_throughput', 1),
            state["avg_buffer_level"] * weights.get('avg_buffer_level', 1),
            state["avg_latency"] * weights.get('avg_latency', 1),
            state["jitter"] * weights.get('jitter', 1),
            state["quality_changes_total"] * weights.get('quality_changes_total', 1),
        ]
        action = int(model.predict([state_vector], deterministic=True)[0])
        logging.info(f"Client {client_id}: RL Fairness-Based selected {action}")
        return action

    # === Pensieve Multi-Fairness Aware (Experiment 04) ===
    elif abr_mode == "PENSIEVE-MULTI-FAIRNESS-AWARE" and model:
        state_vector = [
            state["avg_throughput"],
            state["avg_buffer_level"],
            state["avg_latency"],
            state["jitter"],
            state["quality_changes_total"],
            state["composite_qoe_score"],
            state["tf_qoe"],
            state["bfi"],
            state["qfs"],
            state["saf"]
        ]
        logging.info("----------------->>>>> Calculating Global Score <<<<<-----------------------")
        global_score = calculate_global_fairness_scores(client_metrics, shared_dir)

        logging.info("----------------->>>>> Publishing Global Score <<<<<-----------------------")
        publish_client_score(client_id, group_client, client_metrics, abr_mode, shared_dir)

        logging.info("--- CALLING COMPUTE QUALITY LEVEL ---- ")
        action = compute_quality_level(state_vector, model, urgency_level, weights, global_score,urgency_level_enabled)
        logging.info(f"Client {client_id}: Multi-Fairness RL selected {action}")
        return action

    # === Unknown mode fallback ===
    else:
        logging.warning(f"⚠️ Client {client_id}: Unknown or unsupported ABR mode: {abr_mode}")
        return None


def compute_fairness_and_update_state(client_id, client_metrics, weights, score_range):
    """
    Compute fairness scores, update client metrics snapshot, and return the current state vector.

    :param processed_metrics: Dictionary of processed metrics for the current sample
    :param client_id: ID of the client
    :param client_metrics: Global client metrics dictionary
    :param weights: Dictionary with weights for each metric
    :return: Current state vector (list of floats)
    """

    try:
        metrics = client_metrics[client_id]

        # === 1. Coleta de métricas atuais ===
        avg_throughput = metrics.get('avg_throughput', 0)
        avg_latency = metrics.get('avg_latency', 0)
        avg_jitter = metrics.get('jitter', 0)
        avg_buffer_level = metrics.get('avg_buffer_level',0)
        dropped_frames = metrics.get('dropped_frames', 0)
        buffering_time = metrics.get('buffering_time', 0)
        buffering_count = metrics.get('buffering_count', 0)
        playback_interruptions = metrics.get('playback_interruptions', 0)
        # quality_changes = metrics.get('quality_changes', 0)
        quality_changes_total = metrics.get('quality_changes_total', 0)

        # === 2. Coleta de arrays de amostras ===
        throughput_samples = metrics.get('throughput_samples',[])
        quality_changes_samples = metrics.get('quality_changes_samples', [])
        dropped_frames_samples = metrics.get('dropped_frames_samples', [])
        latency_samples = metrics.get('latency_samples', [])
        timestamp_samples = metrics.get('timestamp_samples', [])

        

        # === 3. Fairness Scores calculation ===
        logging.info(f"=== Updating composite Qoe for client {client_id} === ")
        
        logging.info(f"📦 Client {client_id} - Weights received before composite QoE: {weights}")
        composite_qoe = calculate_composite_qoe(client_id, client_metrics, weights, score_range)
        
        logging.info(f"=== Composite QoE measured for client {client_id} : {composite_qoe} ")
        
        
        tf_qoe_scores = calculate_tf_qoe(client_metrics)
        qfs_scores = calculate_qfs(client_id, client_metrics)
        bfi_scores = calculate_bfi(client_id, client_metrics)
        saf_scores = calculate_saf(client_id, client_metrics)

        tf_qoe = tf_qoe_scores.get(client_id, 0)
        qfs = qfs_scores.get(client_id, 0)
        bfi = bfi_scores.get(client_id, 0)
        saf = saf_scores.get(client_id, 0)

        # Update fairness scores in client_metrics
        update_fairness_scores(client_id, client_metrics, composite_qoe, tf_qoe, qfs, bfi, saf)


        # client_metrics[client_id]['dropped_frames_total'] = client_metrics[client_id].get('dropped_frames_total', 0) + dropped_frames
        # client_metrics[client_id]['quality_changes_total'] = client_metrics[client_id].get('quality_changes_total', 0) + quality_changes


        # === 4. Prepara snapshot para atualização ===
        # snapshot_data = {
            # 'avg_throughput': avg_throughput,
            # 'avg_latency': avg_latency,
            # 'avg_jitter': avg_jitter,
            # 'avg_buffer_level': avg_buffer_level,
            # 'buffering_time': buffering_time,
            # 'playback_interruptions': playback_interruptions,
            # 'buffering_count': buffering_count,
            # 'quality_changes_samples': quality_changes_samples,
            # 'dropped_frames_samples': dropped_frames_samples,
            # 'latency_samples': latency_samples,
            # 'timestamp_samples': timestamp_samples,
            # 'quality_changes_total': client_metrics[client_id].get('quality_changes_total', 0),
            # 'dropped_frames_total': client_metrics[client_id].get('dropped_frames_total', 0)
            # Atualiza as métricas acumuladas
        # }


        # logging.info(f"================ CLIENT {client_id} FAIRNESS SNAPSHOT DATA ================")
        # logging.info(f"Snapshot Data: {snapshot_data}")

        # logging.info(f"Buffering Count         : {snapshot_data.get('buffering_count')}")
        # logging.info(f"Playback Interruptions  : {snapshot_data.get('playback_interruptions')}")
        # logging.info(f"Buffering Time (s)      : {snapshot_data.get('buffering_time')}")
        # logging.info(f"Buffer Level            : {snapshot_data.get('avg_buffer_level')}")
        # logging.info(f"Avg Throughput          : {snapshot_data.get('avg_throughput')}")
        # logging.info(f"Avg Latency             : {snapshot_data.get('avg_latency')}")
        # logging.info(f"Avg Jitter              : {snapshot_data.get('avg_jitter')}")
        # logging.info(f"Dropped Frames Total    : {snapshot_data.get('dropped_frames_total')}")
        # logging.info(f"Quality Changes Total   : {snapshot_data.get('quality_changes_total')}")
        # logging.info(f"============================================================")



        logging.info(f"---------------- ACTUAL client_metrics[{client_id}] DICTIONARY VALUES ----------------")
        logging.info(f"Buffering Count         : {client_metrics[client_id].get('buffering_count')}")
        logging.info(f"Playback Interruptions  : {client_metrics[client_id].get('playback_interruptions')}")
        logging.info(f"Buffering Time (s)      : {client_metrics[client_id].get('buffering_time')}")
        logging.info(f"Buffer Level            : {client_metrics[client_id].get('avg_buffer_level')}")
        logging.info(f"Avg Throughput          : {client_metrics[client_id].get('avg_throughput')}")
        logging.info(f"Avg Latency             : {client_metrics[client_id].get('avg_latency')}")
        # logging.info(f"Avg Jitter              : {client_metrics[client_id].get('jitter')}")
        logging.info(f"Avg Jitter              : {avg_jitter}")
        logging.info(f"Dropped Frames Total    : {client_metrics[client_id].get('dropped_frames_total')}")
        logging.info(f"Quality Changes Total   : {client_metrics[client_id].get('quality_changes_total')}")
        logging.info(f"============================================================")


        # logging.info(f"CLIENT: {client_id} ===== >>> SHOWING SNAPSHOT DATA BEFORE UPDATE CLIENT METRICS -->> {snapshot_data} <<< =====")

        # Snapshot client update
        # update_client_metrics_snapshot(client_id, client_metrics, snapshot_data)


        # === 5. Gera vetor de estado ===
        # state = [
        #     avg_throughput,
        #     avg_buffer_level,
        #     avg_latency,
        #     avg_jitter,
        #     quality_changes,
        #     composite_qoe,
        #     tf_qoe,
        #     bfi,
        #     qfs,
        #     saf
        # ]

        state = {
        "throughput_samples": throughput_samples, #Check if it is needed in state
        "avg_throughput": avg_throughput,
        "avg_buffer_level": avg_buffer_level,
        "avg_latency": avg_latency,
        "jitter": avg_jitter,
        "quality_changes_total": quality_changes_total,
        "composite_qoe_score": composite_qoe,
        "tf_qoe": tf_qoe,
        "bfi": bfi,
        "qfs": qfs,
        "saf": saf
        }


        # logging.info(f"================ CLIENT {client_id} STATE VECTOR ================")
        # logging.info(f"State Vector: {state}")
        # logging.info(f"============================================================")

        # logging.info(f"📊 Client {client_id}: State vector computed sucessfully: {state}")

        return state

    except Exception as e:
        logging.error(f"🚨 Client {client_id}: Error computing fairness and state - {e}")
        return None


def apply_quality(driver, quality, client_id, abr_mode):
    try:
        if quality is not None:
            driver.execute_script("player.updateSettings({streaming: {abr: {enabled: false}}});")
            driver.execute_script(f"player.setQualityFor('video', {int(quality)});")
            logging.info(f"Client {client_id}: Applied quality level {int(quality)}")
        else:
            if abr_mode == "DASH-DEFAULT":
                driver.execute_script("player.updateSettings({streaming: {abr: {enabled: true}}});")
                logging.info(f"Client {client_id}: DASH.js default ABR enabled.")
            elif abr_mode == "BOLA":
                driver.execute_script("player.updateSettings({streaming: {abr: {useDefaultABRRules: false, ABRStrategy: 'abrBola'}}});")
                logging.info(f"Client {client_id}: BOLA ABR strategy activated.")
    except Exception as e:
        logging.error(f"Client {client_id}: Failed to apply quality level - {e}")


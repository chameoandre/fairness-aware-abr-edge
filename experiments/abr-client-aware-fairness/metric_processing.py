import logging
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import psutil, time, threading


def collect_playback_metrics(driver, video_element, client_id):
    logging.info(f"== inside COLLECT PLAYBACK METRICS routine === ")
    try:

        # logging.info(f"== SHOWING ALL PLAYER METRICS for CLIENT {client_id}=== ")
        # # Coleta o conteúdo completo de player.getMetricsFor('video')
        
        # dash_metrics = driver.execute_script("""
        #     return player.getMetricsFor('video');
        # """)

        # logging.info(f"Client {client_id}: DASH.js metrics snapshot: {dash_metrics}")


        metrics = driver.execute_script("""
            return {
                bufferLevel: (typeof getCurrentBufferLevel === 'function') ? getCurrentBufferLevel() : 0,
                bitrate: (typeof getQuality === 'function') ? getQuality() : 0,
                throughput: (typeof getAvgThroughput === 'function') ? getAvgThroughput() : 0,
                droppedFrames: (typeof getDroppedFrames === 'function') ? getDroppedFrames() : 0,
                qualityLevel: (typeof getActualRepresentation === 'function') ? getActualRepresentation() : 0,
                bufferingTime: (window.bufferingEvents) ? window.bufferingEvents.reduce((a, b) => a + b, 0) / 1000.0 : 0,
                bufferingCount: (window.bufferingEvents) ? window.bufferingEvents.length : 0,
                currentPlaybackTime: arguments[0].currentTime || 0,
                latency: getLatency(),
                transfer_latency: getTransferLatency()
            }
        """, video_element)
        
        logging.info(f"================ CLIENT {client_id} COLLECT METRIC ROUTINE [RAW DATA]================")
        logging.info(f"RAW Metrics: {metrics}")

        logging.info(f"Buffering Count         : {metrics.get('bufferingCount')}")
        logging.info(f"Buffering Time (s)      : {metrics.get('bufferingTime')}")
        logging.debug(f"Buffering Events Raw (ms): {metrics.get('bufferingEventsRaw')}")

        logging.info(f"Buffer Level            : {metrics.get('bufferLevel')}")
        logging.info(f"Bitrate                 : {metrics.get('bitrate')}")
        logging.info(f"Throughput              : {metrics.get('throughput')}")
        logging.info(f"Dropped Frames          : {metrics.get('droppedFrames')}")
        logging.info(f"Quality Level           : {metrics.get('qualityLevel')}")
        logging.info(f"Latency                 : {metrics.get('latency')}")
        logging.info(f"Transfer Latency        : {metrics.get('transfer_latency')}")
        logging.info(f"Current Playback Time   : {metrics.get('currentPlaybackTime')}")
        logging.info(f"============================================================")

        return metrics
                # startupDelay: getStartupDelay(),

    except Exception as e:
        logging.error(f"Client {client_id}: Error collecting playback metrics - {e}")
        return {}
    
   
def process_metrics(playback_metrics, client_id, client_metrics):
    """
    Process raw playback metrics and update client metrics with history and derived values like jitter.
    Startup delay isn't calculated here. It's calculated before in MEASURE_QOE function
    
    :param playback_metrics: Dictionary of metrics collected in the current loop.
    :param client_id: ID of the client.
    :return: Dictionary with processed metrics for the current sample.

    # NOTE: buffering_count and playback_interruptions are currently identical.
     # Both store the accumulated count of buffering events sent from the player.
    """
    processed = {}
    segment_duration = 4  # segundos — ajuste conforme o MPD [needs update later]
    max_quality_level = 6  # Níveis de qualidade: 0 a 5 (ex: 320p a 2160p) - [needs update later]


    logging.info(f"======= Inside PROCESS METRICS for client -> {client_id} ======= ")

    try:
        # === Process basic metrics ===
        processed['latency'] = playback_metrics.get('latency', 0)
        processed['avg_throughput'] = playback_metrics.get('throughput', 0)
        processed['buffer_level'] = playback_metrics.get('bufferLevel', 0)
        processed['buffering_time'] = playback_metrics.get('bufferingTime', 0)
        processed['buffering_count'] = playback_metrics.get('bufferingCount', 0)
        processed['dropped_frames'] = playback_metrics.get('droppedFrames', 0)
        processed['quality_level'] = playback_metrics.get('qualityLevel', 0)
        processed['current_playback_time'] = playback_metrics.get('currentPlaybackTime', 0)
        processed['bitrate'] = playback_metrics.get('bitrate',0)



        # === Update snapshot in client_metrics ===
        client_metrics[client_id]['avg_throughput'] = processed['avg_throughput']
        # client_metrics[client_id]['avg_buffer_level'] = processed['buffer_level']

        
        client_metrics[client_id]['buffering_time'] = processed['buffering_time']
        client_metrics[client_id]['buffering_count'] = processed['buffering_count']
        client_metrics[client_id]['playback_interruptions'] = processed['buffering_count']
        client_metrics[client_id]['quality_level'] = processed['quality_level']

        client_metrics[client_id]['current_playback_time'] = processed['current_playback_time']
        current_playback_time = client_metrics[client_id]['current_playback_time']


        # === Generating segment Index ===
        segment_index = int(current_playback_time // segment_duration)
        processed['segment_index'] = segment_index
        client_metrics[client_id]['segment_index'] = segment_index

        # === Adapting max quality level [sintetic]
        processed['max_quality_level'] = max_quality_level
        client_metrics[client_id]['max_quality_level'] = max_quality_level


        # === Update histories ===
        client_metrics[client_id]['latency_samples'].append(processed['latency'])
        client_metrics[client_id]['throughput_samples'].append(processed['avg_throughput'])
        client_metrics[client_id]['buffer_level_samples'].append(processed['buffer_level'])
        client_metrics[client_id]['dropped_frames_samples'].append(processed['dropped_frames'])
        client_metrics[client_id]['bitrate_samples'].append(processed['bitrate'])

        # === Update sum metrics ===
        client_metrics[client_id]['dropped_frames_total'] += processed['dropped_frames']

        # === Calculate avg buffer level using existing function ===
        avg_buffer_level = calculate_avg_buffer_level(client_id, client_metrics)
        client_metrics[client_id]['avg_buffer_level'] = avg_buffer_level
        # logging.info(f"AVG BUFFER LEVEL for client {client_id} calculated and stored -> {client_metrics[client_id]['avg_buffer_level']}")

        # === Calculate jitter using existing function ===
        client_metrics[client_id]['avg_latency'] = calculate_avg_latency(client_id, client_metrics)
        processed['jitter'] = calculate_jitter(client_id, client_metrics)
        client_metrics[client_id]['jitter'] = processed['jitter']
        # print(f"-----> Checking JITTER stored value for client {client_id} => {client_metrics[client_id]['jitter']}")

        client_metrics[client_id]['jitter_samples'].append(processed['jitter'])

        # === Detect and store Quality Changes using existing function ===
        processed['quality_changes'] = detect_quality_changes(client_id, processed['quality_level'], client_metrics)
        client_metrics[client_id]['quality_level_samples'].append(processed['quality_level'])
        client_metrics[client_id]['quality_changes_samples'].append(processed['quality_changes'])
        client_metrics[client_id]['quality_changes_total'] += processed['quality_changes']

        # logging.info(f"Client {client_id}: Processed Metrics calculated successfully: {client_metrics}")

        logging.info(f"================ CLIENT {client_id} PROCESSED METRICS ================")
        # logging.info(f"Processed Metrics: {processed}")

        logging.info(f"Buffering Count         : {processed.get('buffering_count')}")
        logging.info(f"Buffering Time (s)      : {processed.get('buffering_time')}")
        logging.info(f"Buffer Level            : {processed.get('buffer_level')}")
        logging.info(f"Quality Level           : {processed.get('quality_level')}")
        logging.info(f"Throughput              : {processed.get('avg_throughput')}")
        logging.info(f"Dropped Frames          : {processed.get('dropped_frames')}")
        logging.info(f"Latency                 : {processed.get('latency')}")
        logging.info(f"Jitter                  : {processed.get('jitter')}")
        logging.info(f"Current Playback Time   : {processed.get('current_playback_time')}")
        logging.info(f"Current Bitrate Level   : {processed.get('bitrate')}")
        logging.info(f"============================================================")

    except Exception as e:
        logging.error(f"Client {client_id}: Error processing metrics - {e}")

    return processed


def start_global_scalability_sampler(shared_metrics, interval=2.0, stop_event=None):
    shared_metrics.setdefault('global_scalability_samples', [])
    psutil.cpu_percent(None)  # warm-up

    def loop():
        while stop_event is None or not stop_event.is_set():
            vm = psutil.virtual_memory()
            sample = {
                't': time.time(),
                'container_cpu_pct': psutil.cpu_percent(None),
                'container_mem_pct': vm.percent,
            }
            shared_metrics['global_scalability_samples'].append(sample)
            time.sleep(interval)
    th = threading.Thread(target=loop, daemon=True, name="global-scalability")
    th.start()
    return th

def finalize_global_scalability(shared_metrics):
    samples = shared_metrics.get('global_scalability_samples', [])
    if not samples:
        return {}
    import numpy as np
    cpu = np.array([s['container_cpu_pct'] for s in samples], float)
    mem = np.array([s['container_mem_pct'] for s in samples], float)
    return {
        'container_cpu_pct_avg': float(cpu.mean()),
        'container_cpu_pct_p95': float(np.percentile(cpu, 95)),
        'container_mem_pct_avg': float(mem.mean()),
        'container_mem_pct_p95': float(np.percentile(mem, 95)),
    }


def calculate_avg_buffer_level(client_id, client_metrics):
    """
    Calculate the average buffer level for the given client based on buffer level samples.

    :param client_id: ID of the client
    :param client_metrics: Dictionary containing metrics for all clients
    :return: Average buffer level (float)
    """
    try:
        buffer_level_samples = client_metrics[client_id].get('buffer_level_samples', [])
        if not buffer_level_samples:
            return 0.0  # fallback para evitar divisão por zero

        # logging.info(f"---->>>>> Calculating [AVG Buffer] for client {client_id}")
        # logging.info(f"---->>>>> Actual buffer samples for client {client_id} -->> {buffer_level_samples}")

        avg_buffer_level = sum(buffer_level_samples) / len(buffer_level_samples)
        # logging.info(f"---->>>>> AVG Buffer calculated -> {avg_buffer_level} ")
        return avg_buffer_level

    except Exception as e:
        logging.error(f"Error calculating avg_buffer_level for client {client_id}: {e}")
        return 0.0


    

def calculate_avg_latency(client_id, client_metrics):
    """
    Calculate the average latency for the given client based on latency samples.

    :param client_id: ID of the client
    :param client_metrics: Dictionary containing metrics for all clients
    :return: Average latency (float)
    """
    try:
        latency_samples = client_metrics[client_id].get('latency_samples', [])
        if not latency_samples:
            return 0.0  # fallback para evitar divisão por zero

        avg_latency = sum(latency_samples) / len(latency_samples)
        return avg_latency

    except Exception as e:
        logging.error(f"Error calculating avg_latency for client {client_id}: {e}")
        return 0.0
    
    
def calculate_jitter(client_id, client_metrics):
    """
    Calculate jitter for the client based on latency variations.

    :param client_id: ID of the client
    :param client_metrics: Dictionary containing client metrics
    :return: jitter value (float)
    """
    history = client_metrics[client_id].get('latency_samples', [])
    
    if len(history) < 2:
        return 0.0

    # Calcula as diferenças absolutas entre amostras consecutivas
    diffs = [
        abs(safe_float(history[i]) - safe_float(history[i - 1]))
        for i in range(1, len(history))
    ]

    jitter = sum(diffs) / len(diffs) if diffs else 0.0
    logging.info(f"-----> Checking JITTER calculated value for client {client_id} => {jitter}")
    return round(jitter, 3)


def detect_quality_changes(client_id, current_quality_level, client_metrics):
    """
    Detects if there was a change in the quality level.

    :param client_id: ID of the client.
    :param current_quality_level: Current quality level (string or numeric).
    :param client_metrics: Global client metrics dictionary.
    :return: 1 if there was a change, 0 otherwise.
    """
    logging.info(f"=== Inside quality change detection ===")

    samples = client_metrics[client_id].get('quality_level_samples', [])

    if len(samples) >= 1:
        previous_quality = str(samples[-1])
        current_quality = str(current_quality_level)

        if current_quality != previous_quality:
            logging.info(f"⚠️ Quality change detected for client {client_id}: {previous_quality} → {current_quality}")
            return 1
        else:
            return 0
    else:
        # No previous sample, no change detected
        return 0
    

def update_client_metrics_snapshot(client_id, client_metrics, processed_metrics):
    
    """
    Update the snapshot of client metrics for the given client_id.

    :param client_id: ID of the client
    :param client_metrics: Dictionary containing all metrics for all clients
    """

    # logging.info(f"==== Inside UPDATE CLIENT METRICS SNAPSHOT ====")
    try:
        if client_id not in client_metrics:
            logging.warning(f"Client {client_id} not found in client_metrics. Initializing entry.")
            client_metrics[client_id] = {}

        # Perform deep copy to avoid pointer/reference issues
        # client_metrics_snapshot[client_id] = client_metrics[client_id].copy()
        client_metrics[client_id].update(processed_metrics)

        # logging.info(f"Client {client_id}: Metrics snapshot updated successfully.")

    except Exception as e:
        logging.error(f"Error updating metrics snapshot for client {client_id}: {e}")


# == ABR WHEIGHTS DECISION INFLUENCE == 
def calculate_weighted_quality(buffering_time, playback_interruptions, avg_throughput, latency, jitter, avg_buffer_level, dropped_frames_total, quality_changes_total, weights, score_range):
    
    # Criando dicionário de métricas para cálculo
    metrics = {
        'buffering_time': buffering_time,
        'playback_interruptions': playback_interruptions,
        'avg_throughput': avg_throughput,
        'latency': latency,
        'jitter': jitter,
        'avg_buffer_level': avg_buffer_level,
        'dropped_frames_total': dropped_frames_total,
        'quality_changes_total': quality_changes_total
    }

    # Aplicando pesos para calcular o QoE Score
    qoe_score = sum(metrics[key] * weights.get(key, 1) for key in metrics)

    # Atualiza os limites dinâmicos
    # global min_score, max_score
    # min_score = min(min_score, qoe_score)
    # max_score = max(max_score, qoe_score)
    if qoe_score < score_range["min_score"]:
        score_range["min_score"] = qoe_score
        
    if qoe_score > score_range["max_score"]:
        score_range["max_score"] = qoe_score



    # === MODIFIED TO USE DICTIONARY ===
    # Evita problemas com min_score == max_score (caso o qoe_score não varie no início)
    # if max_score > min_score:
    #     normalized_quality = int((qoe_score - min_score) / (max_score - min_score) * 5)
    # else:
    #     normalized_quality = 0  # Se não houver variação ainda, mantém no nível mínimo

   
   
    # === DICTIONARY TO CONTROL MAX / MIN CHECK VALUES ===
    if score_range["max_score"] > score_range["min_score"]:
        normalized_quality = int(
            (qoe_score - score_range["min_score"]) / (score_range["max_score"] - score_range["min_score"]) * 5
        )
    else:
        normalized_quality = 0  # Se não houver variação ainda



    # Garante que o valor final esteja no intervalo válido (0 a 5)
    normalized_quality = max(0, min(normalized_quality, 5))

    # Logs para depuração
    print(f"QoE Score Calculated: {qoe_score}")
    print(f"Min QoE Score Updated: {score_range['min_score']}, Max QoE Score UPDATED: {score_range['max_score']}")
    print(f"Qualidade Normalized: {normalized_quality}")

    return normalized_quality


# --- QOE NORMALIZATION DATA ---
def normalize_qoe_metrics(client_metrics, metrics_to_exclude=None, score_range=(0,1)):
    """
    Normalize QoE metrics for each client using Min-Max Scaling.

    :param client_metrics: Dictionary containing metrics for each client.
    :param metrics_to_exclude: List of metrics to exclude from normalization.
    :param score_range: Tuple defining the normalization range (default 0-1).
    :return: Updated client_metrics with normalized values added.
    """
    logging.info(f"==== Inside NORMALIZE_QOE_METRICS operation ===")
    metrics_to_exclude = metrics_to_exclude or []

    metrics_to_normalize = [
        'buffering_time',
        'playback_interruptions',
        'avg_throughput',
        'latency',
        'jitter',
        'avg_buffer_level',
        'quality_changes_total',
        'dropped_frames_total'
    ]
    metrics_to_normalize = [m for m in metrics_to_normalize if m not in metrics_to_exclude]

    client_ids = list(client_metrics.keys())
    all_metrics = {metric: [] for metric in metrics_to_normalize}

    for metric in metrics_to_normalize:
        for client_id in client_ids:
            value = client_metrics[client_id].get(metric, 0)

            if isinstance(value, list):
                if len(value) == 1:
                    value = value[0]
                else:
                    logging.warning(f"⚠️ Client {client_id}: Metric '{metric}' is a list. Using 0 as fallback.")
                    value = 0

            if not isinstance(value, (int, float)):
                logging.warning(f"⚠️ Client {client_id}: Metric '{metric}' is not numeric. Using 0 as fallback.")
                value = 0

            all_metrics[metric].append(value)

    min_range, max_range = score_range

    for metric, values in all_metrics.items():
        try:
            if not values:
                # logging.warning(f"⚠️ Metric '{metric}' has no values. Normalized to 0.0")
                normalized_values = [0.0 for _ in client_ids]
            elif all(v == 0 for v in values):
                # logging.info(f"ℹ️ Metric '{metric}': All values are zero. Normalized to 0.0")
                normalized_values = [0.0 for _ in values]
            elif min(values) == max(values):
                # logging.info(f"ℹ️ Metric '{metric}': All values are equal (non-zero). Normalized to 1.0")
                normalized_values = [1.0 for _ in values]
            else:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
                # Rescale to desired range
                normalized_values = scaled * (max_range - min_range) + min_range

                # Clip to range explicitly
                normalized_values = [max(min(v, max_range), min_range) for v in normalized_values]

            normalized_values = [float(v) for v in normalized_values]

            # for idx, client_id in enumerate(client_ids):
            #     normalized_metric_name = f"normalized_{metric}"
            #     client_metrics[client_id][normalized_metric_name] = normalized_values[idx]
            for idx, client_id in enumerate(client_ids):
                normalized_metric_name = f"normalized_{metric}"
                value = float(f"{normalized_values[idx]:.4f}")
                client_metrics[client_id][normalized_metric_name] = value

                # Inicializa log por cliente
                # if f"normalized_log_{client_id}" not in locals():
                #     locals()[f"normalized_log_{client_id}"] = []

                # locals()[f"normalized_log_{client_id}"].append((normalized_metric_name, value))

                
        except Exception as e:
            logging.error(f"❌ Error normalizing metric '{metric}': {e}")
        
    # === Showing Normalizated metrics === 
    for client_id in client_ids:
        logging.info(f"\n=== Normalized Metrics for Client {client_id} ===")
        for metric in sorted(client_metrics[client_id]):
            if metric.startswith("normalized_"):
                logging.info(f"• {metric.replace('normalized_', ''):<24}: {client_metrics[client_id][metric]:.4f}")


# == CALCULATING Composite QoE for Each client (Composite QoE) ==

def calculate_composite_qoe(client_id, client_metrics, weights, score_range, metrics_to_exclude=None):
    """
    Calculate Composite QoE for a specific client based on normalized QoE metrics.

    :param client_metrics: Dictionary of current metrics for all clients.
    :param client_id: ID of the client.
    :param weights: Dictionary of weights for QoE metrics.
    :param score_range: Tuple defining min and max range for scores.
    :param metrics_to_exclude: List of metrics to exclude from normalization.
    :return: Composite QoE score (float).
    """
    try:

        # ✅ Corrige score_range se vier como dict ou None
        if isinstance(score_range, dict):
            score_range = (
                score_range.get('min_score', 0),
                score_range.get('max_score', 1)
            )
        score_range = score_range or (0, 1)

        # ✅ Força números finitos e ordem correta
        min_range, max_range = score_range
        if not (math.isfinite(min_range) and math.isfinite(max_range)) or min_range >= max_range:
            score_range = (0, 1)

        logging.info(f"📌 score_range used: {score_range}")


        # ✅ Verifica se há pesos válidos
        if not weights or all(w == 0 for w in weights.values()):
            logging.warning(f"⚠️ All weights are zero or missing. Forcing Composite QoE to 0.0.")
            return 0.0

        # === Normalizar as métricas se ainda não estiverem normalizadas ===
        normalize_qoe_metrics(client_metrics, metrics_to_exclude)

        # === Obter as métricas normalizadas do cliente ===
        current_client_metrics = client_metrics.get(client_id, {})
        logging.info(f"✅ Normalized metrics for Client {client_id}: {current_client_metrics}")

        composite_score = 0
        logging.info(f"🧠 Composite QoE Calculation for Client {client_id}:")
        logging.info(f"Weights used: {weights}")

        for metric, weight in weights.items():
            normalized_metric_name = f"normalized_{metric}"
            value = current_client_metrics.get(normalized_metric_name, None)

            if value is None:
                logging.warning(f"⚠️ Client {client_id}: Missing normalized value for '{metric}'. Skipping.")
                continue

            partial = value * weight
            composite_score += partial

            logging.info(f"→ Component: {metric} | Value: {value} | Weight: {weight} | Partial: {partial:.4f}")

        logging.info(f"🎯 Client {client_id}: Raw Composite QoE = {composite_score:.4f}")

        # === Rescaling final para garantir dentro de score_range ===
        min_expected = sum([min(0, w) for w in weights.values()])
        max_expected = sum([max(0, w) for w in weights.values()])
        diff_expected = max_expected - min_expected

        logging.info(f"min_expected: {min_expected}, max_expected: {max_expected}, diff_expected: {diff_expected}")

        # ✅ Proteção contra divisão inválida ou explosão numérica
        if diff_expected <= 0 or not math.isfinite(diff_expected):
            logging.warning(f"⚠️ Invalid scaling range for Composite QoE (diff_expected={diff_expected}). Forcing 0.0.")
            return 0.0

        normalized_score = (composite_score - min_expected) / diff_expected

        # Clamp para score_range
        if not math.isfinite(normalized_score):
            logging.warning(f"⚠️ Computed normalized_score is not finite ({normalized_score}). Forcing 0.0.")
            normalized_score = 0.0
        else:
            normalized_score = max(score_range[0], min(score_range[1], normalized_score))

        logging.info(f"✅ Client {client_id}: Normalized Composite QoE = {normalized_score:.4f}")
        return normalized_score

    except Exception as e:
        logging.error(f"❌ Error calculating Composite QoE for Client {client_id}: {e}")
        return 0

    
def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


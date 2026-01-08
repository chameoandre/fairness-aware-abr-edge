import logging
import logging
import matplotlib.pyplot as plt
import numpy as np
import math


def update_fairness_scores(client_id, client_metrics, composite_qoe, tf_qoe, qfs, bfi, saf):
    logging.info(f"=== inside UPDATE FAIRNESS SCORES routine ==== ")
    """
    Update fairness-related scores for the client in the current simulation step.

    :param client_id: ID of the client
    :param client_metrics: Dictionary holding client metrics
    :param composite_qoe: Composite QoE Score (float)
    :param tf_qoe: Time-Fair QoE Score (float)
    :param qfs: QoE Fairness Score (float)
    :param bfi: Bitrate Fairness Index (float)
    :param saf: Streaming Adaptability Fairness (float)
    """
    try:
        metrics = client_metrics[client_id]

        # Atualiza os valores atuais (última amostra)
        metrics['composite_qoe'] = composite_qoe
        metrics['tf_qoe'] = tf_qoe
        metrics['qfs'] = qfs
        metrics['bfi'] = bfi
        metrics['saf'] = saf

        # Atualiza os históricos
        metrics.setdefault('composite_qoe_samples', []).append(composite_qoe)
        metrics.setdefault('tf_qoe_samples', []).append(tf_qoe)
        metrics.setdefault('qfs_samples', []).append(qfs)
        metrics.setdefault('bfi_samples', []).append(bfi)
        metrics.setdefault('saf_samples', []).append(saf)


        # === Logging detalhado ===
        # logging.info(
        #     f"Client {client_id}: Updated Fairness Scores -> "
        #     f"CompositeQoE: {composite_qoe:.4f}, TF-QoE: {tf_qoe:.4f}, "
        #     f"QFS: {qfs:.4f}, BFI: {bfi:.4f}, SAF: {saf:.4f}"
        # )
        # logging.debug(
        #     f"📊 Client {client_id} Samples: "
        #     f"composite_qoe_samples={metrics['composite_qoe_samples']}, "
        #     f"tf_qoe_samples={metrics['tf_qoe_samples']}, "
        #     f"qfs_samples={metrics['qfs_samples']}, "
        #     f"bfi_samples={metrics['bfi_samples']}, "
        #     f"saf_samples={metrics['saf_samples']}"
        # )

    except Exception as e:
        logging.error(f"Client {client_id}: Failed to update fairness scores - {e}")


# == CALCULATING TF-QOE []==
def calculate_tf_qoe_old(client_metrics, epsilon=1e-6):
    """
    Calculate the TF-QoE (Time-Fair QoE) for all clients at the current time.

    :param client_metrics: Dictionary containing client metrics with 'composite_qoe' for each client.
    :param epsilon: Small constant to avoid division by zero.
    :return: Dictionary with TF-QoE values for each client.
    """
    try:
        composite_qoes = []

        for client_id, metrics in client_metrics.items():
            qoe = metrics.get('composite_qoe', 0)
            composite_qoes.append(qoe)

        if not composite_qoes:
            logging.warning("⚠️ TF-QoE: No Composite QoE values found.")
            return {client_id: 0 for client_id in client_metrics.keys()}

        avg_qoe = sum(composite_qoes) / len(composite_qoes)

        tf_qoe_scores = {}

        for client_id, metrics in client_metrics.items():
            qoe = metrics.get('composite_qoe', 0)

            tf_qoe = 1 - (abs(qoe - avg_qoe) / (avg_qoe + epsilon))

            # Garante que TF-QoE fique no intervalo [0, 1]
            tf_qoe = max(0, min(tf_qoe, 1))

            tf_qoe_scores[client_id] = tf_qoe

            # Atualiza em client_metrics diretamente
            client_metrics[client_id]['tf_qoe'] = tf_qoe

        # logging.info(f"✅ Calculated TF-QoE scores: {tf_qoe_scores}")
        return tf_qoe_scores

    except Exception as e:
        logging.error(f"🚨 Error calculating TF-QoE: {e}")
        return {client_id: 0 for client_id in client_metrics.keys()}


def calculate_tf_qoe(client_metrics, epsilon=1e-6):
    """
    Calculate the TF-QoE (Time-Fair QoE) for all clients at the current time.
    Based on the definition: TF-QoE_i = QoE_i / max(QoE_j)
    """
    try:
        composite_qoes = [
            metrics.get('composite_qoe', 0) 
            for metrics in client_metrics.values()
        ]

        if not composite_qoes:
            logging.warning("⚠️ TF-QoE: No Composite QoE values found.")
            return {client_id: 0 for client_id in client_metrics.keys()}

        max_qoe = max(composite_qoes)

        tf_qoe_scores = {}
        for client_id, metrics in client_metrics.items():
            qoe = metrics.get('composite_qoe', 0)

            tf_qoe = qoe / (max_qoe + epsilon)

            # Garante intervalo [0,1]
            tf_qoe = max(0, min(tf_qoe, 1))

            tf_qoe_scores[client_id] = tf_qoe
            client_metrics[client_id]['tf_qoe'] = tf_qoe

        return tf_qoe_scores

    except Exception as e:
        logging.error(f"🚨 Error calculating TF-QoE: {e}")
        return {client_id: 0 for client_id in client_metrics.keys()}


# == CALCULATING QoE Fairness Score ==
def calculate_qfs(client_id, client_metrics, mode="online"):
    """
    Calculate QoE Fairness Score (QFS) using Jain's Fairness Index.
    
    Modes:
        - "online" (default): uses the most recent QoE sample per client
        - "final": same as online but meant for end-of-simulation reporting
    """
    try:
        qoes = []

        for cid, metrics in client_metrics.items():
            samples = metrics.get('composite_qoe_samples', [])
            if not samples:
                qoe_value = 0
            else:
                qoe_value = samples[-1]  # last sample (online/final)
            qoes.append(qoe_value)

        if not qoes or all(q == 0 for q in qoes):
            logging.warning("⚠️ QFS: No non-zero QoE values found. Defaulting to QFS=1 (perfect fairness)")
            qfs = 1.0
        else:
            numerator = (sum(qoes)) ** 2
            denominator = len(qoes) * sum([q ** 2 for q in qoes])
            qfs = 1.0 if denominator == 0 else numerator / denominator
            qfs = max(0, min(qfs, 1))

        # ✅ Armazena QFS em cada cliente
        for cid in client_metrics.keys():
            client_metrics[cid]['qfs'] = qfs

        logging.info(f"🔄 Calculated QFS score ({mode} mode): {qfs}")

        # ✅ Retorna no mesmo formato da versão antiga
        return {cid: qfs for cid in client_metrics.keys()}

    except Exception as e:
        logging.error(f"🚨 Error calculating QFS: {e}")
        # ✅ Mesmo fallback da versão antiga
        return {cid: 0 for cid in client_metrics.keys()}


# Old QFS calculation based on average QoE samples, not the most recent one
def calculate_qfs_old(client_id,client_metrics):
    """
    Calculate QoE Fairness Score (QFS) using Jain's Fairness Index formula
    over the average QoE scores actually delivered per client.
    """
    try:
        average_qoes = []

        for client_id, metrics in client_metrics.items():
            samples = metrics.get('composite_qoe_samples', [])
            if not samples:
                avg_qoe = 0
            else:
                avg_qoe = sum(samples) / len(samples)
            average_qoes.append(avg_qoe)

        if not average_qoes or all(q == 0 for q in average_qoes):
            logging.warning("⚠️ QFS: No non-zero QoE averages found. Defaulting to QFS=1 (perfect fairness)")
            qfs = 1.0
        else:
            numerator = (sum(average_qoes)) ** 2
            denominator = len(average_qoes) * sum([q ** 2 for q in average_qoes])
            if denominator == 0:
                qfs = 1.0
            else:
                qfs = numerator / denominator
                qfs = max(0, min(qfs, 1))

        # Optional: store back into metrics if you want
        for client_id in client_metrics.keys():
            client_metrics[client_id]['qfs'] = qfs

        logging.info(f"✅ Calculated QFS score: {qfs}")
        return {client_id: qfs for client_id in client_metrics.keys()}

    except Exception as e:
        logging.error(f"🚨 Error calculating QFS: {e}")
        return {client_id: 0 for client_id in client_metrics.keys()}


# == CALCULATING Bitrate Fairness Index (BFI) ==


QUALITY_MAP = {
    "360p": 1,
    "480p": 2,
    "720p": 3,
    "1080p": 4,
    "2160p": 5 
}


def calculate_bfi_old(client_id, client_metrics):
    """
    Calculate Bitrate Fairness Index (BFI) using Jain's Fairness Index formula
    over the average quality levels actually delivered per client.
    """
    try:
        average_qualities = []

        for client_id, metrics in client_metrics.items():
            samples = metrics.get('quality_level_samples', [])
            logging.info(f"Client {client_id}: Raw quality_level_samples: {samples}")

            if not samples:
                avg_quality = 0
            else:
                numeric_samples = [QUALITY_MAP.get(q, 0) for q in samples]
                logging.info(f"Client {client_id}: Converted numeric_samples (BFI): {numeric_samples}")

                if numeric_samples:
                    avg_quality = sum(numeric_samples) / len(numeric_samples)
                else:
                    avg_quality = 0

            average_qualities.append(avg_quality)

        if not average_qualities or all(q == 0 for q in average_qualities):
            logging.warning("⚠️ BFI: No non-zero quality averages found. Defaulting to BFI=1 (perfect fairness)")
            bfi = 1.0
        else:
            numerator = (sum(average_qualities)) ** 2
            denominator = len(average_qualities) * sum([q ** 2 for q in average_qualities])
            if denominator == 0:
                bfi = 1.0
            else:
                bfi = numerator / denominator
                bfi = max(0, min(bfi, 1))

        for client_id in client_metrics.keys():
            client_metrics[client_id]['bfi'] = bfi

        logging.info(f"✅ Calculated BFI score for client {client_id}: {bfi}")
        return {client_id: bfi for client_id in client_metrics.keys()}

    except Exception as e:
        logging.error(f"🚨 Error calculating BFI: {e}")
        return {client_id: 0 for client_id in client_metrics.keys()}



def calculate_bfi(client_id, client_metrics, use_time_weighting=True, default_to_one=True, normalize_units=True):
    """
    Calcula o Bitrate Fairness Index (BFI) conforme definido no artigo,
    aplicando o Índice de Jain sobre os valores médios de throughput (R_i) de cada cliente.

    Diferenciais em relação à versão anterior:
      - Utiliza diretamente o throughput médio (kbps) por cliente, evitando distorções
        causadas pelo uso de índices de qualidade não-lineares.
      - Permite cálculo mais preciso via média ponderada no tempo, usando
        throughput_samples + timestamp_samples (se disponíveis e use_time_weighting=True).
      - Mantém consistência com a definição teórica do paper:
          BFI = ( (Σ R_i)^2 ) / ( N * Σ R_i^2 )
        onde R_i é o throughput médio recebido pelo cliente i.
      - Adota a convenção de atribuir BFI=1.0 quando não há dados válidos,
        alinhado ao tratamento do SAF.
      - Inclui opção de normalizar unidades (bps → kbps) para garantir consistência.

    Parâmetros:
      - client_metrics: dicionário com métricas por cliente.
      - use_time_weighting (bool): ativa média ponderada no tempo, se possível.
      - default_to_one (bool): define BFI=1.0 quando não há dados válidos.
      - normalize_units (bool): ativa normalização automática de unidades para kbps.

    Efeitos:
      - Grava o valor global de BFI em cada entrada de client_metrics['bfi'].
      - Retorna um dicionário {client_id: bfi}.
    """
    import math

    def _safe_float(x):
        try:
            v = float(x)
            if math.isnan(v):
                return None
            return v
        except (TypeError, ValueError):
            return None

    def _parse_series(val):
        # aceita lista real ou string "a,b,c" / "[a, b, c]"
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [v for v in (_safe_float(x) for x in val) if v is not None]
        s = str(val).strip()
        if not s:
            return []
        # tenta split por vírgula (mais rápido/robusto para seus CSVs)
        parts = [p.strip() for p in s.split(",")]
        return [v for v in (_safe_float(p) for p in parts) if v is not None]

    def _to_kbps(x):
        # heurística: se parecer bps grande, converte para kbps
        # (ex.: valores na casa de 1e6 → ~1000 kbps)
        if not normalize_units or x is None:
            return x
        if x >= 1e5:   # ~ >= 100 kbps em bps
            return x / 1000.0
        return x

    # 1) Computa R_i por cliente (kbps)
    per_client_R = []
    for cid, m in client_metrics.items():
        # caminho principal: avg_throughput
        Ri = _safe_float(m.get("avg_throughput"))

        # opcional: média ponderada no tempo se habilitado e houver amostras
        if use_time_weighting:
            thr = _parse_series(m.get("throughput_samples"))
            tss = _parse_series(m.get("timestamp_samples"))
            if thr and tss and len(thr) == len(tss):
                # ordena por timestamp (caso fora de ordem)
                pairs = sorted(zip(tss, thr), key=lambda x: x[0])
                tss_sorted = [p[0] for p in pairs]
                thr_sorted = [p[1] for p in pairs]

                # pesos = delta_t entre amostras; último ponto não tem delta ⇒ ignora
                dt = []
                for i in range(len(tss_sorted) - 1):
                    d = tss_sorted[i+1] - tss_sorted[i]
                    d = _safe_float(d)
                    dt.append(d if d is not None and d > 0 else 0.0)

                if dt and sum(dt) > 0:
                    # média ponderada por Δt (usa throughput até a próxima amostra)
                    weighted_sum = sum(v * w for v, w in zip(thr_sorted[:-1], dt))
                    Ri_tw = weighted_sum / sum(dt)
                    Ri = Ri_tw if Ri_tw is not None else Ri

        # normaliza unidades (se necessário) para kbps
        Ri = _to_kbps(Ri) if Ri is not None else 0.0
        per_client_R.append(Ri if Ri is not None else 0.0)

    # 2) Jain sobre {R_i}
    valid = [x for x in per_client_R if x is not None]
    if not valid or all(x == 0.0 for x in valid):
        bfi = 1.0 if default_to_one else 0.0
    else:
        s = sum(valid)
        s2 = sum(x * x for x in valid)
        n = len(valid)
        bfi = (s * s) / (n * s2) if s2 > 0 else (1.0 if default_to_one else 0.0)
        bfi = max(0.0, min(1.0, bfi))

    # 3) grava em todos os clientes (como você já faz)
    for cid in client_metrics.keys():
        client_metrics[cid]["bfi"] = bfi

    return {cid: bfi for cid in client_metrics.keys()}



# == CALCULATING Stability Aware Fairness (SAF) ==
def calculate_saf(client_id, client_metrics, epsilon=1e-9, alpha=0.30):
    """
    Calcula:
      - SAF_global (índice do grupo, 1=melhor)
      - SAF_local_i (por cliente, 1=melhor)
      - fairness_deficit_i (desvio relativo de trocas/segmento)
    q_i = quality_changes_total_i / max(1, segment_index_i)

    Persiste em client_metrics[cid]:
      saf, saf_global, saf_local, fairness_deficit
      e em saf_samples (global) para compatibilidade.
    Retorna dict {cid: saf_global} (mesmo valor p/ todos).
    """
    # 1) coleta q_i (trocas por segmento)
    ids, Q = [], []
    for cid, m in client_metrics.items():
        if cid == '_global': 
            continue
        seg_idx = int(m.get('segment_index', 0))
        denom = max(1, seg_idx)
        qtot = max(0, int(m.get('quality_changes_total', 0)))
        qps  = qtot / float(denom)
        ids.append(cid)
        Q.append(qps)

    n = len(Q)
    if n == 0:
        return {}

    s1, s2 = sum(Q), sum(q*q for q in Q)
    if s2 <= epsilon:
        jfi, mean_q = 1.0, 0.0
    else:
        jfi    = (s1 * s1) / (n * (s2 + epsilon))
        jfi    = max(0.0, min(1.0, jfi))
        mean_q = s1 / n

    # 2) SAF global
    saf_global = jfi * math.exp(-alpha * mean_q)
    saf_global = max(0.0, min(1.0, saf_global))

    # 3) métricas por cliente
    out = {}
    for cid, q_i in zip(ids, Q):
        # desvio relativo do cliente em relação à média
        deficit = (q_i - mean_q) / (mean_q + epsilon)

        # SAF local (equidade local × estabilidade do próprio cliente)
        saf_local = (1.0 - abs(q_i - mean_q) / (mean_q + epsilon)) * math.exp(-alpha * q_i)
        saf_local = max(0.0, min(1.0, saf_local))

        cm = client_metrics[cid]
        # compat: 'saf' já usado no CSV → mantenha igual ao global
        cm['saf'] = saf_global
        cm['saf_global'] = saf_global
        cm['saf_local'] = saf_local
        cm['fairness_deficit'] = deficit
        cm.setdefault('saf_samples', []).append(saf_global)

        out[cid] = saf_global

    # (opcional debug global)
    # g = client_metrics.setdefault('_global', {})
    # g['saf_jfi'] = jfi
    # g['saf_mean_q_per_segment'] = mean_q
    # g.setdefault('saf_series', []).append(saf_global)

    return out


# == FAIRNESS CALCULATION ==
def calculate_fairness_index(metric_values):
    """
    Calcula o índice de justiça de Jain para uma métrica específica.
    Se todos os valores forem zero, retorna 1 (situação ideal).
    """
    metric_values = np.array(metric_values)

    # Se todos os valores forem zero, retorna fairness = 1
    if np.all(metric_values == 0):
        return 1.0

    sum_of_values = np.sum(metric_values)
    sum_of_squares = np.sum(np.square(metric_values))
    
    if sum_of_squares == 0:
        return 0  # Evita divisão por zero
    
    fairness = (sum_of_values ** 2) / (len(metric_values) * sum_of_squares)
    return fairness


# == CALCULATING FAIRNESS FOR METRICS ==
def calculate_fairness_metrics(client_metrics):
    """
    Calculate fairness metrics for each client and each QoE metric using Jain's Fairness Index.

    :param client_metrics: Dictionary containing QoE metrics for each client.
    :return: Updated client_metrics with fairness metrics for each metric and overall fairness.
    """
    print("Calculating QOE FAIRNESS METRICS")
    print("============================================================")

    # Inicializar dicionário para armazenar fairness por métrica
    fairness_results = {}

    # Lista das métricas a serem calculadas
    fairness_metrics = ['buffering_time', 'playback_interruptions', 'avg_throughput', 'avg_latency', 
                        'jitter', 'avg_buffer_level', 'quality_changes_total', 'dropped_frames_total']

    # Calcular Fairness para cada métrica, por cliente
    for metric in fairness_metrics:
        # Criar uma lista de valores para a métrica atual
        metric_values = [metrics.get(metric, 0) for client_id, metrics in client_metrics.items()]

        # Verificar se há listas aninhadas e converter para valores únicos
        metric_values = [np.mean(val) if isinstance(val, list) else val for val in metric_values]

        # Garantir fairness = 1 para métricas sem eventos registrados
        if metric in ['buffering_time', 'playback_interruptions'] and all(v == 0 for v in metric_values):
            fairness_results[metric] = 1.0
        else:
            fairness_results[metric] = calculate_fairness_index(metric_values)

        # Associar o Fairness global da métrica a cada cliente
        for client_id, metrics in client_metrics.items():
            metrics[f'fairness_{metric}'] = fairness_results[metric]

    # Calcular Fairness Geral (Overall Fairness) com base em Composite QoE
    composite_qoes = [metrics.get('composite_qoe', 0) for client_id, metrics in client_metrics.items()]
    overall_fairness = calculate_fairness_index(composite_qoes)

    # Adicionar Fairness Geral a cada cliente
    for client_id, metrics in client_metrics.items():
        metrics['fairness_overall'] = overall_fairness

    # Logar resultados para depuração
    print("Jain's Fairness Index for each normalized metric:")
    for metric, fairness in fairness_results.items():
        print(f"{metric}: {fairness:.4f}")
    print(f"Overall QoE Fairness: {overall_fairness:.4f}\n")

    # Inspecionar client_metrics
    print("\nClient Metrics after Fairness Calculation:")
    for client_id, metrics in client_metrics.items():
        print(f"Client {client_id}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("\n")

    return client_metrics

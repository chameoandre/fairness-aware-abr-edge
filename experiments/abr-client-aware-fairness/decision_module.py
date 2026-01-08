
import logging
import numpy as np

def adaptive_fairness_decision(action_rl, urgency_level, fairness_gaps, urgency_level_enabled, max_quality_level=5):
    """
    Adjust RL action using client's urgency level and fairness gaps across multiple metrics.

    :param action_rl: Action predicted by the RL model (quality level).
    :param urgency_level: Client priority (0 = low, 1 = medium, 2 = high).
    :param fairness_gaps: Dictionary with gaps between individual and global fairness metrics.
                          Example: {'tf_qoe': -0.1, 'qfs': 0.3, 'bfi': -0.05, 'saf': 0.15}
    :return: Adjusted action (integer).
    """
    # === URGENCY LEVEL DEACTIVATION FLAG ===
    if not urgency_level_enabled:
        logging.warning(f"Urgency level DISABLED - policy Flag ->>> {urgency_level_enabled}")
        return int(action_rl)

    penalizing_gap_threshold = 0.2  # negative adjustement if exceeded
    compensating_gap_threshold = -0.15  # positive adjustement if client is well below


    # Contador de gaps acima ou abaixo da média
    positive_gaps = sum(1 for g in fairness_gaps.values() if g > penalizing_gap_threshold)
    negative_gaps = sum(1 for g in fairness_gaps.values() if g < compensating_gap_threshold)

    # Penalizar clientes com vários gaps positivos e baixa prioridade
    if positive_gaps >= 2 and urgency_level < 2:
        adjusted = max(action_rl - 1, 0)
        return adjusted

    # Recompensar clientes com vários gaps negativos e alta urgência
    if negative_gaps >= 2 and urgency_level == 2:
        adjusted = min(action_rl + 1, max_quality_level)
        return adjusted

    # Sem ajuste necessário
    return int(action_rl)



def compute_quality_level(state_vector, model, urgency_level, weights, global_score,urgency_level_enabled):
    """
    Decide the quality level using the RL model and applies fairness-aware adjustments.

    :param state_vector: List with the current state [QoE + Fairness metrics]
    :param model: Trained RL model
    :param urgency_level: Priority level of the client (0=low, 1=medium, 2=high)
    :param weights: Dict of weights used in the state (not directly used here, but can help future extensions)
    :param global_score: Dict with aggregated scores: {'tf_qoe', 'qfs', 'bfi', 'saf'}
    :return: Adjusted quality level (integer)
    """
    logging.info('===== Inside COMPUTE QUALITY LEVEL routine ===== ')
    logging.info(f"Client State Vector: {state_vector}")
    logging.info(f"Global Scores: {global_score}")
    logging.info(f"Urgency Level: {urgency_level}")
    logging.info(f"Wheights: {weights}")
    logging.info(f"Urgency Level Enabled: {urgency_level_enabled}")


    logging.info("--- INSIDE COMPUTE QUALITY LEVEL ---- ")


    try:
        # Predict action using RL model
        action_rl, _ = model.predict(np.array(state_vector).reshape(1, -1), deterministic=True)
        logging.info(f"Predicted quality level (RL raw action): {action_rl}")
    except Exception as e:
        raise RuntimeError(f"[compute_quality_level] Error during RL model prediction: {e}")

    # === Extract individual fairness scores from state_vector ===
    try:
        tf_qoe = state_vector[6]
        qfs = state_vector[7]
        bfi = state_vector[8]
        saf = state_vector[9]
    except IndexError:
        raise ValueError(f"[compute_quality_level] Invalid state_vector: missing fairness scores (got {len(state_vector)} elements)")

    # === Calculate fairness gaps ===
    fairness_gaps = {
        'tf_qoe': tf_qoe - global_score.get('tf_qoe', 0),
        'qfs':    qfs    - global_score.get('qfs', 0),
        'bfi':    bfi    - global_score.get('bfi', 0),
        'saf':    saf    - global_score.get('saf', 0)
    }

    logging.info(f"Fairness gaps: {fairness_gaps}")

    # === Final quality decision with fairness-aware logic ===
    adjusted_action = adaptive_fairness_decision(
        action_rl=action_rl,
        urgency_level=urgency_level,
        urgency_level_enabled=urgency_level_enabled,
        fairness_gaps=fairness_gaps
    )

    logging.info(f"Adjusted quality level after fairness policy: {adjusted_action}")
    return int(adjusted_action)
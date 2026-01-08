# hotdash.py
# Implementation based on paper description: 
# "HotDash: Hot Data-Aware Rate Adaptation for Dynamic Adaptive Streaming Over HTTP"
# DOI: 10.1109/ICC.2018.8422625

def simulate_popularity_map(total_segments=60):
    """
    Simula mapa de popularidade dos chunks.
    1 = hot (popular), 0 = cold (menos popular)
    """
    # Exemplo: segmentos 0–9, 20–29 e 45–49 são populares
    hot_segments = set(range(0, 10)) | set(range(20, 30)) | set(range(45, 50))
    return [1 if i in hot_segments else 0 for i in range(total_segments)]


def hotdash_selector(state, max_quality_level, segment_index, total_segments=60):
    """
    Implementação baseada no artigo HotDash.
    Recebe o estado do cliente, índice do segmento e total de qualidades.

    Seleciona a qualidade do segmento atual com base na lógica do HotDash.
    Entradas:
        - throughput_history: lista de valores em kbps
        - buffer_level: valor atual de buffer (segundos)
        - max_quality_level: número máximo de níveis de qualidade
        - segment_index: índice do segmento atual
    Saída:
        - índice de qualidade (int)
    """

    # === 1. Algorithm parameters ===
    margin_factor = 1.3              # margem de segurança para throughput
    buffer_threshold = 10            # limite mínimo de buffer seguro (segundos)
    bitrate_levels = [300, 750, 1200, 1850, 2850, 4300]  # bitrates em kbps #needs post checking


    # === 2. Check segment popularity === (poderia vir de servidor no futuro)
    popularity_map = simulate_popularity_map(total_segments=total_segments)
    is_hot = popularity_map[segment_index] if segment_index < len(popularity_map) else 0

    # === 3. Estimates average throughput from history
    samples = state.get("throughput_samples", [])
    if samples and len(samples) >= 1:
        estimated_throughput = sum(samples[-5:]) / min(5, len(samples))
    else:
        estimated_throughput = state.get("avg_throughput", 1000)  # fallback seguro em kbps
    
    buffer_level = state.get("avg_buffer_level", 0)

    # === 4. Define bitrate alvo com margem ===
    target_bitrate = estimated_throughput / margin_factor


   # === 5. Seleciona qualidade com base no target_bitrate ===
    chosen_quality = 0
    for i, br in enumerate(bitrate_levels):
        if br <= target_bitrate:
            chosen_quality = i

    # === 6. Penalizações ===
    if not is_hot:
        chosen_quality = max(0, chosen_quality - 1)
    if buffer_level < buffer_threshold:
        chosen_quality = max(0, chosen_quality - 1)


    # === 7. Garante faixa válida ===
    chosen_quality = min(chosen_quality, max_quality_level - 1)

    # === 8. Logs auxiliares ===
    print(f"[HotDash] Segment {segment_index} {'HOT' if is_hot else 'COLD'} | "
          f"Estimated TP: {round(estimated_throughput, 2)} kbps | "
          f"Buffer: {buffer_level:.2f}s | "
          f"Target BR: {round(target_bitrate, 2)} | "
          f"Chosen Quality: {chosen_quality}")
    

    # === 9. Return selected quality level
    return chosen_quality

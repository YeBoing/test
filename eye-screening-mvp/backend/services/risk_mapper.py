def level_from_score(score: float) -> str:
    if score < 0.25:
        return "none"
    if score < 0.5:
        return "low"
    if score < 0.75:
        return "medium"
    return "high"


def recommendation(dr_level: str, htn_level: str) -> str:
    levels = [dr_level, htn_level]
    if "high" in levels:
        return "建议尽快前往眼科/专科医院进一步检查（建议 1 周内）。"
    if "medium" in levels:
        return "建议 1-3 个月内复查，并咨询专科医生。"
    if "low" in levels:
        return "建议保持生活方式管理，3-6 个月复查。"
    return "当前未见明显风险提示，建议年度常规复查。"

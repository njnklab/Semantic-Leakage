"""
Depression Cue Categories
抑郁线索类别定义
"""
from typing import Dict, List, Set


# 英文抑郁关键词
ENGLISH_CATEGORIES = {
    "mood": {
        "keywords": ["sad", "depressed", "down", "low", "hopeless", "empty", "numb", "cry", "tear", "blue", "gloomy", "melancholy"],
        "description": "Depressed mood or negative emotions",
    },
    "sleep": {
        "keywords": ["insomnia", "sleep", "awake", "wakeup", "tired", "rest", "bed", "night", "dream", "sleepless"],
        "description": "Sleep disturbance",
    },
    "fatigue": {
        "keywords": ["tired", "exhausted", "fatigue", "weak", "energy", "lethargic", "drained", "burnout", "weary"],
        "description": "Fatigue or energy loss",
    },
    "appetite": {
        "keywords": ["appetite", "eat", "hunger", "weight", "food", "nausea", "vomit", "starve", "overeating"],
        "description": "Appetite or weight changes",
    },
    "guilt": {
        "keywords": ["guilt", "guilty", "shame", "worthless", "failure", "blame", "regret", "burden", "useless"],
        "description": "Feelings of worthlessness or guilt",
    },
    "suicide": {
        "keywords": ["suicide", "suicidal", "die", "death", "kill", "hurt", "harm", "end", "meaningless", "alive", "not worth living"],
        "description": "Thoughts of death or self-harm",
    },
    "cognition": {
        "keywords": ["concentrate", "focus", "attention", "forget", "confused", "thinking", "memory", "foggy", "distracted"],
        "description": "Concentration difficulties",
    },
    "interest": {
        "keywords": ["interest", "boring", "enjoy", "pleasure", "hobby", "fun", "motivation", "excited", "enthusiasm"],
        "description": "Loss of interest or pleasure",
    },
    "anxiety": {
        "keywords": ["anxiety", "anxious", "worry", "fear", "panic", "nervous", "scared", "restless", "uneasy"],
        "description": "Anxiety symptoms",
    },
    "social": {
        "keywords": ["isolate", "alone", "lonely", "withdraw", "friends", "social", "avoid", "reject", "secluded"],
        "description": "Social withdrawal",
    },
    "physical": {
        "keywords": ["pain", "ache", "headache", "chest", "heart", "tremble", "sweat", "dizzy", "somatic"],
        "description": "Physical symptoms",
    },
}


# 中文抑郁关键词
CHINESE_CATEGORIES = {
    "mood": {
        "keywords": ["抑郁", "难过", "伤心", "绝望", "低落", "沮丧", "悲观", "空虚", "麻木", "哭泣", "悲伤"],
        "description": "情绪低落",
    },
    "sleep": {
        "keywords": ["失眠", "睡不着", "早醒", "睡眠", "困倦", "疲倦", "休息", "噩梦", "入睡困难"],
        "description": "睡眠问题",
    },
    "fatigue": {
        "keywords": ["疲劳", "乏力", "没劲", "倦怠", "虚弱", "精力不足", "累", "疲惫", "无力"],
        "description": "疲劳或精力不足",
    },
    "appetite": {
        "keywords": ["食欲", "吃不下", "暴食", "体重", "食物", "恶心", "呕吐", "饥饿", "没胃口"],
        "description": "食欲变化",
    },
    "guilt": {
        "keywords": ["自责", "内疚", "无用", "废物", "失败", "后悔", "拖累", "愧疚", "负罪"],
        "description": "自责或内疚",
    },
    "suicide": {
        "keywords": ["自杀", "想死", "轻生", "不想活", "结束生命", "死亡", "伤害自己", "了结", "没有意义"],
        "description": "自杀想法",
    },
    "cognition": {
        "keywords": ["注意力不集中", "走神", "忘记", "混乱", "思维", "记忆", "迟钝", "糊涂"],
        "description": "认知问题",
    },
    "interest": {
        "keywords": ["没兴趣", "无聊", "不喜欢", "没乐趣", "爱好", "动力", "快感缺失", "索然无味"],
        "description": "兴趣丧失",
    },
    "anxiety": {
        "keywords": ["焦虑", "紧张", "害怕", "恐惧", "恐慌", "担心", "不安", "心慌", "烦躁"],
        "description": "焦虑症状",
    },
    "social": {
        "keywords": ["孤立", "独处", "孤独", "回避", "朋友", "社交", "躲避", "拒绝", "封闭"],
        "description": "社交退缩",
    },
    "physical": {
        "keywords": ["头痛", "胸闷", "疼痛", "心慌", "手抖", "出汗", "头晕", "身体不适"],
        "description": "躯体症状",
    },
}


def get_categories(language: str = "en") -> Dict[str, Dict]:
    """获取类别定义"""
    if language == "zh":
        return CHINESE_CATEGORIES
    return ENGLISH_CATEGORIES


def get_all_keywords(language: str = "en") -> Set[str]:
    """获取所有关键词"""
    categories = get_categories(language)
    keywords = set()
    for cat in categories.values():
        keywords.update(cat["keywords"])
    return keywords


def categorize_cue(text: str, language: str = "en") -> str:
    """
    对线索文本进行分类

    Returns:
        类别名称，如 "mood", "sleep" 等
    """
    text_lower = text.lower() if language == "en" else text
    categories = get_categories(language)

    scores = {}
    for cat_name, cat_info in categories.items():
        score = sum(1 for kw in cat_info["keywords"] if kw in text_lower)
        if score > 0:
            scores[cat_name] = score

    if not scores:
        return "depression_related"

    return max(scores, key=scores.get)

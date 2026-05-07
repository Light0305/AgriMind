#!/usr/bin/env python3
"""
generate_agrireason.py — AgriReason Benchmark 自动生成
=====================================================
生成首个农业多模态推理评测数据集 (500-800 samples)。
利用 VLM 自动生成推理链, 并用自一致性过滤保证质量。

任务类型分布:
  disease       35%  单一病害诊断
  pest          25%  虫害识别
  differential  20%  相似病害鉴别诊断
  nutrient      10%  营养缺素诊断
  uncertainty   10%  模糊/信息不足 → 表达不确定性

用法:
  python generate_agrireason.py \\
      --input data/processed/unified_dataset.jsonl \\
      --output benchmark/data/agrireason_v1.json \\
      --api-key DASHSCOPE_KEY \\
      --max-samples 800 --consistency-runs 3

  python generate_agrireason.py \\
      --input data/processed/unified_dataset.jsonl \\
      --output benchmark/data/agrireason_v1.json \\
      --use-local --max-samples 500
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# 作物种植上下文 (无需 API, 本地随机生成)
# ---------------------------------------------------------------------------
CROP_CONTEXTS: dict[str, dict[str, Any]] = {
    "番茄": {
        "regions": ["华北", "华东", "西北", "华中"],
        "seasons": ["6-9月"],
        "weathers": ["近期多雨", "高温高湿", "连续阴雨", "气温骤降后回升"],
    },
    "马铃薯": {
        "regions": ["华北", "东北", "西北", "西南"],
        "seasons": ["5-8月"],
        "weathers": ["近期多雨", "昼夜温差大", "连续阴天", "露水频繁"],
    },
    "苹果": {
        "regions": ["华北", "西北", "山东", "陕西"],
        "seasons": ["4-9月"],
        "weathers": ["近期多雨", "连续阴雨", "高温干旱后降雨", "春季倒春寒"],
    },
    "葡萄": {
        "regions": ["华北", "西北", "华东", "新疆"],
        "seasons": ["5-9月"],
        "weathers": ["高温高湿", "近期多雨", "通风不良", "连续阴天"],
    },
    "玉米": {
        "regions": ["华北", "东北", "华中", "西南"],
        "seasons": ["6-9月"],
        "weathers": ["高温多雨", "连续阴雨", "暴风雨后", "干旱后灌溉"],
    },
    "辣椒": {
        "regions": ["华中", "西南", "华南", "华东"],
        "seasons": ["5-9月"],
        "weathers": ["高温多雨", "连续阴雨", "浇水过多", "通风不良"],
    },
    "草莓": {
        "regions": ["华东", "华北", "华中", "西南"],
        "seasons": ["3-5月"],
        "weathers": ["春雨频繁", "低温阴雨", "气温回升", "大棚高湿"],
    },
    "樱桃": {
        "regions": ["华北", "山东", "辽宁", "陕西"],
        "seasons": ["4-6月"],
        "weathers": ["近期多雨", "高温高湿", "通风不良", "果实成熟期多雨"],
    },
    "桃": {
        "regions": ["华北", "华东", "西北", "华中"],
        "seasons": ["4-8月"],
        "weathers": ["近期多雨", "春季多雨", "高温高湿", "连续阴雨"],
    },
    "南瓜": {
        "regions": ["华北", "华东", "西南", "西北"],
        "seasons": ["5-9月"],
        "weathers": ["干旱少雨", "通风不良", "昼夜温差大", "高温干燥"],
    },
    "大豆": {
        "regions": ["东北", "华北", "华中", "华东"],
        "seasons": ["6-9月"],
        "weathers": ["高温多雨", "连续阴雨", "暴雨后", "高湿闷热"],
    },
    "覆盆子": {
        "regions": ["华东", "华北", "东北", "西南"],
        "seasons": ["5-7月"],
        "weathers": ["近期多雨", "通风不良", "高湿", "春末阴雨"],
    },
    "蓝莓": {
        "regions": ["东北", "华东", "西南", "云南"],
        "seasons": ["4-7月"],
        "weathers": ["春雨频繁", "高湿", "近期多雨", "排水不良"],
    },
    "柑橘": {
        "regions": ["华南", "西南", "华中", "华东"],
        "seasons": ["全年"],
        "weathers": ["近期高温", "台风后", "连续阴雨", "冬季低温"],
    },
    "小麦": {
        "regions": ["华北", "西北", "华中", "华东"],
        "seasons": ["3-5月"],
        "weathers": ["春季多雨", "倒春寒", "灌浆期高温", "连续阴雨"],
    },
    "水稻": {
        "regions": ["华中", "华南", "华东", "西南"],
        "seasons": ["6-9月"],
        "weathers": ["高温高湿", "连续阴雨", "台风后", "暴雨频繁"],
    },
    "棉花": {
        "regions": ["新疆", "华北", "华中", "华东"],
        "seasons": ["6-9月"],
        "weathers": ["高温干旱", "近期灌溉", "连续阴雨", "暴风雨后"],
    },
}

# 默认上下文 (兜底)
_DEFAULT_CONTEXT = {
    "regions": ["华北", "华东", "华中", "华南", "西南", "西北", "东北"],
    "seasons": ["春季", "夏季", "秋季"],
    "weathers": ["近期多雨", "高温高湿", "气候正常", "干旱少雨"],
}

# ---------------------------------------------------------------------------
# 相似病害对 — 用于 differential 类型
# ---------------------------------------------------------------------------
SIMILAR_DISEASE_PAIRS: list[tuple[str, str]] = [
    ("番茄晚疫病", "番茄早疫病"),
    ("番茄叶霉病", "番茄晚疫病"),
    ("番茄细菌性斑点病", "番茄壳针孢叶斑病"),
    ("番茄靶斑病", "番茄早疫病"),
    ("苹果黑星病", "苹果黑腐病"),
    ("苹果黑腐病", "苹果雪松锈病"),
    ("葡萄黑腐病", "葡萄黑麻疹病"),
    ("葡萄叶枯病", "葡萄黑腐病"),
    ("马铃薯早疫病", "马铃薯晚疫病"),
    ("玉米灰斑病", "玉米北方叶枯病"),
    ("玉米普通锈病", "玉米灰斑病"),
    ("樱桃白粉病", "南瓜白粉病"),
]

# 根据中文标签找到配对的混淆病害
_DIFFERENTIAL_MAP: dict[str, list[str]] = defaultdict(list)
for a, b in SIMILAR_DISEASE_PAIRS:
    _DIFFERENTIAL_MAP[a].append(b)
    _DIFFERENTIAL_MAP[b].append(a)

# ---------------------------------------------------------------------------
# 营养缺素关键词 — 用于 nutrient 类型标签识别/模拟
# ---------------------------------------------------------------------------
NUTRIENT_KEYWORDS = ["缺素", "缺氮", "缺磷", "缺钾", "缺铁", "缺锌", "缺镁",
                     "nutrient", "deficiency", "nitrogen", "phosphorus",
                     "potassium", "iron", "zinc", "magnesium"]

# ---------------------------------------------------------------------------
# VLM 提示词
# ---------------------------------------------------------------------------
PROMPT_DISEASE = """\
你是资深植物病理学家。请对这张作物图片进行病害诊断，给出完整推理链。

要求按以下步骤输出（每步一段，不要使用markdown格式）：
1. 观察：描述图片中的具体症状特征（颜色、形态、分布位置等）
2. 分析：这些特征的病理学意义，指向什么病因
3. 排除：排除至少1个其他可能的病害，说明排除理由
4. 结论：最终诊断及置信度

已知参考标签: {label_cn}
{context_hint}
请基于图片实际特征给出专业中文推理过程，保证结论与参考标签一致。"""

PROMPT_PEST = """\
你是资深农业昆虫学家。请对这张作物图片进行虫害识别，给出完整推理链。

要求按以下步骤输出（每步一段，不要使用markdown格式）：
1. 观察：描述图片中昆虫或为害状的具体特征（形态、颜色、大小、为害部位等）
2. 分析：这些特征的昆虫学意义，指向什么害虫
3. 排除：排除至少1个其他可能的害虫，说明排除理由
4. 结论：最终识别结果及置信度

已知参考标签: {label_cn}
{context_hint}
请基于图片实际特征给出专业中文推理过程，保证结论与参考标签一致。"""

PROMPT_DIFFERENTIAL = """\
你是资深植物病理学家。这张图片中的作物可能患有 {label_cn} 或 {confusable}，\
两者症状相似。请对图片进行鉴别诊断。

要求按以下步骤输出（每步一段，不要使用markdown格式）：
1. 观察：详细描述图片中的症状特征
2. 对比分析：将观察到的特征分别与 {label_cn} 和 {confusable} 的典型症状进行对比
3. 关键鉴别点：指出区分两种病害最关键的特征差异
4. 排除理由：明确说明为什么不是 {confusable}
5. 结论：最终诊断为 {label_cn}，给出置信度

{context_hint}
基于图片实际特征给出专业推理。"""

PROMPT_NUTRIENT = """\
你是资深作物营养学家。请对这张作物图片进行营养缺素诊断，给出完整推理链。

要求按以下步骤输出（每步一段，不要使用markdown格式）：
1. 观察：描述叶片/植株的异常表现（黄化模式、坏死分布、生长状态等）
2. 营养分析：这些症状在营养学上的意义，可能缺乏什么元素
3. 排除：排除至少1种其他可能的缺素症状，说明理由
4. 结论：最终判断及建议补充方案

已知参考标签: {label_cn}
{context_hint}
请基于图片实际表现给出专业推理。"""

PROMPT_UNCERTAINTY = """\
你是资深植物病理学家。请观察这张作物图片并尝试诊断。

注意：这张图片可能存在以下情况之一，请务必如实反映你的判断：
- 图片模糊、光线不足，无法看清关键特征
- 植株表现健康，无明显病害症状
- 症状不典型，多种病害均有可能

要求按以下步骤输出（每步一段，不要使用markdown格式）：
1. 观察：描述图片中能看到/不能看到的特征
2. 分析：现有信息是否足以做出可靠诊断
3. 不确定性说明：明确表达信息不足或无法确诊的原因
4. 建议：建议采集更多信息（如更清晰照片、不同角度、病史等）

{context_hint}
核心要求：如果信息不足以确诊，必须诚实表达不确定性，不要强行给出诊断。"""

# ---------------------------------------------------------------------------
# 图片工具
# ---------------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str | None:
    """Read an image file and return base64 encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠ 无法读取图片 {image_path}: {e}")
        return None


def get_image_mime(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# 上下文生成 (本地随机, 无需 API)
# ---------------------------------------------------------------------------
def generate_context(crop: str) -> str:
    """为给定作物随机生成合理的种植上下文字符串."""
    ctx = CROP_CONTEXTS.get(crop, _DEFAULT_CONTEXT)
    region = random.choice(ctx["regions"])
    season_spec = random.choice(ctx["seasons"])

    # 将 "6-9月" 格式转为具体月份
    m = re.match(r"(\d+)-(\d+)月", season_spec)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        month = random.randint(start, end)
        season_str = f"{month}月{'上' if random.random() < 0.33 else '中' if random.random() < 0.5 else '下'}旬"
    elif season_spec == "全年":
        month = random.randint(1, 12)
        season_str = f"{month}月"
    else:
        season_str = season_spec

    weather = random.choice(ctx["weathers"])
    return f"该样本采自{region}地区，{season_str}，{weather}"


# ---------------------------------------------------------------------------
# VLM 后端
# ---------------------------------------------------------------------------
class DashScopeBackend:
    """DashScope MultiModalConversation API 后端."""

    def __init__(self, api_key: str, model: str = "qwen2.5-vl-72b-instruct"):
        self.api_key = api_key
        self.model = model
        self._lock = threading.Lock()
        self._request_times: list[float] = []
        self._rpm_limit = 15

    def _rate_limit(self):
        with self._lock:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            if len(self._request_times) >= self._rpm_limit:
                sleep_time = 60 - (now - self._request_times[0]) + 0.5
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._request_times.append(time.time())

    def generate(self, prompt: str, image_path: str) -> str | None:
        try:
            from dashscope import MultiModalConversation
        except ImportError:
            return self._generate_openai_compat(prompt, image_path)

        self._rate_limit()
        abs_path = str(Path(image_path).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{abs_path}"},
                    {"text": prompt},
                ],
            }
        ]
        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
            )
            return response.output.choices[0].message.content[0]["text"].strip()
        except Exception as e:
            print(f"  ⚠ DashScope API 调用失败: {e}")
            return None

    def _generate_openai_compat(self, prompt: str, image_path: str) -> str | None:
        try:
            from openai import OpenAI
        except ImportError:
            print("错误: 请安装 dashscope 或 openai 库")
            sys.exit(1)

        b64 = encode_image_base64(image_path)
        if b64 is None:
            return None
        mime = get_image_mime(image_path)
        self._rate_limit()

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ OpenAI 兼容 API 调用失败: {e}")
            return None


class LocalBackend:
    """本地 Qwen2.5-VL-7B 4-bit 量化推理后端."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        print("正在加载本地模型 Qwen2.5-VL-7B-Instruct (4-bit) ...")
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as e:
            print(f"错误: 缺少依赖: {e}")
            print("请安装: pip install torch transformers bitsandbytes accelerate qwen-vl-utils")
            sys.exit(1)

        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print("模型加载完成。")

    def generate(self, prompt: str, image_path: str) -> str | None:
        import torch
        from qwen_vl_utils import process_vision_info

        with self._lock:
            try:
                abs_path = str(Path(image_path).resolve())
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{abs_path}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=2000,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                    )
                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                result = self.processor.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return result.strip()
            except Exception as e:
                print(f"  ⚠ 本地推理失败: {e}")
                return None


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_input_data(input_file: str) -> list[dict]:
    """Load unified_dataset.jsonl records."""
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 推理链解析 — 将 VLM 自由文本拆成结构化步骤
# ---------------------------------------------------------------------------
_STEP_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(\d+)\s*[.、)\]]\s*"),                       # 1. / 1、
    re.compile(r"(?:^|\n)\s*(观察|分析|排除|结论|对比|鉴别|建议|不确定)[：:]\s*"),  # 中文关键词
]


def parse_reasoning_chain(text: str) -> list[str]:
    """将 VLM 生成的推理文本解析为推理链列表.

    尝试按编号 (1. 2. 3. 4.) 或关键词 (观察: 分析: 排除: 结论:) 分段。
    如果都不匹配则按自然段落分割。
    """
    # 尝试用编号分段
    numbered = re.split(r'\n\s*\d+\s*[.、)\]]\s*', text)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 3:
        return numbered

    # 尝试用中文关键词分段
    keywords = ["观察", "分析", "排除", "结论", "对比", "鉴别", "建议", "不确定", "关键"]
    segments = []
    current = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # 检查是否以关键词开头
        starts_new = False
        for kw in keywords:
            if stripped.startswith(kw) and len(stripped) > len(kw):
                if stripped[len(kw)] in "：:：":
                    starts_new = True
                    break
        if starts_new and current:
            segments.append(current.strip())
            current = stripped
        else:
            current += ("\n" if current else "") + stripped
    if current:
        segments.append(current.strip())
    if len(segments) >= 3:
        return segments

    # 兜底: 按段落分割
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs

    # 实在拆不开就整段返回
    return [text.strip()] if text.strip() else []


# ---------------------------------------------------------------------------
# 自一致性检验
# ---------------------------------------------------------------------------
def extract_diagnosis_label(text: str, label_cn: str) -> str:
    """提取文本中的诊断结论用于一致性比较."""
    if label_cn in text:
        return label_cn
    for keyword in ["最终诊断", "诊断为", "判断为", "确诊为", "结论", "最终识别"]:
        idx = text.find(keyword)
        if idx != -1:
            snippet = text[idx: idx + 80]
            return snippet.strip()
    return text[-100:].strip()


def run_with_consistency(
    backend, prompt: str, image_path: str,
    label_cn: str, n_runs: int = 3,
) -> str | None:
    """运行 n_runs 次, 仅在所有运行诊断一致时返回结果."""
    responses: list[str] = []
    labels: list[str] = []

    for _ in range(n_runs):
        resp = backend.generate(prompt, image_path)
        if resp is None:
            return None
        responses.append(resp)
        labels.append(extract_diagnosis_label(resp, label_cn))

    # 全部一致
    if len(set(labels)) == 1:
        return responses[0]

    # 多数投票: 至少 n_runs-1 票一致
    counter = Counter(labels)
    most_common_label, count = counter.most_common(1)[0]
    if count >= n_runs - 1:
        idx = labels.index(most_common_label)
        return responses[idx]

    return None  # 不一致, 丢弃


# ---------------------------------------------------------------------------
# 分层采样
# ---------------------------------------------------------------------------
def classify_record(rec: dict) -> str:
    """根据记录的标签内容分配初始任务类型候选."""
    label_cn = rec.get("label_cn", "")
    label_en = rec.get("label_en", "")
    source = rec.get("source", "")

    # 健康样本 → uncertainty 候选
    if "healthy" in label_en.lower() or "健康" in label_cn:
        return "uncertainty"

    # IP102 数据集 → pest
    if source == "IP102":
        return "pest"

    # 虫害关键词
    pest_keywords = ["螨", "蚜", "虫", "pest", "mite", "aphid", "insect",
                     "beetle", "moth", "larva", "fly", "bug"]
    combined = (label_cn + label_en).lower()
    for kw in pest_keywords:
        if kw in combined:
            return "pest"

    # 营养缺素
    for kw in NUTRIENT_KEYWORDS:
        if kw in combined:
            return "nutrient"

    # 有混淆对的 → differential 候选
    if label_cn in _DIFFERENTIAL_MAP:
        return "differential_candidate"

    # 默认 → disease
    return "disease"


def stratified_sample(
    records: list[dict],
    max_samples: int,
    seed: int = 42,
) -> list[dict]:
    """按任务类型分布做分层采样.

    目标分布: disease=35%, pest=25%, differential=20%, nutrient=10%, uncertainty=10%
    """
    rng = random.Random(seed)

    # 分桶
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        cat = classify_record(rec)
        buckets[cat].append(rec)

    # differential_candidate 属于 disease 类但可以抽取出来用于 differential
    # 保持一份在 disease 桶中, 一份在 differential 桶中
    diff_candidates = buckets.pop("differential_candidate", [])
    buckets["disease"].extend(diff_candidates)
    buckets["differential"] = list(diff_candidates)  # 副本

    # 各桶打乱
    for v in buckets.values():
        rng.shuffle(v)

    # 目标数量
    targets = {
        "disease": int(max_samples * 0.35),
        "pest": int(max_samples * 0.25),
        "differential": int(max_samples * 0.20),
        "nutrient": int(max_samples * 0.10),
        "uncertainty": int(max_samples * 0.10),
    }

    # 如果某类不够, 将余量分配给 disease
    selected: dict[str, list[dict]] = {}
    overflow = 0
    for task_type, target_n in targets.items():
        available = buckets.get(task_type, [])
        take = min(target_n, len(available))
        selected[task_type] = available[:take]
        overflow += target_n - take

    # 额外 disease 填补不足
    disease_extra = buckets.get("disease", [])[targets["disease"]:]
    selected["disease"].extend(disease_extra[:overflow])

    # 标注 task_type
    result = []
    for task_type, recs in selected.items():
        for rec in recs:
            annotated = dict(rec)
            annotated["_task_type"] = task_type
            # 为 differential 类型选取混淆病害
            if task_type == "differential":
                confusables = _DIFFERENTIAL_MAP.get(rec.get("label_cn", ""), [])
                annotated["_confusable"] = rng.choice(confusables) if confusables else None
            result.append(annotated)

    rng.shuffle(result)

    print(f"\n分层采样结果:")
    type_counts = Counter(r["_task_type"] for r in result)
    for t in ["disease", "pest", "differential", "nutrient", "uncertainty"]:
        print(f"  {t:<15} {type_counts.get(t, 0):>5} 条")
    print(f"  {'总计':<15} {len(result):>5} 条\n")

    return result


# ---------------------------------------------------------------------------
# 难度自动评估
# ---------------------------------------------------------------------------
def estimate_difficulty(
    task_type: str,
    reasoning_chain: list[str],
    label_cn: str,
) -> str:
    """根据任务类型和推理链长度估算难度."""
    if task_type == "differential":
        return "hard"
    if task_type == "uncertainty":
        return "hard"
    if task_type == "nutrient":
        return "medium"
    # disease / pest: 根据推理链长度
    total_len = sum(len(s) for s in reasoning_chain)
    if total_len > 500:
        return "hard"
    elif total_len > 200:
        return "medium"
    return "easy"


# ---------------------------------------------------------------------------
# 问题模板
# ---------------------------------------------------------------------------
QUESTION_TEMPLATES: dict[str, list[str]] = {
    "disease": [
        "请诊断这张图片中的作物病害，给出完整推理过程",
        "请分析这张作物图片的病害情况，逐步说明你的诊断依据",
        "观察这张图片中作物的症状，进行病害诊断并解释推理过程",
    ],
    "pest": [
        "请识别这张图片中的害虫或虫害症状，给出完整推理过程",
        "请分析这张作物图片中的虫害情况，逐步说明识别依据",
        "观察这张图片，识别可能的害虫并解释你的判断过程",
    ],
    "differential": [
        "请诊断这张图片中的作物病害，特别注意与相似病害的鉴别，给出完整推理过程",
        "这张图片中的作物症状可能对应多种病害，请进行鉴别诊断并解释排除理由",
    ],
    "nutrient": [
        "请分析这张作物图片可能的营养缺素症状，给出完整推理过程",
        "观察这张图片中作物的生长异常，判断是否存在营养缺素并说明理由",
    ],
    "uncertainty": [
        "请观察这张作物图片并尝试进行诊断，如果信息不足请如实说明",
        "请分析这张图片中作物的健康状况，给出你的判断及置信度",
    ],
}


# ---------------------------------------------------------------------------
# 单样本处理
# ---------------------------------------------------------------------------
def _build_prompt(rec: dict) -> str:
    """根据任务类型和记录信息构造 VLM 提示词."""
    task_type = rec["_task_type"]
    label_cn = rec.get("label_cn", "")
    crop = rec.get("crop", "未知作物")
    context = generate_context(crop)
    context_hint = f"种植背景: {context}"

    if task_type == "disease":
        return PROMPT_DISEASE.format(label_cn=label_cn, context_hint=context_hint), context
    elif task_type == "pest":
        return PROMPT_PEST.format(label_cn=label_cn, context_hint=context_hint), context
    elif task_type == "differential":
        confusable = rec.get("_confusable", "其他相似病害")
        return PROMPT_DIFFERENTIAL.format(
            label_cn=label_cn, confusable=confusable, context_hint=context_hint
        ), context
    elif task_type == "nutrient":
        return PROMPT_NUTRIENT.format(label_cn=label_cn, context_hint=context_hint), context
    elif task_type == "uncertainty":
        return PROMPT_UNCERTAINTY.format(context_hint=context_hint), context
    else:
        return PROMPT_DISEASE.format(label_cn=label_cn, context_hint=context_hint), context


def process_one_sample(
    idx: int,
    rec: dict,
    backend,
    consistency_runs: int,
) -> dict | None:
    """处理一条记录, 生成 benchmark 样本. 失败或不一致返回 None."""
    task_type = rec["_task_type"]
    label_cn = rec.get("label_cn", "")
    image_path = rec.get("image_path", "")

    prompt, context = _build_prompt(rec)

    # uncertainty 不需要自一致性 (本身就是不确定的)
    if task_type == "uncertainty" or consistency_runs <= 1:
        response = backend.generate(prompt, image_path)
    else:
        response = run_with_consistency(
            backend, prompt, image_path, label_cn, consistency_runs
        )

    if response is None:
        return None

    # 解析推理链
    reasoning_chain = parse_reasoning_chain(response)

    # 格式检查: 推理链至少3步
    if len(reasoning_chain) < 3:
        # 尝试更积极的分割
        reasoning_chain = [s.strip() for s in response.split("。") if len(s.strip()) > 10]
        if len(reasoning_chain) < 3:
            return None  # 质量不达标, 丢弃

    # 确定 ground_truth
    if task_type == "uncertainty":
        ground_truth = "不确定/信息不足"
    else:
        ground_truth = label_cn

    # 确定 question
    question = random.choice(QUESTION_TEMPLATES.get(task_type, QUESTION_TEMPLATES["disease"]))

    difficulty = estimate_difficulty(task_type, reasoning_chain, label_cn)

    sample = {
        "id": f"agrireason_{idx:04d}",
        "image_path": image_path,
        "context": context,
        "question": question,
        "ground_truth": ground_truth,
        "reasoning_chain": reasoning_chain,
        "difficulty": difficulty,
        "task_type": task_type,
        "source_dataset": rec.get("source", "unknown"),
    }

    # differential 额外字段
    if task_type == "differential":
        sample["confusable_disease"] = rec.get("_confusable", "")

    return sample


# ---------------------------------------------------------------------------
# 主生成循环
# ---------------------------------------------------------------------------
def generate_benchmark(
    records: list[dict],
    backend,
    output_file: str,
    consistency_runs: int,
    workers: int,
):
    """并发生成 benchmark 数据集."""
    print(f"\n开始生成 AgriReason Benchmark ({len(records)} 条待处理)")
    print(f"  一致性检验: {consistency_runs} 次/样本")
    print(f"  并发数: {workers}")

    # 本地模型强制单线程
    if isinstance(backend, LocalBackend):
        workers = 1

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    results: list[dict] = []
    skipped = 0

    pbar = tqdm(total=len(records), desc="生成 AgriReason", unit="sample")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_one_sample, i, rec, backend, consistency_runs
            ): i
            for i, rec in enumerate(records, start=1)
        }

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                skipped += 1
            pbar.update(1)

    pbar.close()

    # 重新编号 (按 task_type 排序后连续编号)
    results.sort(key=lambda x: (x["task_type"], x["id"]))
    for i, sample in enumerate(results, start=1):
        sample["id"] = f"agrireason_{i:04d}"

    # 输出
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 统计
    type_counts = Counter(s["task_type"] for s in results)
    diff_counts = Counter(s["difficulty"] for s in results)

    print(f"\n{'═' * 60}")
    print(f"AgriReason Benchmark 生成完成")
    print(f"{'═' * 60}")
    print(f"  总样本数: {len(results)}")
    print(f"  丢弃数:   {skipped} (不一致/格式不达标)")
    print(f"\n  ── 按任务类型 ──")
    for t in ["disease", "pest", "differential", "nutrient", "uncertainty"]:
        cnt = type_counts.get(t, 0)
        pct = cnt / len(results) * 100 if results else 0
        print(f"    {t:<15} {cnt:>5} ({pct:.1f}%)")
    print(f"\n  ── 按难度 ──")
    for d in ["easy", "medium", "hard"]:
        cnt = diff_counts.get(d, 0)
        pct = cnt / len(results) * 100 if results else 0
        print(f"    {d:<15} {cnt:>5} ({pct:.1f}%)")
    print(f"\n  输出: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AgriReason Benchmark 自动生成 — 首个农业多模态推理评测数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # DashScope API
  python generate_agrireason.py \\
      --input data/processed/unified_dataset.jsonl \\
      --output benchmark/data/agrireason_v1.json \\
      --api-key sk-xxx --max-samples 800

  # 本地模型
  python generate_agrireason.py \\
      --input data/processed/unified_dataset.jsonl \\
      --output benchmark/data/agrireason_v1.json \\
      --use-local --max-samples 500

  # 3次自一致性过滤 + 自定义模型
  python generate_agrireason.py \\
      --input data/processed/unified_dataset.jsonl \\
      --output benchmark/data/agrireason_v1.json \\
      --api-key sk-xxx --consistency-runs 3 \\
      --model qwen2.5-vl-72b-instruct
""",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/unified_dataset.jsonl",
        help="输入: unified_dataset.jsonl 路径 (default: data/processed/unified_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/data/agrireason_v1.json",
        help="输出 JSON 文件路径 (default: benchmark/data/agrireason_v1.json)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DashScope API key (也可通过 DASHSCOPE_API_KEY 环境变量设置)",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        default=False,
        help="使用本地 Qwen2.5-VL-7B 模型 (回退方案, 需要 GPU)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=800,
        help="最大生成样本数 (default: 800)",
    )
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=3,
        help="自一致性过滤: 每个样本生成 N 次, 仅保留诊断一致的 (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="并发 API 调用数 (default: 5, 本地模型自动设为 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-vl-72b-instruct",
        help="DashScope 模型名 (default: qwen2.5-vl-72b-instruct)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    args.workers = min(args.workers, 5)

    # API key
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not args.use_local and not api_key:
        print("错误: 请通过 --api-key 或 DASHSCOPE_API_KEY 环境变量提供 API key")
        print("      或使用 --use-local 回退到本地模型")
        sys.exit(1)

    # 验证输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        print("请先运行 prepare_data.py 生成 unified_dataset.jsonl")
        sys.exit(1)

    backend_cn = "本地 7B" if args.use_local else f"DashScope ({args.model})"

    print("=" * 60)
    print("AgriReason Benchmark 自动生成")
    print("=" * 60)
    print(f"  输入文件:      {args.input}")
    print(f"  输出文件:      {args.output}")
    print(f"  后端:          {backend_cn}")
    print(f"  最大样本数:    {args.max_samples}")
    print(f"  并发数:        {args.workers}")
    print(f"  一致性检验:    {args.consistency_runs} 次/样本")
    print()

    # 加载数据
    records = load_input_data(args.input)
    print(f"加载 {len(records)} 条输入记录")

    # 分层采样
    sampled = stratified_sample(records, args.max_samples, seed=args.seed)

    # 初始化后端
    if args.use_local:
        backend = LocalBackend()
    else:
        backend = DashScopeBackend(api_key=api_key, model=args.model)

    # 生成
    generate_benchmark(
        records=sampled,
        backend=backend,
        output_file=args.output,
        consistency_runs=args.consistency_runs,
        workers=args.workers,
    )

    print("\n✓ AgriReason Benchmark 生成完成")


if __name__ == "__main__":
    main()

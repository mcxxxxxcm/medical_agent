"""知识库-规则同步扫描器

v9.2 漏洞5修复：AC 自动机规则与知识库脱耦

问题：
    药物关键词表/同义词字典/症状关键词均为人工维护，
    与知识库文档无同步机制。新增文档后规则不更新 → 覆盖率下降。

方案：
    1. 知识库变更时自动扫描新实体（药物名、症状名）
    2. 与现有 AC 自动机规则对比，生成差异报告
    3. 差异报告写入文件，提醒人工更新规则
    4. 提供 API 接口手动触发扫描

扫描策略：
    - 从 ChromaDB 全量文档中提取潜在实体（基于 NER 规则）
    - 与 keyword_matcher.py 中的现有关键词做差集
    - 生成三类报告：missing_drugs（缺失药物）、missing_symptoms（缺失症状）、missing_synonyms（缺失同义词）

限制：
    - 不自动更新规则（医疗安全需要人工审核）
    - 仅扫描，不修改
"""
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.core.app_logging import get_logger
from app.core.config import get_config

logger = get_logger(__name__)


# ===== 实体提取规则（基于医学文本模式） =====

# 药物名模式：中文药名（2-6字）+ 常见后缀
_DRUG_SUFFIXES = {"片", "胶囊", "颗粒", "口服液", "注射液", "滴丸", "散剂", "膏", "贴", "栓"}
# 常见药物名正则（匹配"XX芬"、"XX韦"等药物通用名模式）
_DRUG_NAME_PATTERNS = [
    r"[\u4e00-\u9fff]{2,4}(?:芬|韦|唑|西林|匹林|昔布|替丁|普利|沙坦|地平|洛尔|松|龙|霉素|米星|西林|曲林|噻嗪)",
]
# 剂型正则
_DOSAGE_PATTERN = r"(\d+(?:\.\d+)?(?:mg|g|ml|片|粒|滴|支|瓶|袋|IU|U)[/／]?(?:\d+(?:\.\d+)?(?:mg|g|ml|片|粒|滴|支|瓶|袋|IU|U))?)"

# 症状名模式：常见症状描述词
_SYMPTOM_PATTERNS = [
    r"[\u4e00-\u9fff]{1,3}(?:痛|疼|痒|肿|麻|晕|烧|咳|泻|吐|闷|悸|疹|挛)",
    r"(?:持续|反复|间歇|阵发|慢性|急性)[\u4e00-\u9fff]{1,4}(?:痛|疼|痒|肿|麻|晕|烧|咳|泻|吐|闷|悸|疹|挛)",
]


class KBRuleSyncScanner:
    """知识库-规则同步扫描器"""

    def __init__(self):
        self._config = get_config()
        self._report_dir = self._config.DATA_DIR / "kb_rule_sync"
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def scan(self) -> Dict:
        """执行全量扫描，生成差异报告

        Returns:
            {
                "timestamp": "...",
                "kb_doc_count": 123,
                "existing_drugs": [...],
                "existing_symptoms": [...],
                "missing_drugs": [...],
                "missing_symptoms": [...],
                "potential_synonyms": {...},
                "report_path": "...",
            }
        """
        logger.info("知识库-规则同步扫描开始...")

        # 1. 加载知识库文档
        documents = self._load_kb_documents()
        if not documents:
            logger.warning("知识库文档为空，跳过扫描")
            return {"error": "知识库文档为空"}

        all_text = "\n".join(doc.page_content for doc in documents)

        # 2. 提取知识库中的实体
        kb_drugs = self._extract_drug_names(all_text)
        kb_symptoms = self._extract_symptom_names(all_text)

        # 3. 获取现有规则中的关键词
        existing_drugs = self._get_existing_drugs()
        existing_symptoms = self._get_existing_symptoms()

        # 4. 计算差集
        missing_drugs = sorted(kb_drugs - existing_drugs)
        missing_symptoms = sorted(kb_symptoms - existing_symptoms)

        # 5. 同义词发现
        potential_synonyms = self._discover_synonyms(all_text, existing_symptoms)

        # 6. 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "kb_doc_count": len(documents),
            "existing_drugs_count": len(existing_drugs),
            "existing_symptoms_count": len(existing_symptoms),
            "kb_drugs_count": len(kb_drugs),
            "kb_symptoms_count": len(kb_symptoms),
            "missing_drugs": missing_drugs[:100],  # 上限100
            "missing_symptoms": missing_symptoms[:100],
            "potential_synonyms": potential_synonyms,
            "coverage_rate": {
                "drugs": f"{(1 - len(missing_drugs) / max(len(kb_drugs), 1)) * 100:.1f}%",
                "symptoms": f"{(1 - len(missing_symptoms) / max(len(kb_symptoms), 1)) * 100:.1f}%",
            },
        }

        # 7. 持久化报告
        report_path = self._save_report(report)
        report["report_path"] = str(report_path)

        logger.info(
            f"扫描完成：知识库 {len(documents)} 篇文档，"
            f"缺失药物 {len(missing_drugs)} 个，缺失症状 {len(missing_symptoms)} 个，"
            f"药物覆盖率 {report['coverage_rate']['drugs']}，"
            f"症状覆盖率 {report['coverage_rate']['symptoms']}"
        )

        return report

    def get_latest_report(self) -> Optional[Dict]:
        """获取最新的扫描报告"""
        reports = sorted(self._report_dir.glob("sync_report_*.json"), reverse=True)
        if not reports:
            return None
        try:
            return json.loads(reports[0].read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_kb_documents(self) -> list:
        """从向量库加载全部文档"""
        try:
            from app.rag.vector_store import get_vector_store_manager
            manager = get_vector_store_manager()
            if manager.vector_store is None:
                logger.warning("向量库未初始化，无法扫描")
                return []
            return manager.load_all_documents()
        except Exception as e:
            logger.error(f"加载知识库文档失败：{e}")
            return []

    def _extract_drug_names(self, text: str) -> Set[str]:
        """从文本中提取药物名称"""
        drugs = set()

        # 1. 基于药物名正则模式
        for pattern in _DRUG_NAME_PATTERNS:
            matches = re.findall(pattern, text)
            drugs.update(matches)

        # 2. 基于剂型后缀
        for suffix in _DRUG_SUFFIXES:
            pattern = rf"([\u4e00-\u9fff]{{2,6}}{suffix})"
            matches = re.findall(pattern, text)
            drugs.update(matches)

        # 3. 过滤噪声（太短的、包含非药物词的）
        noise_words = {"一片", "两片", "三片", "什么片", "这个片", "那个片", "这种片", "那种片"}
        drugs = drugs - noise_words

        return drugs

    def _extract_symptom_names(self, text: str) -> Set[str]:
        """从文本中提取症状名称"""
        symptoms = set()

        for pattern in _SYMPTOM_PATTERNS:
            matches = re.findall(pattern, text)
            symptoms.update(matches)

        # 过滤噪声
        noise_words = {"不痛", "无痛", "没痛", "减轻痛", "缓解痛"}
        symptoms = symptoms - noise_words

        return symptoms

    def _get_existing_drugs(self) -> Set[str]:
        """获取现有规则中的药物关键词"""
        try:
            from app.core.keyword_matcher import get_drug_matcher
            matcher = get_drug_matcher()
            return set(matcher._keywords)
        except Exception:
            return set()

    def _get_existing_symptoms(self) -> Set[str]:
        """获取现有规则中的症状关键词"""
        try:
            from app.core.keyword_matcher import get_symptom_matcher
            matcher = get_symptom_matcher()
            # 返回标准名（mapping 的 values）
            return set(matcher._mapping.values())
        except Exception:
            return set()

    def _discover_synonyms(self, text: str, existing_symptoms: Set[str]) -> Dict[str, List[str]]:
        """发现潜在同义词（与现有症状共现的相似词）

        策略：在现有症状前后 5 字符范围内，提取候选同义词
        """
        synonyms = {}
        for symptom in existing_symptoms:
            # 查找症状词在文本中的位置，提取上下文中的候选同义词
            pattern = rf"([\u4e00-\u9fff]{{1,3}}(?:或|、|和|与|以及|亦称|又称|也叫|又名)[\u4e00-\u9fff]{{1,3}}{re.escape(symptom)})"
            matches = re.findall(pattern, text)
            if matches:
                # 提取"或/和/又称"前面的候选同义词
                candidates = []
                for match in matches[:5]:  # 每个症状最多5个候选
                    parts = re.split(r"(?:或|、|和|与|以及|亦称|又称|也叫|又名)", match)
                    for part in parts:
                        part = part.strip()
                        if part and part != symptom and len(part) >= 2:
                            candidates.append(part)
                if candidates:
                    synonyms[symptom] = candidates
        return synonyms

    def _save_report(self, report: Dict) -> Path:
        """保存扫描报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sync_report_{timestamp}.json"
        path = self._report_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 只保留最近 10 份报告
        reports = sorted(self._report_dir.glob("sync_report_*.json"))
        for old_report in reports[:-10]:
            old_report.unlink()

        logger.info(f"扫描报告已保存：{path}")
        return path


# ===== 全局单例 =====
_scanner: Optional[KBRuleSyncScanner] = None


def get_kb_rule_sync_scanner() -> KBRuleSyncScanner:
    """获取扫描器单例"""
    global _scanner
    if _scanner is None:
        _scanner = KBRuleSyncScanner()
    return _scanner


def scan_kb_rule_sync() -> Dict:
    """执行知识库-规则同步扫描（便捷函数）"""
    scanner = get_kb_rule_sync_scanner()
    return scanner.scan()

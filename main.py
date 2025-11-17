import hashlib
import json
import uuid
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time
import re
import requests  # <--- 新增导入
import os  # <-- 新增: 用于处理文件路径
import docx  # <-- 新增: 用于读取 .docx
import fitz  # <-- 新增: (PyMuPDF) 用于读取 .pdf
# --- 数据结构 (Data Structures) ---

@dataclass
class DocumentChunk:
    """ 对应文档中的语义块 """
    id: str
    content: str
    chunk_type: str  # "paragraph", "title", "table", "formula"
    position: int
    hash: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class KnowledgeTriple:
    """ 知识三元组 (Subject, Predicate, Object) """
    subject: str
    predicate: str
    object: str
    source_chunk_id: str  # 确保可追溯性
    confidence: float
    timestamp: float
    is_active: bool = True  # 用于标记知识是否在冲突中被覆盖


# --- 模拟 LLM 客户端 (Mock LLM Client) ---
# (保留 MockLLMClient 用于模拟)
class MockLLMClient:
    """ 模拟 LLM API 调用 (用于测试和模拟) """

    def generate(self, prompt: str) -> str:
        """ 模拟 LLM 生成响应。"""
        prompt = prompt.lower()

        # 1. 模拟三元组提取 (RATE)
        if "提取实体关系三元组" in prompt:
            if "公司a生产设备x" in prompt:
                return '[{"subject": "设备X", "predicate": "manufactured_by", "object": "公司A"}, {"subject": "设备X", "predicate": "located_in", "object": "工厂Y"}]'
            if "公司b也生产设备x" in prompt:
                # 注意：这里故意制造了一个拼写错误 'manufacted_by' 来测试实体解析
                return '[{"subject": "设备X", "predicate": "manufacted_by", "object": "公司B"}, {"subject": "设备X", "predicate": "located_in", "object": "工厂Z"}]'
            if "设备a由公司c制造" in prompt:
                return '[{"subject": "设备A", "predicate": "manufactured_by", "object": "公司C"}, {"subject": "设备A", "predicate": "used_for", "object": "石油开采"}]'
            return '[]'

        # 2. 模拟指代消解
        elif "解析以下文本中的指代" in prompt:
            if "它位于" in prompt or "它自己" in prompt:
                return '{"设备X": ["它", "它自己"]}'
            return '{}'

        # 3. 模拟变更语义分类
        elif "请对以下变更进行语义分类" in prompt:
            if "公司b也生产设备x" in prompt:
                return '{"change_type": "modify_entity_attribute", "reason": "修改了制造商属性"}'
            else:
                return '{"change_type": "new_entity", "reason": "引入了新信息"}'

        # 4. 模拟 LLM 引导的实体解析
        elif "请判断以下两个实体是否等效" in prompt:
            if "manufacted_by" in prompt and "manufactured_by" in prompt:
                return '{"equivalent": true, "canonical_form": "manufactured_by", "reason": "拼写错误，语义相同"}'
            return '{"equivalent": false, "reason": "实体代表不同概念"}'

        # 5. 模拟多智能体辩论
        elif "作为支持新事实的智能体" in prompt:  # Pro-New-Fact Agent
            return '{"preferred_triple_index": 0, "reasoning": "新文档通常包含最新的信息，旧信息可能已过时。", "confidence": 0.8}'

        elif "作为支持现有事实的智能体" in prompt:  # Pro-Existing-Fact Agent
            return '{"preferred_triple_index": 1, "reasoning": "现有事实已经过验证，新事实未经证实，可能是错误的。", "confidence": 0.7}'

        elif "作为中立裁决智能体" in prompt:  # Neutral Adjudication Agent
            return '{"preferred_triple_index": 0, "reasoning": "分析所有证据后，新事实 (index 0) 的来源更权威，时间戳更新。", "confidence": 0.95}'

        else:
            return "{}"


# --- (新增) 真实的 DeepSeek API 客户端 ---
class DeepSeekClient:
    """
    实现对 DeepSeek API 的真实调用
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key must be provided")
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        print("    [DeepSeekClient] 客户端已初始化。")

    def generate(self, prompt: str) -> str:
        """
        使用 DeepSeek API 生成响应
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建 OpenAI 格式的 payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant. You must respond ONLY with the requested JSON object or text, with no preamble or explanation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # 对于结构化输出，保持低温
            "max_tokens": 1024
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=30)

            # 检查 HTTP 错误
            response.raise_for_status()

            response_data = response.json()

            # 提取内容
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                # 清理可能的 markdown 标记
                content_cleaned = self.cleanup_json_response(content)
                return content_cleaned
            else:
                print(f"    [DeepSeekClient] API 响应中未找到 'choices' 或 'choices' 为空: {response_data}")
                return "{}"  # 返回空JSON作为后备

        except requests.exceptions.HTTPError as http_err:
            print(f"    [DeepSeekClient] HTTP 错误: {http_err} - {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"    [DeepSeekClient] 请求错误: {req_err}")
        except json.JSONDecodeError:
            print(f"    [DeepSeekClient] 无法解析 API 响应: {response.text}")
        except Exception as e:
            print(f"    [DeepSeekClient] 未知错误: {e}")

        return "{}"  # 任何错误都返回空JSON

    def cleanup_json_response(self, text: str) -> str:
        """ 清理 LLM 响应，提取 JSON。"""
        # 寻找 ```json ... ```
        match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()

        # 寻找第一个 '{' 或 '['
        start_brace = text.find('{')
        start_bracket = text.find('[')

        if start_brace == -1 and start_bracket == -1:
            return text  # 未找到 JSON，按原样返回

        # 确定 JSON 的起始位置
        if start_brace == -1:
            start_index = start_bracket
            end_char = ']'
        elif start_bracket == -1:
            start_index = start_brace
            end_char = '}'
        else:
            start_index = min(start_brace, start_bracket)
            end_char = ']' if start_index == start_bracket else '}'

        # 寻找最后一个匹配的结束符
        end_index = text.rfind(end_char)

        if end_index > start_index:
            return text[start_index:end_index + 1]

        return text.strip()  # 后备


# --- 模块 1: 文档预处理 (Document Preprocessing) ---
# (此部分无变化)
class DocumentPreprocessor:
    """ 负责文档解析、分块和上下文综合 """
    def __init__(self, llm_client):
        self.llm = llm_client

    def layout_analysis(self, document_text: str) -> List[DocumentChunk]:
        chunks = []
        # 更智能的段落分割
        paragraphs = re.split(r'\n\s*\n', document_text.strip())

        for i, para in enumerate(paragraphs):
            if para.strip():
                chunk_id = f"chunk_{i}_{uuid.uuid4().hex[:8]}"
                content_hash = hashlib.sha256(para.encode()).hexdigest()
                chunk_type = self._classify_chunk_type(para)

                chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=para,
                    chunk_type=chunk_type,
                    position=i,
                    hash=content_hash
                ))
        return chunks

    def coreference_resolution(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        修复的指代消解方法
        """
        resolved_chunks = []
        full_text = " ".join([c.content for c in chunks])

        # 调用 LLM 进行指代消解
        prompt = f"""
        请解析以下文本中的指代关系，识别代词（如它、他们、这个等）所指代的实体。
        返回 JSON 格式，键为实体名，值为该实体对应的代词列表。

        文本:
        {full_text}

        示例输出: {{"设备X": ["它", "该设备"], "工厂Y": ["那里"]}}
        """

        try:
            response = self.llm.generate(prompt)
            print(f"    [Preprocessor] LLM 指代消解响应: {response}")

            # 安全地解析 JSON 响应
            if response.strip():
                coref_map = json.loads(response)
            else:
                coref_map = {}

            # 确保 coref_map 是字典类型
            if not isinstance(coref_map, dict):
                print(f"    [Preprocessor] 警告: 指代消解响应不是字典格式: {coref_map}")
                coref_map = {}

        except json.JSONDecodeError as e:
            print(f"    [Preprocessor] JSON 解析错误: {e}, 响应: {response}")
            coref_map = {}
        except Exception as e:
            print(f"    [Preprocessor] 指代消解错误: {e}")
            coref_map = {}

        # 应用指代消解
        for chunk in chunks:
            resolved_content = chunk.content

            if coref_map:
                for entity, mentions in coref_map.items():
                    if not isinstance(mentions, list):
                        continue
                    for mention in mentions:
                        if isinstance(mention, str) and mention:
                            # 使用单词边界进行精确替换
                            pattern = r'\b' + re.escape(mention) + r'\b'
                            resolved_content = re.sub(pattern, entity, resolved_content)

            # 如果内容被修改，重新计算哈希值
            if resolved_content != chunk.content:
                new_hash = hashlib.sha256(resolved_content.encode()).hexdigest()
            else:
                new_hash = chunk.hash

            resolved_chunks.append(DocumentChunk(
                id=chunk.id,
                content=resolved_content,
                chunk_type=chunk.chunk_type,
                position=chunk.position,
                hash=new_hash
            ))

        print(f"    [Preprocessor] 指代消解完成，处理了 {len(chunks)} 个块。")
        return resolved_chunks

    def _classify_chunk_type(self, text: str) -> str:
        """ 简单的块类型分类 """
        text_len = len(text)
        if text_len < 100 and (text.endswith(':') or not '.' in text):
            return "title"
        if '|' in text or '\t' in text:
            return "table"
        return "paragraph"


# --- 模块 2: 知识提取 (Knowledge Extraction) ---
# (此部分无变化)
class KnowledgeExtractor:
    """ 负责执行 RAG 增强的三元组提取 (RATE) """
    def __init__(self, llm_client, ontology_schema):
        self.llm = llm_client
        self.ontology = ontology_schema  #

    def extract_triples(self, chunks: List[DocumentChunk]) -> List[KnowledgeTriple]:
        """ 从文档块中提取知识三元组 """
        all_triples = []

        for chunk in chunks:
            # 仅处理有意义的文本块
            if chunk.chunk_type in ["paragraph", "table"]:
                triples = self._extract_from_chunk(chunk)
                validated_triples = self._pre_validate_triples(triples)  #
                all_triples.extend(validated_triples)

        print(f"    [Extractor] 提取并验证了 {len(all_triples)} 个三元组。")
        return all_triples

    def _extract_from_chunk(self, chunk: DocumentChunk) -> List[KnowledgeTriple]:
        """ (RATE) 从单个块中提取三元组 """
        valid_predicates_str = ", ".join(self.ontology.get("predicates", []))
        prompt = f"""
        基于以下本体谓词: [{valid_predicates_str}]
        请从以下文本中提取实体关系三元组，格式为JSON:
        文本:
        {chunk.content}
        输出格式: [{{"subject": "...", "predicate": "...", "object": "..."}}]
        """
        try:
            response = self.llm.generate(prompt)
            print(f"    [Extractor] LLM 提取响应: {response}")

            # 安全地解析响应
            if response.strip():
                triples_data = json.loads(response)
            else:
                triples_data = []

            if not isinstance(triples_data, list):
                print(f"    [Extractor] 警告: 响应不是列表格式: {triples_data}")
                triples_data = []

            triples = []
            for triple_data in triples_data:
                if isinstance(triple_data, dict) and all(k in triple_data for k in ["subject", "predicate", "object"]):
                    triple = KnowledgeTriple(
                        subject=str(triple_data["subject"]).strip(),
                        predicate=str(triple_data["predicate"]).strip(),
                        object=str(triple_data["object"]).strip(),
                        source_chunk_id=chunk.id,
                        confidence=0.9,
                        timestamp=time.time()
                    )
                    triples.append(triple)

            return triples
        except json.JSONDecodeError as e:
            print(f"    [Extractor] JSON 解析错误: {e}, 响应: {response}")
            return []
        except Exception as e:
            print(f"    [Extractor] 提取错误: {e}")
            return []

    def _pre_validate_triples(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        valid_triples = []
        valid_predicates = set(self.ontology.get("predicates", []))

        for triple in triples:
            if triple.predicate not in valid_predicates:
                print(f"    [Validator] 丢弃无效谓词: {triple.predicate}")
                continue
            if not triple.subject or not triple.object:
                print(f"    [Validator] 丢弃空主体或客体")
                continue
            valid_triples.append(triple)

        return valid_triples

# --- 模块 3: 增量融合引擎 (Incremental Fusion Engine) ---
# (此部分无变化)
class IncrementalFusionEngine:
    """
    管理变更检测、目标性知识提取和知识融合
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        # 实体解析映射 (用于去重)
        self.entity_resolution_map = {}

    def detect_changes(self, old_chunks: List[DocumentChunk],
                       new_chunks: List[DocumentChunk]) -> Dict[str, List[DocumentChunk]]:
        """
        使用哈希值比较检测文档块级别的变化
        """
        old_hashes = {chunk.hash: chunk for chunk in old_chunks}
        new_hashes = {chunk.hash: chunk for chunk in new_chunks}

        # (改进变更检测逻辑)
        old_id_map = {chunk.id: chunk for chunk in old_chunks}
        new_id_map = {chunk.id: chunk for chunk in new_chunks}

        old_ids = set(old_id_map.keys())
        new_ids = set(new_id_map.keys())

        added_chunks = [new_id_map[id] for id in new_ids - old_ids]
        deleted_chunk_ids = list(old_ids - new_ids)

        potential_modified_ids = old_ids.intersection(new_ids)
        modified_chunks = []

        for id in potential_modified_ids:
            if old_id_map[id].hash != new_id_map[id].hash:
                modified_chunks.append(new_id_map[id])

        # 对变更进行语义分类
        classified_changes = {
            "added": self._classify_changes(added_chunks),
            "modified": self._classify_changes(modified_chunks),
            "deleted_ids": deleted_chunk_ids
        }

        print(
            f"    [ChangeDetect] 检测到: {len(classified_changes['added'])} 新增, {len(classified_changes['modified'])} 修改, {len(classified_changes['deleted_ids'])} 删除。")
        return classified_changes

    def _classify_changes(self, chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, str]]:
        """
        使用 LLM 对变更块进行语义分类
        """
        classified = []
        for chunk in chunks:
            prompt = f"请对以下变更进行语义分类 (例如: 'new_entity', 'modify_entity_attribute', 'change_relationship'):\n{chunk.content}"
            try:
                response = json.loads(self.llm.generate(prompt))
                change_type = response.get("change_type", "unknown")
                classified.append((chunk, change_type))
                print(f"    [ChangeDetect] 块 {chunk.id[:5]}... 被分类为: {change_type}")
            except Exception:
                classified.append((chunk, "unknown"))
        return classified

    def entity_resolution(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """
        使用 LLM 引导的聚类/判断来执行实体解析
        """
        resolved_triples = []
        all_entities = set()
        for t in triples:
            all_entities.add(t.subject)
            all_entities.add(t.object)
            all_entities.add(t.predicate)

        # 更新内部解析映射
        entities_list = sorted(list(all_entities))
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                e1, e2 = entities_list[i], entities_list[j]

                # 避免重复检查
                if e1 in self.entity_resolution_map and self.entity_resolution_map[e1] == e2:
                    continue
                if e2 in self.entity_resolution_map and self.entity_resolution_map[e2] == e1:
                    continue

                # 仅在实体看起来相似时才调用 LLM (节省 API 调用)
                if self._are_entities_similar(e1, e2):
                    prompt = f"请判断以下两个实体是否等效 (例如，拼写错误、缩写): \n实体1: '{e1}'\n实体2: '{e2}'"
                    try:
                        response = json.loads(self.llm.generate(prompt))
                        if response.get("equivalent"):
                            canonical = response.get("canonical_form", e1)  # 选择一个标准形式
                            self.entity_resolution_map[e1] = canonical
                            self.entity_resolution_map[e2] = canonical
                            print(f"    [EntityResolve] LLM 判定 '{e1}' 和 '{e2}' 等效 -> '{canonical}'")
                    except Exception:
                        pass

        # 应用解析映射
        for t in triples:
            resolved_triples.append(KnowledgeTriple(
                subject=self.entity_resolution_map.get(t.subject, t.subject),
                predicate=self.entity_resolution_map.get(t.predicate, t.predicate),
                object=self.entity_resolution_map.get(t.object, t.object),
                source_chunk_id=t.source_chunk_id,
                confidence=t.confidence,
                timestamp=t.timestamp
            ))

        return resolved_triples

    def _are_entities_similar(self, e1: str, e2: str) -> bool:
        """ 简单的相似度检查，用于决定是否调用 LLM """
        # (这里可以使用更复杂的
        # Levenshtein 距离等)
        e1_norm = e1.lower().replace("_", "")
        e2_norm = e2.lower().replace("_", "")

        if e1_norm == e2_norm:
            return True
        if abs(len(e1_norm) - len(e2_norm)) > 5:  # 长度差异过大
            return False

        return True  # 默认进行更宽松的检查


# --- 模块 4: 冲突解决 (Conflict Resolution) ---
# (此部分无变化)
class MultiAgentConflictResolver:
    """
    多智能体冲突解决机制 (CRM)
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        # 实例化智能体
        self.agents = [
            ProNewFactAgent(llm_client),
            ProExistingFactAgent(llm_client),
            NeutralAdjudicationAgent(llm_client)
        ]

    def resolve_conflict(self, new_triple: KnowledgeTriple,
                         conflicting_triple: KnowledgeTriple,
                         evidence_chunks: List[DocumentChunk]) -> Tuple[KnowledgeTriple, KnowledgeTriple]:
        """
        执行多智能体辩论和投票
        """
        print(f"    [CRM] 冲突检测到!")
        print(f"      - 新事实 (0): ({new_triple.subject}, {new_triple.predicate}, {new_triple.object})")
        print(
            f"      - 现有事实 (1): ({conflicting_triple.subject}, {conflicting_triple.predicate}, {conflicting_triple.object})")

        # 1. RAG 证据增强
        evidence_text = "\n---\n".join(
            [f"来源 {c.id[:5]} (类型: {c.chunk_type}):\n{c.content}" for c in evidence_chunks])

        # 2. 多智能体辩论
        arguments = []
        triples_in_conflict = [new_triple, conflicting_triple]

        for agent in self.agents:
            argument_json = agent.debate(triples_in_conflict, evidence_text)
            try:
                argument = json.loads(argument_json)
                arguments.append(argument)
                print(
                    f"      - {agent.__class__.__name__} 论点: {argument['reasoning']} (选择: {argument['preferred_triple_index']}, 置信度: {argument['confidence']})")
            except Exception:
                print(f"      - {agent.__class__.__name__} 论点解析失败。")

        # 3. 投票和真值推断
        # (使用置信度加权投票)
        votes = defaultdict(float)
        for arg in arguments:
            idx = arg.get("preferred_triple_index")
            conf = arg.get("confidence", 0.5)
            if idx in [0, 1]:
                votes[idx] += conf

        if not votes:
            # 默认保留最新的 (Last-Write-Wins)
            winner_idx = 0
            print("    [CRM] 投票失败，默认选择新事实。")
        else:
            winner_idx = max(votes.items(), key=lambda x: x[1])[0]
            print(f"    [CRM] 投票结果: {dict(votes)}")

        # 4. 决策
        if winner_idx == 0:
            print(f"    [CRM] 决策: 接受新事实，覆盖旧事实。")
            conflicting_triple.is_active = False  # 禁用旧事实
            new_triple.is_active = True
            return new_triple, conflicting_triple  # 返回获胜者和失败者
        else:
            print(f"    [CRM] 决策: 保留旧事实，拒绝新事实。")
            new_triple.is_active = False  # 禁用新事实
            conflicting_triple.is_active = True
            return conflicting_triple, new_triple  # 返回获胜者和失败者


class ProNewFactAgent:
    """ 支持新事实的智能体 """

    def __init__(self, llm_client): self.llm = llm_client

    def debate(self, triples: List[KnowledgeTriple], evidence: str) -> str:
        prompt = f"""
        作为支持新事实的智能体，请分析以下证据。新事实 (index 0) 通常是更新的。
        证据: {evidence}
        候选三元组:
        0: {(triples[0].subject, triples[0].predicate, triples[0].object)} (来源: {triples[0].source_chunk_id[:5]})
        1: {(triples[1].subject, triples[1].predicate, triples[1].object)} (来源: {triples[1].source_chunk_id[:5]})
        请输出JSON格式: {{"preferred_triple_index": index, "reasoning": "...", "confidence": 0.8}}
        """
        return self.llm.generate(prompt)


class ProExistingFactAgent:
    """ (已实现) 支持现有事实的智能体 """

    def __init__(self, llm_client): self.llm = llm_client

    def debate(self, triples: List[KnowledgeTriple], evidence: str) -> str:
        prompt = f"""
        作为支持现有事实的智能体，请分析以下证据。现有事实 (index 1) 是图谱中已验证的。
        证据: {evidence}
        候选三元组:
        0: {(triples[0].subject, triples[0].predicate, triples[0].object)} (来源: {triples[0].source_chunk_id[:5]})
        1: {(triples[1].subject, triples[1].predicate, triples[1].object)} (来源: {triples[1].source_chunk_id[:5]})
        请输出JSON格式: {{"preferred_triple_index": index, "reasoning": "...", "confidence": 0.7}}
        """
        return self.llm.generate(prompt)


class NeutralAdjudicationAgent:
    """ (已实现) 中立裁决智能体 """

    def __init__(self, llm_client): self.llm = llm_client

    def debate(self, triples: List[KnowledgeTriple], evidence: str) -> str:
        prompt = f"""
        作为中立裁决智能体，请仅基于证据和CoT推理，客观地评估哪个事实更可信。
        证据: {evidence}
        候选三元组:
        0: {(triples[0].subject, triples[0].predicate, triples[0].object)} (来源: {triples[0].source_chunk_id[:5]})
        1: {(triples[1].subject, triples[1].predicate, triples[1].object)} (来源: {triples[1].source_chunk_id[:5]})
        请输出JSON格式: {{"preferred_triple_index": index, "reasoning": "...", "confidence": 0.95}}
        """
        return self.llm.generate(prompt)


# --- 评估器 (Evaluator) ---
# (此部分无变化)
class IDGraphRAGEvaluator:
    """
    实现三维评估体系
    A: KGC 质量, B: 融合一致性, C: 增量效率
    """

    def __init__(self):
        self.metrics = {}

    def evaluate_kg_quality(self, extracted_triples: List[KnowledgeTriple],
                            ground_truth: List[KnowledgeTriple]) -> Dict[str, float]:
        """ 评估 KGC 质量 (维度 A) """
        predicted_set = {(t.subject, t.predicate, t.object) for t in extracted_triples if t.is_active}
        actual_set = {(t.subject, t.predicate, t.object) for t in ground_truth}

        tp = len(predicted_set.intersection(actual_set))
        fp = len(predicted_set - actual_set)
        fn = len(actual_set - predicted_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        hallucination_rate = fp / (tp + fp) if (tp + fp) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,  #
            "hallucination_rate": hallucination_rate  #
        }

    def evaluate_conflict_resolution(self, resolved_conflicts: List[Dict],
                                     ground_truth_conflicts: List[Dict]) -> Dict[str, float]:
        """ 评估冲突解决 F1 分数 (维度 B) """
        # (简化实现)
        if not ground_truth_conflicts:
            return {"conflict_resolution_f1": 0.0}

        correct_resolutions = 0
        for res, gt in zip(resolved_conflicts, ground_truth_conflicts):
            if res.get("winner_triple_id") == gt.get("expected_winner_id"):
                correct_resolutions += 1

        # 假设 P=R=A (简化)
        accuracy = correct_resolutions / len(ground_truth_conflicts)
        return {"conflict_resolution_f1": accuracy}  #

    def evaluate_efficiency(self, full_rebuild_time: float,
                            incremental_update_time: float,
                            full_rebuild_storage: float,
                            incremental_storage: float) -> Dict[str, float]:
        """ 评估增量效率 (维度 C) """

        # CPU 时间缩减倍数
        cpu_time_reduction = full_rebuild_time / incremental_update_time if incremental_update_time > 0 else float(
            'inf')

        # 存储开销缩减
        storage_reduction = full_rebuild_storage / incremental_storage if incremental_storage > 0 else float('inf')

        return {
            "cpu_time_reduction_multiple": cpu_time_reduction,
            "storage_overhead_reduction": storage_reduction,
            "avg_update_latency": incremental_update_time  #
        }


# --- 主系统 (Main System) ---
# (此部分无变化)
class IDGraphRAGSystem:
    """
    ID-GraphRAG 框架的闭环增量构建系统
    """

    def __init__(self, llm_client, ontology_schema):
        self.preprocessor = DocumentPreprocessor(llm_client)
        self.extractor = KnowledgeExtractor(llm_client, ontology_schema)
        self.fusion_engine = IncrementalFusionEngine(llm_client)
        self.conflict_resolver = MultiAgentConflictResolver(llm_client)
        self.evaluator = IDGraphRAGEvaluator()

        self.knowledge_graph: Dict[str, KnowledgeTriple] = {}  # 使用 Dict 便于查找
        self.document_chunks: Dict[str, DocumentChunk] = {}  # 存储当前版本的文档块
        self.chunk_id_to_triple_ids: Dict[str, set] = defaultdict(set)

        self.resolved_conflicts_log = []  # 用于评估CRM

    def process_document(self, document_text: str, document_id: str, is_update: bool = False) -> Dict[str, Any]:
        """
        处理文档（初始构建或增量更新）
        """
        start_time = time.time()

        # 1. 文档预处理与上下文综合
        chunks_list = self.preprocessor.layout_analysis(document_text)
        resolved_chunks_list = self.preprocessor.coreference_resolution(chunks_list)
        new_chunks_map = {c.id: c for c in resolved_chunks_list}

        target_chunks_for_extraction = []

        if is_update:
            # 2. 变更检测
            print("  [System] 执行增量更新...")
            old_chunks_list = list(self.document_chunks.values())
            changes = self.fusion_engine.detect_changes(old_chunks_list, resolved_chunks_list)

            # 3. 目标性知识提取 (Targeted KGC)
            # 仅对新增和修改的块执行提取
            target_chunks_for_extraction = [c[0] for c in changes["added"]] + [c[0] for c in changes["modified"]]

            # 处理删除
            deleted_triple_ids = set()
            for chunk_id in changes["deleted_ids"]:
                if chunk_id in self.chunk_id_to_triple_ids:
                    deleted_triple_ids.update(self.chunk_id_to_triple_ids[chunk_id])
                    del self.chunk_id_to_triple_ids[chunk_id]

            for triple_id in deleted_triple_ids:
                if triple_id in self.knowledge_graph:
                    # 改为标记为 'is_active = False' 而不是硬删除
                    self.knowledge_graph[triple_id].is_active = False

            print(f"  [System] (Targeted KGC) 仅处理 {len(target_chunks_for_extraction)} 个变更块。")

        else:
            # 2. 初始构建
            print("  [System] 执行初始构建...")
            target_chunks_for_extraction = resolved_chunks_list

        # 3. 知识提取
        new_triples = self.extractor.extract_triples(target_chunks_for_extraction)

        # 4. 实例级知识融合与实体解析
        resolved_triples = self.fusion_engine.entity_resolution(new_triples)

        # 5. 融合与冲突解决
        fused_count = 0
        for triple in resolved_triples:
            fused_count += 1
            self._fuse_knowledge(triple)

        # 更新系统状态
        self.document_chunks = new_chunks_map

        processing_time = time.time() - start_time
        return {
            "triples_extracted": len(new_triples),
            "triples_fused": fused_count,
            "knowledge_graph_size": len([t for t in self.knowledge_graph.values() if t.is_active]),
            "processing_time": processing_time
        }

    def _fuse_knowledge(self, new_triple: KnowledgeTriple):
        """
        融合单个三元组，在需要时触发 CRM
        """
        # (S, P) 作为冲突检测的键
        conflict_key = (new_triple.subject, new_triple.predicate)

        # 生成唯一的三元组ID (基于来源和内容)
        triple_id = hashlib.sha256(
            f"{new_triple.source_chunk_id}{new_triple.subject}{new_triple.predicate}{new_triple.object}".encode()).hexdigest()
        new_triple.id = triple_id  # (给三元组也加上ID)

        existing_triple = None
        for t in self.knowledge_graph.values():
            if t.is_active and (t.subject, t.predicate) == conflict_key:
                existing_triple = t
                break

        if existing_triple and existing_triple.object != new_triple.object:
            # 5.1 触发多智能体冲突解决 (CRM)
            evidence_chunks = self._gather_evidence(new_triple, [existing_triple])

            winner, loser = self.conflict_resolver.resolve_conflict(
                new_triple, existing_triple, evidence_chunks
            )

            # 更新图谱
            # (确保两个三元组都在图谱中，但只有一个是 'active')
            self.knowledge_graph[new_triple.id] = new_triple
            self.knowledge_graph[existing_triple.id] = existing_triple

            self.chunk_id_to_triple_ids[new_triple.source_chunk_id].add(new_triple.id)

            # 记录用于评估
            self.resolved_conflicts_log.append({
                "winner_triple_id": winner.id,
                "loser_triple_id": loser.id
            })

        elif not existing_triple:
            # 5.2 无冲突，直接添加
            self.knowledge_graph[triple_id] = new_triple
            self.chunk_id_to_triple_ids[new_triple.source_chunk_id].add(triple_id)

        # 5.3 如果 (S,P,O) 完全相同，则忽略 (隐式处理)

    def _gather_evidence(self, triple: KnowledgeTriple,
                         conflicts: List[KnowledgeTriple]) -> List[DocumentChunk]:
        """
        为 CRM 收集 RAG 证据
        """
        evidence_chunk_ids = {triple.source_chunk_id}
        for conflict in conflicts:
            evidence_chunk_ids.add(conflict.source_chunk_id)

        # 检索源文档块
        evidence_chunks = [chunk for chunk_id, chunk in self.document_chunks.items()
                           if chunk_id in evidence_chunk_ids]

        return evidence_chunks

    def get_active_graph(self) -> List[KnowledgeTriple]:
        return [t for t in self.knowledge_graph.values() if t.is_active]


def load_document_text(file_path: str) -> str:
    """
    根据文件扩展名加载文档内容。
    支持 .txt, .docx, .pdf
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"  [Loader] 错误: 文件未找到 {file_path}")
        return ""

    # 获取文件扩展名
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    content = ""

    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        elif ext == '.docx':
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs]
            # 使用两个换行符连接段落，以模拟原始文档的结构
            content = "\n\n".join(paragraphs)

        elif ext == '.pdf':
            with fitz.open(file_path) as doc:
                # 遍历 PDF 的每一页
                for page in doc:
                    content += page.get_text() + "\n\n"  # 每页后加换行

        elif ext == '.xml':
            print(f"  [Loader] 处理 XML 文件，尝试提取文本内容...")
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
                # 简单的 XML 文本提取：移除标签
                content = re.sub(r'<[^>]+>', ' ', xml_content)
                content = re.sub(r'\s+', ' ', content).strip()
        else:
            print(f"  [Loader] 警告: 不支持的文件类型 {ext}。尝试按txt读取...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                print(f"  [Loader] 错误: 无法将 {ext} 作为txt读取。")

        print(f"  [Loader] 成功加载文档: {file_path} (大小: {len(content)} 字符)")
        return content.strip()

    except Exception as e:
        print(f"  [Loader] 加载文件 {file_path} 失败: {e}")
        return ""

# --- 模拟运行和评估 (Simulation and Evaluation) ---

if __name__ == "__main__":

    USE_REAL_API = True
    DEEPSEEK_API_KEY = "sk-ddffacf05aa840bdb09ce2175c0d8f17"

    llm_client = None
    if USE_REAL_API:
        print("=" * 50)
        print(" 正在使用真实 DeepSeek API。")
        print("=" * 50)
        try:
            llm_client = DeepSeekClient(api_key=DEEPSEEK_API_KEY)
            # 快速连接测试
            print("  [System] 正在测试 DeepSeek API 连接...")
            test_response = llm_client.generate("Hello! Respond with 'OK'")
            if "OK" in test_response:
                print(f"  [System] API 连接成功。响应: {test_response}")
            else:
                print(f"  [System] API 连接测试失败! 响应: {test_response}")
                raise Exception("API Connection Test Failed")
        except Exception as e:
            print(f"  [System] 启动 DeepSeek 客户端失败: {e}. 回退到 MockLLMClient。")
            USE_REAL_API = False

    if not USE_REAL_API:
        print("\n[System] 正在使用 MockLLMClient (模拟客户端) 运行。")
        llm_client = MockLLMClient()

    # 1. 初始化系统
    ontology_schema = {
        "predicates": [
            "manufactured_by", "located_in", "used_for", "has_property",
            "part_of", "related_to", "contains", "operates", "produces"
        ]
    }

    print("=" * 50)
    print(" ID-GraphRAG 系统模拟启动 ")
    print("=" * 50)

    FILE_V1 = "C:/Users/86135/PycharmProjects/pythonProject/MINE_txt/1.txt"
    FILE_V2 = "C:/Users/86135/PycharmProjects/pythonProject/MINE_txt/2.txt"

    # --- 场景 1: 初始构建 (Initial Construction) ---
    system_incremental = IDGraphRAGSystem(llm_client, ontology_schema)

    # (!!!) 修改: 从文件加载 document1
    document1_text = load_document_text(FILE_V1)

    print("\n--- 场景 1: 初始构建 (V1) ---")
    # (!!!) 修改: 传入文本内容 和 文件路径 (作为ID)
    report1 = system_incremental.process_document(document1_text, FILE_V1, is_update=False)
    print(f"\n[V1 报告] 处理时间: {report1['processing_time']:.4f}s, 图谱大小: {report1['knowledge_graph_size']}")

    # --- 场景 2: 增量更新 (Incremental Update) ---

    # (!!!) 修改: 从文件加载 document2
    document2_text = load_document_text(FILE_V2)

    print("\n--- 场景 2: 增量更新 (V2) ---")
    # (!!!) 修改: 传入文本内容 和 文件路径 (作为ID)
    report2 = system_incremental.process_document(document2_text, FILE_V2, is_update=True)
    print(f"\n[V2 增量报告] 处理时间: {report2['processing_time']:.4f}s, 图谱大小: {report2['knowledge_graph_size']}")

    # --- 场景 3: 全量重建 (Full Rebuild) - 用于对比效率 ---
    print("\n--- 场景 3: 全量重建 (V2) (用于对比) ---")
    system_full_rebuild = IDGraphRAGSystem(llm_client, ontology_schema)

    # (!!!) 修改: 合并 V1 和 V2 的文本内容
    document_v2_full = document1_text + "\n\n" + document2_text

    report_full = system_full_rebuild.process_document(document_v2_full, "doc1_full", is_update=False)
    print(
        f"\n[V2 全量报告] 处理时间: {report_full['processing_time']:.4f}s, 图谱大小: {report_full['knowledge_graph_size']}")

    print("\n" + "=" * 50)
    print(" 最终评估与分析 ")
    print("=" * 50)

    # --- 效率评估 (Dimension C) ---
    print("\n--- 维度 C: 增量效率评估 ---")

    # 全量存储 = V1块 + V2全量块
    storage_full = report1['triples_extracted'] + report_full['triples_extracted']  # 简化为三元组数量
    storage_inc = report1['triples_extracted'] + report2['triples_extracted']

    efficiency_metrics = system_incremental.evaluator.evaluate_efficiency(
        full_rebuild_time=report_full['processing_time'],
        incremental_update_time=report2['processing_time'],
        full_rebuild_storage=storage_full if storage_full > 0 else 1,
        incremental_storage=storage_inc if storage_inc > 0 else 1
    )

    print(f"  - 全量重建 CPU 时间: {report_full['processing_time']:.4f}s")
    print(f"  - 增量更新 CPU 时间: {report2['processing_time']:.4f}s")
    # (注意: 毫秒级的模拟运行可能导致 'inf' 或巨大差异)
    print(f"  - CPU 时间缩减倍数: {efficiency_metrics['cpu_time_reduction_multiple']:.2f}x")
    print(f"  - 存储开销缩减倍数: {efficiency_metrics['storage_overhead_reduction']:.2f}x")

    # --- 质量评估 (Dimension A & B) ---
    print("\n--- 维度 A & B: 知识图谱质量与一致性 ---")

    # 定义 V2 的“真实”知识 (CRM 应该选择公司B, 工厂Z, 'manufactured_by' 拼写正确)
    ground_truth_v2 = [
        KnowledgeTriple("设备X", "manufactured_by", "公司B", "gt", 1.0, 0),
        KnowledgeTriple("设备X", "located_in", "工厂Z", "gt", 1.0, 0),
        KnowledgeTriple("设备X", "used_for", "生产", "gt", 1.0, 0)
    ]

    final_graph = system_incremental.get_active_graph()
    quality_metrics = system_incremental.evaluator.evaluate_kg_quality(final_graph, ground_truth_v2)

    print(f"  - 最终图谱 F1-Score (vs V2 Ground Truth): {quality_metrics['f1_score']:.2f}")
    print(f"  - 冲突解决 (CRM): 在 V2 更新中触发了 {len(system_incremental.resolved_conflicts_log)} 次。")

    print("\n--- 最终知识图谱 (增量系统) ---")
    for i, triple in enumerate(final_graph):
        print(
            f"  {i + 1}. ({triple.subject}, {triple.predicate}, {triple.object}) [来源: {triple.source_chunk_id[:5]}...]")

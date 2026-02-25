# llm_client_hf.py
"""
HF-based LLM Client (drop-in replacement for the previous OpenAI-based client)
Default model: Qwen/Qwen3-8B
"""
# src/utils/llm_client_openllm.py
import json, time, logging, os, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional,Set
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect
import copy
# ✅ 兼容：不同 transformers 版本可能没有 DynamicCache
try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except Exception:
    DynamicCache = None
    _HAS_DYNAMIC_CACHE = False

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str
    response_time: float

class LLMClient:
    """
    HF Transformers-based language model client (no OpenAI dependency).
    """
    def __init__(
        self,
        api_key: str = "",
        model: str = "Qwen/Qwen3-8B",
        base_url: Optional[str] = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        enable_thinking: Optional[bool] = None,
        attn_impl: Optional[str] = "sdpa",   # 你的环境没装 flash-attn，用 sdpa 最稳
    ):
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url
        if enable_thinking is None:
            enable_thinking = os.getenv("HF_ENABLE_THINKING", "1") == "1"
        self.enable_thinking = enable_thinking

        # retry 配置（和你旧版保持一致）
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30.0

        # 将字符串 dtype 映射到 torch.dtype
        if dtype == "auto":
            dtype = "auto"
        else:
            _map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            dtype = _map.get(str(dtype).lower(), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl or "sdpa",
        )

        self.eos_token_id = self.tokenizer.eos_token_id

        # Qwen 的 </think> 特殊 token（若存在）
        try:
            self.end_think_id = self.tokenizer.convert_tokens_to_ids("</think>")
            if isinstance(self.end_think_id, list):
                self.end_think_id = self.end_think_id[0]
        except Exception:
            self.end_think_id = None

    # ===== 新增：带 logprobs 的生成，用于“不确定性估计” =====
    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 5,
    ) -> Dict[str, Any]:
        """
        生成最多 max_tokens 个 token，并返回：
        {
            "text": 生成文本（去掉 special tokens 后）,
            "logprobs": [log p(token_1), ..., log p(token_n)]
        }
        其中 logprobs 是按 time-step 对应生成出来的 token 的对数概率。
        """
        # 组装成 chat 格式，复用现有 chat_completion 的模板
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 用 Qwen 的 chat 模板展开成单个字符串
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # 某些模型不支持 enable_thinking 参数
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成配置：一定要打开 output_scores + return_dict_in_generate
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(0.0, min(2.0, temperature)),
            pad_token_id=self.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # out.sequences: [1, prompt_len + gen_len]
        # out.scores: 长度为 gen_len 的 list，每个元素 shape [1, vocab_size]
        full_ids = out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = full_ids[prompt_len:]          # 只取新生成的部分，shape [gen_len]
        scores = out.scores                      # list(len = gen_len)

        logprobs_all: List[float] = []

        # 预先收集要跳过的 special token（例如 eos, </think>）
        skip_ids = set()
        if self.eos_token_id is not None:
            skip_ids.add(self.eos_token_id)
        if getattr(self, "end_think_id", None) is not None:
            skip_ids.add(self.end_think_id)

        # 逐步计算每个生成 token 的 log p
        for t, token_id in enumerate(gen_ids):
            # scores[t][0]: logits for step t, shape [vocab]
            logits_t = scores[t][0]                        # [vocab_size]
            log_probs_t = torch.log_softmax(logits_t, dim=-1)

            lp = log_probs_t[token_id].item()
            logprobs_all.append(lp)

        # 过滤掉特别的 token（如 eos、</think>），只保留“有效 token”的 logprob
        filtered_logprobs: List[float] = []
        filtered_token_ids: List[int] = []
        for token_id, lp in zip(gen_ids.tolist(), logprobs_all):
            if token_id in skip_ids:
                continue
            filtered_token_ids.append(token_id)
            filtered_logprobs.append(lp)
            if len(filtered_logprobs) >= max_tokens:
                break

        # 如果全被滤掉了，就退回到原始前 max_tokens 个（保证不为空）
        if not filtered_logprobs:
            take = min(max_tokens, len(gen_ids))
            filtered_token_ids = gen_ids[:take].tolist()
            filtered_logprobs = logprobs_all[:take]

        # 解码成可读文本（跳过 special tokens）
        text_out = self.tokenizer.decode(filtered_token_ids, skip_special_tokens=True).strip()

        return {
            "text": text_out,
            "logprobs": filtered_logprobs,
        }

    # ===== 核心：用 transformers 实现的 chat_completion，不走 OpenAI =====
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        start = time.time()

        # Qwen 的 chat 模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen3 支持 enable_thinking 开关；其他模型忽略此参数也没关系
            enable_thinking=self.enable_thinking
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens or 1024,
            do_sample=(temperature > 0),
            temperature=max(0.0, min(2.0, temperature)),
            pad_token_id=self.eos_token_id,
        )
        # 生成
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        # 取新生成的 token
        new_ids = out_ids[0][len(inputs.input_ids[0]):].tolist()

        # 如果有 </think>，截掉思考部分
        if self.end_think_id and self.end_think_id in new_ids:
            idx = len(new_ids) - new_ids[::-1].index(self.end_think_id)
        else:
            idx = 0
        thinking = self.tokenizer.decode(new_ids[:idx], skip_special_tokens=True).strip()
        content = self.tokenizer.decode(new_ids[idx:], skip_special_tokens=True).strip()

        elapsed = time.time() - start
        usage = {
            "prompt_tokens_est": int(inputs.input_ids.numel()),
            "completion_tokens_est": len(new_ids),
            "total_tokens_est": int(inputs.input_ids.numel()) + len(new_ids),
        }
        return LLMResponse(
            content=content,
            usage=usage,
            model=self.model_name,
            finish_reason="stop",
            response_time=elapsed
        )

    def build_prefix_cache(self, prefix_text: str) -> Dict[str, Any]:
        """
        对 prefix_text 做一次前向，返回可复用的 KV cache + prefix token 长度等。
        注意：这里直接用 tokenizer(prefix_text) 编码，不走 chat_template。
        prefix_text 必须是你最终喂给模型的“完整前缀字符串”。
        """
        inputs = self.tokenizer([prefix_text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, use_cache=True)
        pkv = out.past_key_values

        return {
            "prefix_text": prefix_text,
            "past_key_values": pkv,
            "prefix_len": int(inputs["input_ids"].shape[1]),
        }

    def generate_with_logprobs_from_cache(
        self,
        prefix_cache: Dict[str, Any],
        middle_text: str,
        suffix_text: str,
        temperature: float = 0.0,
        max_tokens: int = 5,
    ) -> Dict[str, Any]:
        """
        复用 prefix_cache 的 past_key_values，只跑 middle+suffix，然后生成 max_tokens 个 token，
        返回 {"text": ..., "logprobs": [...], "middle_suffix_len": int}.
        """
        assert "past_key_values" in prefix_cache
        past = prefix_cache["past_key_values"]

        # 1) 先把 middle+suffix 当作“续接上下文”喂进去，更新 past
        cont_text = (middle_text or "") + (suffix_text or "")
        cont_inputs = self.tokenizer([cont_text], return_tensors="pt").to(self.model.device)

        # attention_mask 需要覆盖 prefix+cont 的总长度
        # 简单做法：prefix_mask 全 1 + cont_mask（全 1）
        prefix_len = int(prefix_cache.get("prefix_len", 0))
        cont_len = int(cont_inputs["input_ids"].shape[1])
        attention_mask = torch.ones((1, prefix_len + cont_len), dtype=torch.long, device=self.model.device)

        with torch.no_grad():
            out = self.model(
                input_ids=cont_inputs["input_ids"],
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        logits_last = out.logits[:, -1, :]  # 用于生成第一个新 token

        # 2) 生成 token（手动 greedy / sampling），并记录 logprobs
        skip_ids = set()
        if getattr(self, "eos_token_id", None) is not None:
            skip_ids.add(self.eos_token_id)
        if getattr(self, "end_think_id", None) is not None:
            skip_ids.add(self.end_think_id)

        gen_token_ids: List[int] = []
        gen_logprobs: List[float] = []

        # 允许跳过 special token，因此最多多跑一些步
        max_steps = max_tokens + 8

        for _ in range(max_steps):
            log_probs = torch.log_softmax(logits_last[0], dim=-1)  # [vocab]
            if temperature and temperature > 0:
                probs = torch.softmax(logits_last[0] / max(1e-6, temperature), dim=-1)
                token_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                token_id = int(torch.argmax(logits_last[0]).item())

            lp = float(log_probs[token_id].item())

            # 更新 past：把这个 token 喂进去拿下一步 logits
            token_tensor = torch.tensor([[token_id]], device=self.model.device, dtype=torch.long)
            # 新 attention_mask 长度再 +1
            attention_mask = torch.ones((1, attention_mask.shape[1] + 1), dtype=torch.long, device=self.model.device)

            with torch.no_grad():
                out_step = self.model(
                    input_ids=token_tensor,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            past = out_step.past_key_values
            logits_last = out_step.logits[:, -1, :]

            # 过滤 special token
            if token_id in skip_ids:
                continue

            gen_token_ids.append(token_id)
            gen_logprobs.append(lp)

            if len(gen_logprobs) >= max_tokens:
                break

        text_out = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()

        return {
            "text": text_out,
            "logprobs": gen_logprobs,
            "middle_suffix_len": cont_len,
            "gen_len": len(gen_token_ids),
        }

    def _supports_arg(self, arg_name: str) -> bool:
        try:
            sig = inspect.signature(self.model.forward)
            return arg_name in sig.parameters
        except Exception:
            return False

    # ---------- 核心：冻结 + 还原 ----------

    def _freeze_to_legacy(self, pkv):
        """
        ✅ 把 DynamicCache/Cache 冻结成 legacy tuple，防止后续 forward in-place 写脏缓存对象。
        """
        if pkv is None:
            return None
        if hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()  # tuple[(k,v), ...]
        # 已经是 legacy
        return pkv

    def _make_working_cache(self, legacy_pkv):
        """
        ✅ 每次使用都创建“工作 cache”，让它增长也不影响 thread_local 里保存的 legacy 前缀。
        """
        if legacy_pkv is None:
            return None
        if _HAS_DYNAMIC_CACHE and isinstance(legacy_pkv, (tuple, list)):
            return DynamicCache.from_legacy_cache(legacy_pkv)
        return legacy_pkv  # 旧版本不支持 DynamicCache 时只能传 legacy

    def _legacy_seq_len(self, legacy_pkv) -> int:
        if not isinstance(legacy_pkv, (tuple, list)) or len(legacy_pkv) == 0:
            return -1
        k0 = legacy_pkv[0][0]  # [bs, heads, seq, dim]
        return int(k0.shape[2])

    # ---------- prefix cache：只存 legacy ----------

    def build_prefix_cache_plain(self, prefix_text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            [prefix_text],
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model(**inputs, use_cache=True)

        # ✅ 关键：冻结成 legacy tuple 存起来（永不写脏）
        legacy = self._freeze_to_legacy(out.past_key_values)

        return {
            "prefix_text": prefix_text,
            "past_key_values_legacy": legacy,  # ✅ 永远 legacy
            "prefix_len": int(inputs["input_ids"].shape[1]),
            "prefix_attention_mask": inputs.get("attention_mask", None),
        }

    # ---------- plain：你可以继续用 generate（简单稳定） ----------

    def generate_with_logprobs_plain(
            self,
            plain_text: str,
            temperature: float = 0.0,
            max_tokens: int = 5,
    ) -> Dict[str, Any]:
        inputs = self.tokenizer(
            [plain_text],
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature and temperature > 0.0),
            temperature=float(max(0.0, min(2.0, temperature))),
            pad_token_id=self.eos_token_id,
            eos_token_id=self.eos_token_id,  # ✅
            output_scores=True,
            return_dict_in_generate=True,
        )

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        full_ids = out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = full_ids[prompt_len:]
        scores = out.scores

        stop_ids = set()
        if self.eos_token_id is not None:
            stop_ids.add(int(self.eos_token_id))
        if getattr(self, "end_think_id", None) is not None:
            stop_ids.add(int(self.end_think_id))

        token_ids: List[int] = []
        token_pieces: List[str] = []
        logprobs: List[float] = []

        for t, tid in enumerate(gen_ids.tolist()):
            if t >= len(scores):
                break
            if tid in stop_ids:
                break
            logits_t = scores[t][0].float()
            lp = float(torch.log_softmax(logits_t, dim=-1)[int(tid)].item())
            token_ids.append(int(tid))
            logprobs.append(lp)
            token_pieces.append(self.tokenizer.decode([int(tid)]))
            if len(logprobs) >= max_tokens:
                break

        text_out = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return {"text": text_out, "logprobs": logprobs, "token_ids": token_ids, "token_pieces": token_pieces}

    # ---------- cache 路径：每次创建 working cache（不污染前缀） ----------

    def generate_with_logprobs_from_cache_plain(
            self,
            prefix_cache: Dict[str, Any],
            continuation_text: str,
            temperature: float = 0.0,
            max_tokens: int = 5,
    ) -> Dict[str, Any]:

        legacy = prefix_cache["past_key_values_legacy"]  # ✅ 不会变
        past = self._make_working_cache(legacy)  # ✅ 每次新建工作 cache

        cont_inputs = self.tokenizer(
            [continuation_text],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        cont_ids = cont_inputs["input_ids"]
        cont_mask = cont_inputs.get("attention_mask", torch.ones_like(cont_ids))
        cont_len = int(cont_ids.shape[1])

        prefix_len = int(prefix_cache.get("prefix_len", 0))
        pref_mask = prefix_cache.get("prefix_attention_mask", None)

        if pref_mask is not None:
            attention_mask = torch.cat([pref_mask, cont_mask], dim=1)
        else:
            attention_mask = torch.ones((1, prefix_len + cont_len), device=self.model.device, dtype=torch.long)

        device = cont_ids.device
        cur_pos_start = prefix_len
        cur_pos_next = prefix_len + cont_len

        # 先把 continuation 喂进去
        kwargs = dict(
            input_ids=cont_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        if self._supports_arg("cache_position"):
            kwargs["cache_position"] = torch.arange(cur_pos_start, cur_pos_start + cont_len, device=device,
                                                    dtype=torch.long)
        if self._supports_arg("position_ids"):
            kwargs["position_ids"] = torch.arange(cur_pos_start, cur_pos_start + cont_len, device=device,
                                                  dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            out = self.model(**kwargs)

        past = out.past_key_values
        logits_last = out.logits[:, -1, :]

        stop_ids = set()
        if self.eos_token_id is not None:
            stop_ids.add(int(self.eos_token_id))
        if getattr(self, "end_think_id", None) is not None:
            stop_ids.add(int(self.end_think_id))

        token_ids: List[int] = []
        token_pieces: List[str] = []
        logprobs: List[float] = []

        cur_pos = int(cur_pos_next)

        for _ in range(max_tokens):
            logits = logits_last[0].float()
            token_id = int(torch.argmax(logits).item())  # 你 temperature=0 为主，先用 greedy
            if token_id in stop_ids:
                break

            lp = float(torch.log_softmax(logits, dim=-1)[token_id].item())
            token_ids.append(token_id)
            logprobs.append(lp)
            token_pieces.append(self.tokenizer.decode([token_id]))

            # ✅ attention_mask 正确增长
            one = torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, one], dim=1)

            step_kwargs = dict(
                input_ids=torch.tensor([[token_id]], device=device, dtype=torch.long),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            if self._supports_arg("cache_position"):
                step_kwargs["cache_position"] = torch.tensor([cur_pos], device=device, dtype=torch.long)
            if self._supports_arg("position_ids"):
                step_kwargs["position_ids"] = torch.tensor([[cur_pos]], device=device, dtype=torch.long)

            with torch.no_grad():
                out_step = self.model(**step_kwargs)

            past = out_step.past_key_values
            logits_last = out_step.logits[:, -1, :]
            cur_pos += 1

        text_out = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return {
            "text": text_out,
            "logprobs": logprobs,
            "token_ids": token_ids,
            "token_pieces": token_pieces,
            "cont_len": cont_len,
            "gen_len": len(token_ids),
        }

    def generate_text_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        resp = self.chat_completion(messages=msgs, temperature=temperature, max_tokens=max_tokens)
        return resp.content

    def generate_json_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        default_response: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            resp = self.chat_completion(messages=msgs, temperature=temperature, max_tokens=max_tokens)
            parsed = self._extract_json_from_response(resp.content, default_response=None)
            if parsed is not None and not (
                parsed.get("reason") == "JSON parsing failed, using default response" and
                parsed.get("topic_summary") == "Parsing error"
            ):
                return parsed
            # 第一次失败后，补一条“只返回JSON”提示并重试
            if attempt == 0:
                msgs.append({"role": "assistant", "content": resp.content})
                msgs.append({"role": "user", "content": "Please reply with ONLY valid JSON. No extra text."})
            temperature = min(0.5, temperature + 0.1)

        if default_response is not None:
            return default_response
        return {"error": "JSON parsing failed after all retries", "attempts": max_retries}

    # 和你旧版一致的 JSON 提取逻辑（略微整理）
    def _extract_json_from_response(self, content: str, default_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except Exception:
            pass
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except Exception:
            pass
        try:
            # 尝试抓第一个大括号块
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and start < end:
                return json.loads(content[start:end+1])
        except Exception:
            pass
        logger.warning("JSON parsing failed, returning marker response")
        return {
            "should_end": False,
            "reason": "JSON parsing failed, using default response",
            "confidence": 0.5,
            "topic_summary": "Parsing error"
        }

    def count_tokens(self, text: str) -> int:
        # 粗略估算：英文4字/词，中文1.5字/词
        chinese = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        english = len(text) - chinese
        return int(chinese / 1.5 + english / 4)

    def validate_response(self, response: str, expected_format: str = "json") -> bool:
        if expected_format == "json":
            try:
                json.loads(response)
                return True
            except Exception:
                return False
        return True

    def get_model_info(self) -> Dict[str, Any]:
        device = str(self.model.device)
        return {
            "model": self.model_name,
            "device": device,
            "enable_thinking": self.enable_thinking,
        }



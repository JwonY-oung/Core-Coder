"""LLM provider layer - thin wrapper over OpenAI-compatible APIs.

Since most providers (DeepSeek, Qwen, Kimi, GLM, Ollama, etc.) expose an
OpenAI-compatible endpoint, we just use the openai SDK directly.  Switch
provider by changing OPENAI_BASE_URL + OPENAI_API_KEY. That's it.

For providers that are NOT OpenAI-compatible (AWS Bedrock, Google Vertex,
etc.), use the LiteLLM backend which routes to 100+ providers through a
single unified interface. Set CORECODER_PROVIDER=litellm.
"""

import json
import time
from dataclasses import dataclass, field

from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# pricing per million tokens: (input, output)
# sources: openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#          platform.moonshot.ai, alibabacloud.com/help/en/model-studio
_PRICING = {
    # OpenAI - current flagships
    "gpt-5.5": (5, 30),
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - previous gen (still widely used)
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k2.5": (0.6, 3),
}

# Ceiling used for models missing from the table. estimated_cost stays None for
# them - a made-up number is worse than no number - but a hard budget has to
# assume something, and assuming the most expensive case is the only assumption
# that fails safe. Override per deployment with CORECODER_MAX_COST_PER_MTOK.
_MAX_COST_PER_MTOK = (0.27, 1.10)


class BudgetExceededError(RuntimeError):
    """Raised instead of sent when the spend cap is already blown.

    A RuntimeError, not an APIError: nothing went wrong on the provider side,
    the local ledger just stopped authorising further requests.
    """


class LLM:
    # class-level defaults so an instance built without __init__ (tests,
    # subclasses) still answers budget questions instead of AttributeError
    budget_usd: float | None = None
    fallback_models: list[str] = []
    embedding_model: str | None = None

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        fallback_models: list[str] | None = None,
        budget_usd: float | None = None,
        embedding_model: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # Embeddings normally ride the same endpoint as chat. Only when the
        # user points them somewhere else (a dedicated embedding server) do we
        # build a second client - lazily, so the common path allocates nothing.
        self.embedding_model = embedding_model
        self._embed_creds = (embedding_api_key or api_key, embedding_base_url or base_url)
        self._embed_client = None
        self.extra = kwargs  # temperature, max_tokens, etc.
        # Ordered chain of stand-ins for self.model. Defaults to empty, so
        # existing callers see identical behaviour.
        self.fallback_models = list(fallback_models or [])
        self.budget_usd = budget_usd
        # Cost is bucketed per model, not totalled: a fallback swaps the model
        # mid-conversation, and one running token counter would price the whole
        # session at whichever model happens to be current.
        self.cost_by_model: dict[str, dict] = {}
        self._fallback_used: list[str] = []
        self._on_switch = None  # set by the CLI to print a notice

    # -- token counters ----------------------------------------------------
    # Kept as properties over the per-model buckets so that every existing
    # reader (agent, context, CLI) keeps working unchanged.

    @property
    def total_prompt_tokens(self) -> int:
        return sum(b["prompt"] for b in self._ledger().values())

    @total_prompt_tokens.setter
    def total_prompt_tokens(self, value: int):
        self._bucket(self.model)["prompt"] = value

    @property
    def total_completion_tokens(self) -> int:
        return sum(b["completion"] for b in self._ledger().values())

    @total_completion_tokens.setter
    def total_completion_tokens(self, value: int):
        self._bucket(self.model)["completion"] = value

    def _ledger(self) -> dict:
        """Lazily created so an LLM built without __init__ still counts tokens."""
        ledger = self.__dict__.get("cost_by_model")
        if ledger is None:
            ledger = self.cost_by_model = {}
        return ledger

    def _bucket(self, model: str) -> dict:
        return self._ledger().setdefault(model, {"prompt": 0, "completion": 0})

    @staticmethod
    def _usd(bucket: dict, rate: tuple[float, float]) -> float:
        rin, rout = rate
        return bucket["prompt"] * rin / 1_000_000 + bucket["completion"] * rout / 1_000_000

    @property
    def estimated_cost(self) -> float | None: #按照表里的已定价模型给出精确值 未定价的模型返回none
        """Rough cost estimate in USD. None unless *every* model used is priced.

        Partial pricing would be worse than none: a session that spent 99% of
        its tokens on an unpriced model would report a confident, tiny number.
        """
        if any(m not in _PRICING for m in self.cost_by_model):
            return None
        return sum(
            self._usd(bucket, _PRICING[m]) for m, bucket in self.cost_by_model.items()
        )

    @property
    def cost_bound(self) -> float: #兜底 未定价模型按_MAX_COST_PER_MTOK 永不为none
        """Upper bound on spend, USD. Unlike estimated_cost it never returns None:
        unpriced models are charged at the worst-case rate, so a budget can't be
        evaded by naming a model the table doesn't know."""
        return sum(
            self._usd(bucket, _PRICING.get(model, _MAX_COST_PER_MTOK))
            for model, bucket in self.cost_by_model.items()
        )

    def over_budget(self) -> bool:
        """True once the hard cap is reached. No budget configured -> never True."""
        return self.budget_usd is not None and self.cost_bound >= self.budget_usd

    def budget_status(self) -> str:
        """One-line budget readout for /budget and /tokens."""
        if self.budget_usd is None:
            return f"Budget: none. Spent ~${self.cost_bound:.4f} (worst case)."
        return (
            f"Budget: ${self.cost_bound:.4f} / ${self.budget_usd:.2f}"
            f" ({self.cost_bound / self.budget_usd * 100:.1f}%)"
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls.

        Walks self.model then self.fallback_models, but only while a failure is
        still replayable. See _can_fall_back for the exact rule.
        """
        # The gate. Checked before the request, not after: money already spent
        # can't be unspent, and this is the last moment where stopping is free.
        if self.over_budget():
            raise BudgetExceededError(
                f"budget exhausted: ${self.cost_bound:.4f} of ${self.budget_usd:.2f} "
                f"spent, request not sent"
            )

        base: dict = {
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            base["tools"] = tools

        #fallback链条构造
        chain = [self.model] + [m for m in self.fallback_models if m != self.model]
        #遍历尝试
        for i, model in enumerate(chain):
            params = {**base, "model": model}
            delivered = {"text": False}

            def record(text: str, delivered=delivered) -> None:
                # The rule is about what the *user* has seen. Once real text is
                # on screen, replaying on another model would print two answers;
                # with no callback attached (context compression, one-shot
                # summarise) nothing was displayed and the partial buffer is
                # discarded on retry, so replay stays safe. Partial tool-call
                # deltas never reach the screen either, so they don't count.
                if on_token is not None:
                    delivered["text"] = True
                    on_token(text)

            try:
                return self._attempt(model, params, record if on_token else None)
            except Exception as e:  # noqa: BLE001
                # The last model in the chain has nowhere left to go.
                if i == len(chain) - 1 or delivered["text"] or not self._can_fall_back(e):
                    raise
                self._fallback_used.append(model)
                if self.over_budget():
                    # The failed attempt can itself have spent the rest of the
                    # budget. Raise here rather than at the top of the next
                    # iteration so the API error stays attached as context.
                    raise BudgetExceededError(
                        f"budget exhausted after {model} failed: "
                        f"${self.cost_bound:.4f} of ${self.budget_usd:.2f}"
                    ) from e
                if self._on_switch:
                    self._on_switch(model, chain[i + 1], e)

        raise AssertionError("unreachable: the final chain link always returns or raises")

    @staticmethod
    def _can_fall_back(error: Exception) -> bool:
        """Only provider-side failures that another endpoint might not share.

        A 4xx is a complaint about *this request* - a bad param, a context over
        the limit, an unauthorised key - and a different model would reject it
        for exactly the same reason. Retrying those just multiplies the failure.
        """
        if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(error, APIError):
            status = getattr(error, "status_code", None)
            return bool(status and status >= 500)
        return False

    def _attempt(
        self,
        model: str,
        params: dict,
        on_token=None,
    ) -> LLMResponse:
        """One full request against one model. Raises if it can't be completed."""
        # stream_options is an OpenAI extension; fall back only when the provider
        # rejects the param (400 BadRequest), not on transient errors that
        # _call_with_retry already exhausted - otherwise we'd double the retries.
        # LiteLLM never lands here: drop_params strips it for providers without
        # support, so the fallback is unreachable on that path
        params["stream_options"] = {"include_usage": True}
        try:
            stream = self._call_with_retry(params)
        except BadRequestError:
            params.pop("stream_options", None)
            stream = self._call_with_retry(params)

        content_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        prompt_tok = 0
        completion_tok = 0

        # Consumption is lazy: the provider stream is only pulled here, so a
        # failure before the first content delta leaves the turn fully replayable
        # on another model. That first delta ends the option - see `record`.
        for chunk in stream:
            # usage info comes in the final chunk; getattr-style reads stay safe
            # across OpenAI SDK objects and litellm's provider-varying shapes
            usage = getattr(chunk, "usage", None)
            if usage:
                # some providers send usage with null fields; coerce to 0 so the
                # running totals below don't blow up on int + None
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # accumulate text
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # accumulate tool calls across chunks
            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        bucket = self._bucket(model)
        bucket["prompt"] += prompt_tok
        bucket["completion"] += completion_tok

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )

    def _embedding_client(self):
        """The client used for embeddings: self.client unless a separate
        embedding endpoint was configured."""
        key, base = getattr(self, "_embed_creds", (None, None))
        if key == self.api_key and base == self.base_url:
            return self.client
        if self._embed_client is None:
            self._embed_client = OpenAI(api_key=key, base_url=base)
        return self._embed_client

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed texts via the same OpenAI-compatible client as chat.

        Reusing self.client means embeddings inherit the key/base_url the user
        already configured - no second credential set to manage. The model
        defaults to CORECODER_EMBEDDING_MODEL, then the chat model itself
        (Ollama-style local servers often serve both under one name).

        Transient failures get the same backoff-retry as chat; unlike chat
        there is no fallback chain - a different embedding model would produce
        vectors incompatible with the ones already indexed, so falling back
        mid-corpus would silently corrupt retrieval.
        """
        if not texts:
            return []
        params = {"input": texts, "model": model or self.embedding_model or self.model}
        client = self._embedding_client()
        for attempt in range(3):
            try:
                resp = client.embeddings.create(**params)
                break
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
            except APIError as e:
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < 2:
                    time.sleep(2**attempt)
                else:
                    raise
        # providers may return items out of order; the index field is the contract
        items = sorted(resp.data, key=lambda d: d.index)
        return [list(d.embedding) for d in items]

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
            except APIError as e:
                # retry 5xx server errors but not 4xx; base APIError has no status_code so read it defensively
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


class LiteLLM(LLM):
    """LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set CORECODER_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        fallback_models: list[str] | None = None,
        budget_usd: float | None = None,
        embedding_model: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an OpenAI client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs
        self.embedding_model = embedding_model
        self._embed_creds = (embedding_api_key or api_key, embedding_base_url or base_url)
        self._embed_client = None
        self.fallback_models = list(fallback_models or [])
        self.budget_usd = budget_usd
        self.cost_by_model: dict[str, dict] = {}
        self._fallback_used: list[str] = []
        self._on_switch = None

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embeddings through litellm, for providers without an OpenAI-compatible
        /embeddings endpoint. Same no-fallback rule as LLM.embed."""
        import litellm

        if not texts:
            return []
        params: dict = {"input": texts, "model": model or self.embedding_model or self.model}
        key, base = self._embed_creds
        if key:
            params["api_key"] = key
        if base:
            params["api_base"] = base
        for attempt in range(3):
            try:
                resp = litellm.embedding(**params)
                break
            except Exception as e:  # noqa: BLE001
                err = str(e).lower()
                transient = any(kw in err for kw in ["rate_limit", "timeout", "connection", "500", "502", "503", "529"])
                if transient and attempt < 2:
                    time.sleep(2**attempt)
                else:
                    raise
        items = sorted(resp.data, key=lambda d: d["index"])
        return [list(d["embedding"]) for d in items]

    @staticmethod
    def _can_fall_back(error: Exception) -> bool:
        """litellm raises its own exception types, so the openai isinstance
        checks in LLM don't apply here. Same rule - provider-side only - read
        off the status code when litellm set one, else off the message."""
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status == 429 or status >= 500
        err = str(error).lower()
        return any(kw in err for kw in ["rate_limit", "timeout", "connection", "overloaded"]) or any(
            f" {code}" in err for code in ("429", "500", "502", "503", "504", "529")
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff via litellm."""
        import litellm

        params["drop_params"] = True
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        for attempt in range(max_retries):
            try:
                return litellm.completion(**params)
            except Exception as e:
                err = str(e).lower()
                is_transient = any(
                    kw in err
                    for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"]
                )
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

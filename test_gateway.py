"""Model calls through the gateway: routes, keys, JSON, embeddings and the spending lock."""
from __future__ import annotations

from datetime import datetime

import pytest

import config
import llm


def test_gateway_address_from_the_email_is_accepted_as_it_is():
    assert config._gateway_base("https://bifrost.example.com/anthropic/") == "https://bifrost.example.com"
    assert config._gateway_base("https://bifrost.example.com/v1") == "https://bifrost.example.com"
    assert config._gateway_base("https://x.example.com/llm/openai/v1") == "https://x.example.com/llm"


def test_budget_settings_can_never_exceed_the_hard_limit():
    assert config.clamp_budget(500) == config.HARD_DAILY_LIMIT_USD == 100
    assert config.clamp_budget(-3) == 0
    assert config.clamp_budget(20) == 20


def test_routes_with_and_without_prompt_caching(gateway, monkeypatch):
    assert llm.chat("Say OK", max_tokens=5) == "OK"
    assert llm.gateway_route() == "anthropic"                 # caching on (default): the Anthropic route
    monkeypatch.setattr(config, "PROMPT_CACHE", False)
    monkeypatch.setattr(llm, "_route", None)
    assert llm.chat("Say OK", max_tokens=5) == "OK"
    assert llm.gateway_route() == "openai"


def test_falls_back_to_anthropic_route_when_openai_route_is_missing(gateway, monkeypatch):
    monkeypatch.setattr(config, "PROMPT_CACHE", False)
    gateway.state["openai_route"] = False
    assert llm.chat("Say OK", max_tokens=5) == "OK"
    assert llm.gateway_route() == "anthropic"
    assert "/anthropic/v1/messages" in gateway.state["calls"]


def test_falls_back_to_openai_route_when_anthropic_route_is_missing(gateway):
    gateway.state["anthropic_route"] = False
    assert llm.chat("Say OK", max_tokens=5) == "OK"
    assert llm.gateway_route() == "openai"


def test_material_is_cached_and_reread_cheaper(gateway):
    material = "Joining tables on a shared key. " * 400                       # long enough to cache
    first_mark = llm.meter_mark()
    llm.chat(material + " Define VTT. Answer in JSON.", system="You write lessons. " * 300)
    llm.chat([{"role": "user", "content": material + " Define VTT. Answer in JSON."},
              {"role": "assistant", "content": "draft"}, {"role": "user", "content": "Improve it."}],
             system="You write lessons. " * 300)
    first, second = llm.calls_since(first_mark)
    assert first["cache_read_tokens"] == 0 and second["cache_read_tokens"] > 1000
    assert second["cost_usd"] < first["cost_usd"]                               # read from the cache: cheaper


def test_caching_switches_off_when_the_gateway_refuses_it(gateway):
    gateway.state["reject_cache"] = True
    assert llm.chat("x " * 3000 + "Say OK", max_tokens=5) == "OK"
    assert llm._cache_ok is False


def test_image_description_works_on_both_routes(gateway, tmp_path):
    from PIL import Image
    img = tmp_path / "slide.png"
    Image.new("RGB", (64, 36), "white").save(img)
    assert llm.describe_image(img, "Describe").startswith("TYPE:")
    gateway.state["openai_route"] = False
    llm._route = None
    assert llm.describe_image(img, "Describe").startswith("TYPE:")


def test_wrong_key_gives_a_readable_message(gateway, monkeypatch):
    monkeypatch.setattr(config, "BIFROST_API_KEY", "wrong")
    with pytest.raises(llm.LLMError, match="rejected"):
        llm.chat("Say OK")


def test_json_still_works_when_gateway_refuses_response_format(gateway, monkeypatch):
    monkeypatch.setattr(config, "PROMPT_CACHE", False)   # response_format is on the OpenAI-style route
    gateway.state["reject_response_format"] = True
    schema = {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}
    assert llm.chat_json("Define VTT. Answer in JSON.", schema)["term"] == "VTT file"


def test_embeddings_switch_to_one_text_per_request(gateway):
    vectors = llm.embed(["joining tables", "merge types", "clean keys"])
    assert len(vectors) == 3 and llm._embed_one_by_one


def test_lock_per_command_stops_further_calls(gateway, monkeypatch):
    monkeypatch.setattr(config, "RUN_BUDGET_USD", 0.0001)
    llm.chat("Say OK", max_tokens=5)                  # first call is allowed and costs something
    with pytest.raises(llm.BudgetExceeded, match="per command"):
        llm.chat("Say OK", max_tokens=5)


def test_daily_lock_counts_what_was_spent_earlier_today(gateway):
    config.COST_LOG.write_text("time,step,backend,model,in_tokens,in_tokens_est,out_tokens,seconds,est_cost_usd\n"
                               f"{datetime.now():%Y-%m-%dT%H:%M:%S},storyboard,bifrost,m,1,1,1,1,25.0\n")
    with pytest.raises(llm.BudgetExceeded, match="today"):
        llm.chat("Say OK")


def test_cost_estimate_is_never_zero_and_regional_ids_cost_more():
    assert llm.price("bedrock/some-new-model", "bifrost") == (5.0, 25.0)
    eu = llm.price("bedrock/eu.anthropic.claude-sonnet-5", "bifrost")
    plain = llm.price("anthropic.claude-sonnet-5", "bifrost")
    assert eu[0] > plain[0] and llm.price("llama3.1:8b", "ollama") == (0.0, 0.0)


def test_which_models_the_key_may_use(gateway, monkeypatch, capsys):
    """doctor.py --models when the gateway gives no list: each Claude model is tried once."""
    import fake_gateway

    import doctor
    haiku, opus = "bedrock/eu.anthropic.claude-haiku-5-test", "bedrock/eu.anthropic.claude-opus-4-6-v1"
    denied = "bedrock/eu.anthropic.claude-opus-4-7"
    monkeypatch.setattr(fake_gateway, "MODELS", fake_gateway.MODELS + [haiku, opus, denied])
    monkeypatch.setenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", haiku)
    gateway.state["denied"] = {denied}
    found = llm.candidate_models(["my-own-id", "bedrock/deepseek-r1"])
    assert found[config.CHAT_MODEL] == ["CHAT_MODEL", "VISION_MODEL"] and found[haiku] == ["ANTHROPIC_DEFAULT_HAIKU_MODEL"]
    assert opus in found and found["my-own-id"] == ["--try"]                     # known ids get CHAT_MODEL's prefix
    assert llm.try_model(config.CHAT_MODEL) == ("ok", "answers")
    assert llm.try_model(denied)[0] == "no access" and "403" in llm.try_model(denied)[1]
    assert llm.try_model("bedrock/eu.anthropic.claude-opus-4-1-20250805-v1:0")[0] == "unknown id"
    assert llm.try_model("bedrock/deepseek-r1")[0] == "not allowed"              # never even sent
    doctor.show_access([])
    out = capsys.readouterr().out
    assert f"[OK]          {haiku}" in out and "[NO ACCESS  ]" in out and "[UNKNOWN ID ]" in out
    assert f"RERANK_MODEL={haiku}" in out and f"JUDGE_MODEL={opus}" in out       # the denied, newer Opus is not offered

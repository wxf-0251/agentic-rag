import agentic_rag.config as config


def test_deep_research_defaults_allow_multi_step_reports():
    assert config.MAX_SUB_QUESTIONS == 7
    assert config.MAX_REVIEW_ITERATIONS == 3
    assert config.MAX_TOKENS_PER_SECTION >= 6000
    assert config.ANTHROPIC_TIMEOUT_SECONDS >= 90
    assert config.ANTHROPIC_MAX_RETRIES >= 4

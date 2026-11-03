# API Configuration

## Gemini API setup

1. Sign in to [Google AI Studio](https://aistudio.google.com/apikey).
2. Create an API key.
3. Copy it into your `.env` file, as described below.

Rate limits and pricing change, so check [Google's pricing page](https://ai.google.dev/pricing) for current details.

## Environment variable configuration

Create `.env` in the project folder:

```powershell
Copy-Item .env.example .env
notepad .env
```

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | For AI features | Your Gemini API key |
| `GEMINI_MODEL` | No | Overrides the default model |

Restart the app after changing `.env`.

Without a key, you can still upload, parse and index documents, but the AI Assistant and Contract Intelligence will show a clear error.

## API key security

- Keep the key only in `.env`.
- Never put a real key in source code, documentation, screenshots or chat messages.
- `.env` is listed in `.gitignore`, and it is excluded from the ZIP.
- If a key is exposed, delete it in Google AI Studio and create a new one.

## Data privacy

Each question sends the retrieved passages to Google. Each analysis sends the contract text, or for a very long contract, selected passages. Google's pricing page says that free-tier usage may be used to improve its products. Do not process confidential contracts with a free-tier key.

## Model configuration

The default model is `gemini-3.5-flash-lite`, chosen because it was listed in November 2026 with a free tier and no announced shutdown date.

To change it, add this line to `.env`:

```text
GEMINI_MODEL=gemini-3.8-flash
```

Gemini models are retired regularly, and `gemini-2.0-flash` shut down on 1 June 2026. If you see a "model not found" error, check the [Gemini deprecations page](https://ai.google.dev/gemini-api/docs/deprecations) for a current model.

Other settings are in `config/settings.py`:

| Setting | Default | Purpose |
|---|---|---|
| `llm_temperature` | `0.1` | Lower values give more consistent output |
| `llm_timeout_seconds` | `60` | Time to wait for a response |
| `llm_max_retries` | `2` | Automatic retries by the client library |

## Error handling

`core/llm_client.py` converts provider errors into clear messages:

| Situation | Message |
|---|---|
| Missing key | No Gemini API key found |
| Invalid key | Gemini rejected the API key |
| Rate limit | The rate limit was reached |
| Retired or mistyped model | The model was not found |
| Slow response | Gemini took too long |
| Network failure | Gemini could not be reached |
| Blocked or empty output | Gemini returned an empty response |

The original exception text is never displayed, because it could echo contract content. The app keeps running after these errors, and in Contract Intelligence, one failed analysis does not discard the others.

## Replacing Gemini with another provider

The application depends only on one method:

```python
def generate(self, system_prompt: str, user_prompt: str) -> str: ...
```

For example, to add Anthropic Claude:

**1. Install the integration and add it to `requirements.txt`:**

```powershell
pip install langchain-anthropic
```

**2. Add `import os` and this client to `core/llm_client.py`:**

```python
class ClaudeClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        model = ChatAnthropic(model=os.environ["ANTHROPIC_MODEL"])
        try:
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as exc:
            raise LLMError(describe_llm_failure(exc)) from None
        return content_to_text(response.content).strip()
```

**3. Register it in `create_llm_client`:**

```python
if settings.llm_provider == "claude":
    return ClaudeClient(settings)
```

**4. Set the provider** by changing `llm_provider` in `config/settings.py`, and add `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` to `.env`.

This example shows the pattern. It is a proposal, not an implemented feature, and it has not been tested. The RAG pipeline, prompts, features and interface need no changes, although prompts are worth re-testing with any new model. `describe_llm_failure` says "Gemini" in its messages, so update that wording too.

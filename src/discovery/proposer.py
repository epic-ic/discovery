"""Proposes new models by asking an LLM to rewrite a model's marked blocks."""

from dataclasses import dataclass, field
from typing import Any

from discovery.model import Model, parse_blocks

SYSTEM_PROMPT = """\
You are a scientist searching for a better model. Models are programs.

You will be shown a program. Regions you may rewrite are delimited by
`# @discovery:begin <name>` and `# @discovery:end` comments. Everything outside
those markers is fixed: it is the harness that runs and scores the model, and
you must not restate or alter it.

Reply with a single fenced code block containing only the rewritten regions,
each still wrapped in its own begin/end markers and keeping its original name.
Do not include any other code. Keep indentation exactly as given. Propose a
genuine change: a different functional form, a different distribution, a new
term or structure, not a cosmetic edit.\
"""


@dataclass
class ModelProposer:
    """Stores the LLM url and the transcript, and creates the prompt.

    The transcript is the running conversation: it is what makes successive
    proposals aware of what has already been tried and how it scored.
    """

    url: str
    llm_model: str
    api_key: str = "EMPTY"
    temperature: float = 1.0
    system_prompt: str = SYSTEM_PROMPT
    client: Any = None
    transcript: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(base_url=self.url, api_key=self.api_key)
        if not self.transcript:
            self.transcript.append({"role": "system", "content": self.system_prompt})

    def prompt(self, model: Model, feedback: str | None = None) -> str:
        """The user turn describing the model to improve."""
        names = ", ".join(block.name for block in model.blocks) or "(none)"
        parts = [
            "Here is the current model.",
            "",
            f"```python\n{model.source.rstrip()}\n```",
            "",
            f"Rewritable regions: {names}.",
        ]
        if feedback:
            parts += ["", "Results for this model:", feedback]
        parts += ["", "Propose an improved version of the rewritable regions."]
        return "\n".join(parts)

    def propose(self, model: Model, feedback: str | None = None, **kwargs: Any) -> Model:
        """Ask for one new model, appending both turns to the transcript."""
        self.transcript.append({"role": "user", "content": self.prompt(model, feedback)})
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=list(self.transcript),
            temperature=self.temperature,
            **kwargs,
        )
        reply = response.choices[0].message.content or ""
        self.transcript.append({"role": "assistant", "content": reply})

        bodies = parse_blocks(reply)
        if not bodies:
            raise ValueError("proposal contained no @discovery blocks")
        return model.substitute(bodies)

    def reset(self) -> None:
        """Drop the conversation, keeping only the system prompt."""
        self.transcript[:] = [{"role": "system", "content": self.system_prompt}]

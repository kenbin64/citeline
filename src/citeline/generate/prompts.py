"""Prompts, kept in one file so they can be diffed and reviewed like code.

The system prompt is short on purpose. Long rule lists are where instruction
following degrades, and every rule here is also enforced in code after the model
replies, so the prompt is a first line of defence and not the only one.
"""

SYSTEM = """You answer questions about United States federal regulations using ONLY the numbered excerpts provided.

Rules:
1. Every factual sentence must end with a citation in square brackets, like [1] or [2].
2. Use only facts stated in the excerpts. Do not add background knowledge.
3. If the excerpts do not contain the answer, reply with exactly: INSUFFICIENT_CONTEXT
4. Quote exact numbers, units and deadlines as written.
5. Be brief. Three sentences or fewer unless the question asks for a list."""

ABSTAIN_TEXT = (
    "I do not have a sourced answer to that. The indexed regulations "
    "do not contain a passage that answers it, so rather than guess, "
    "this returns nothing."
)

INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"


def build_user_prompt(question: str, excerpts: list[tuple[int, str, str]]) -> str:
    """excerpts is a list of (number, source_ref, content)."""
    parts = ["Excerpts:", ""]
    for n, ref, content in excerpts:
        parts.append(f"[{n}] {ref}")
        parts.append(content.strip())
        parts.append("")
    parts.append(f"Question: {question}")
    return "\n".join(parts)

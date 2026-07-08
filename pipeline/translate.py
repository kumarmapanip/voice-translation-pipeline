import json
import os
from collections import deque

from google import genai
from google.genai import types

from .config import GROQ_MT_MODEL, MT_MODEL
from .normalizer import normalize

SYSTEM = """Translate English clauses into natural spoken Hindi. Output only the Hindi text.

Rules:
- Keep proper nouns, product names and technical terms in Latin script (Google Meet, Asterisk).
- Drop fillers (uh, um, you know, like).
- Localize numbers, times and dates: "5 PM" -> "शाम 5 बजे".
- Spell out any remaining abbreviations naturally in Hindi.
- Preserve perspective: "I" -> "मैं", "we" -> "हम", "you" -> "आप" (or "तुम" if clearly informal).
- The clause may continue a previous sentence - make it flow naturally without repeating earlier output.
- "Move the meeting to [time]" means reschedule/fix the meeting for that time, not extend it. Use "रख सकते हैं" or "तय कर सकते हैं", never "बढ़ा सकते हैं" for this meaning.
- If the clause is only filler or noise, output an empty string.

Examples:
English: Can we move the meeting to 5 PM tomorrow?
Hindi: क्या हम मीटिंग को कल शाम 5 बजे के लिए रख सकते हैं?

English: Also, can you send the Google Meet link?
Hindi: साथ ही, क्या आप Google Meet का लिंक भेज सकते हैं?"""


class BaseTranslator:
    def __init__(self, history_len: int = 6):
        self.history: deque[tuple[str, str]] = deque(maxlen=history_len)
        self.glossary = {}
        if os.path.exists("glossary.json"):
            with open("glossary.json") as f:
                self.glossary = json.load(f)

    def _system(self) -> str:
        system = SYSTEM
        if self.glossary:
            system += "\n- Glossary (always use these renderings): " + json.dumps(
                self.glossary, ensure_ascii=False
            )
        return system

    async def translate(self, text: str) -> str:
        # History keeps the normalized form — it's what the model actually saw,
        # so the rolling context stays consistent with its own past turns.
        normalized = normalize(text)
        hindi = (await self._generate(normalized)).strip()
        if hindi:
            self.history.append((normalized, hindi))
        return hindi


class GeminiTranslator(BaseTranslator):
    def __init__(self, client: genai.Client, model: str = MT_MODEL, history_len: int = 6):
        super().__init__(history_len)
        self.client = client
        self.model = model

    async def _generate(self, text: str) -> str:
        contents = []
        for en, hi in self.history:
            contents.append(types.Content(role="user", parts=[types.Part(text=en)]))
            contents.append(types.Content(role="model", parts=[types.Part(text=hi)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system(),
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text or ""


class GroqTranslator(BaseTranslator):
    def __init__(self, client, model: str = GROQ_MT_MODEL, history_len: int = 6):
        super().__init__(history_len)
        self.client = client
        self.model = model

    async def _generate(self, text: str) -> str:
        messages = [{"role": "system", "content": self._system()}]
        for en, hi in self.history:
            messages.append({"role": "user", "content": en})
            messages.append({"role": "assistant", "content": hi})
        messages.append({"role": "user", "content": text})
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0, max_tokens=512
        )
        return resp.choices[0].message.content or ""

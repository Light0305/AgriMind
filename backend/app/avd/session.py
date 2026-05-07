"""AVD session — manages state for a multi-turn diagnostic conversation."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

from PIL import Image

from app.schemas import AVDQuestion, DiagnosisContext


@dataclass
class AVDSession:
    """Manages state for a multi-turn AVD conversation.

    Each session tracks accumulated images, per-image descriptions, and the
    questions the engine has already asked so it can avoid repetition.
    """

    session_id: str
    images: list[Image.Image] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    questions_asked: list[AVDQuestion] = field(default_factory=list)
    user_context: str | None = None
    max_rounds: int = 3
    sufficiency_threshold: float = 0.75

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_round(self) -> int:
        """How many images have been collected so far."""
        return len(self.images)

    @property
    def can_ask_more(self) -> bool:
        """Whether we haven't hit the max-rounds cap yet."""
        return self.current_round < self.max_rounds

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_image(self, image: Image.Image, description: str = "") -> None:
        """Append a new image (and its optional description) to the session."""
        self.images.append(image)
        self.descriptions.append(description)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_diagnosis_context(self) -> DiagnosisContext:
        """Convert session state to a :class:`DiagnosisContext` for DDP.

        PIL images are serialised to base64-encoded PNGs.
        """
        images_b64: list[str] = []
        for img in self.images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode())

        return DiagnosisContext(
            images=images_b64,
            image_descriptions=self.descriptions,
            user_context=self.user_context,
        )

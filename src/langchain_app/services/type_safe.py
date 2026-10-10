"""TypeSafe judgments used while generating yoga sequences."""

from collections.abc import Sequence

from langchain_typesafe import Choice, TypeSafeClassifier
from langchain_typesafe.client import TypeSafeError
from pydantic import JsonValue

from langchain_app.schemas import CatalogItemSchema


class TypeSafeServiceError(Exception):
    """Base exception raised by :class:`TypeSafeService`."""


class TypeSafeSelectionError(TypeSafeServiceError):
    """Raised when Jev cannot select the next catalog item."""


class TypeSafeService:
    """Use Jev to make typed decisions for generated yoga sequences."""

    _QUESTION_ID = "next_yoga_pose"
    _RULES = (
        "Each element must be a valid yoga pose.",
        "Do not repeat any pose already selected for the sequence.",
        "Prefer alternating between flexion and extension poses.",
        "Add at least one twist pose.",
        "Do not use 'starting' poses except at the beginning of the sequence.",
        "If the first pose is lying down (any Position/Lying tag), the second pose "
        "must have the Position/Lying on the back tag.",
        "If the first pose has the Position/Standing tag, prefer a second pose with "
        "Position/Standing; the second pose must not have any Position/Lying tag.",
        "Prefer ending with a finishing pose."
    )

    def __init__(
        self,
        *,
        classifier: TypeSafeClassifier | None = None,
    ) -> None:
        self._classifier = classifier or TypeSafeClassifier()

    def choose_next_catalog_item(
        self,
        catalog_items: Sequence[CatalogItemSchema],
        *,
        selected_items: Sequence[CatalogItemSchema],
        request: str,
        elements_count: int,
    ) -> CatalogItemSchema:
        """Choose the next item from a closed set of catalog candidates."""

        next_element_position = len(selected_items) + 1
        if next_element_position > elements_count:
            raise ValueError("The requested sequence already has all of its elements")

        selected_ids = {item.id for item in selected_items}
        candidates = [item for item in catalog_items if item.id not in selected_ids]
        if not candidates:
            raise TypeSafeSelectionError(
                "No unused catalog items with thumbnails are available for selection"
            )
        if len(candidates) == 1:
            return candidates[0]

        candidates_by_id = {str(item.id): item for item in candidates}
        criteria: dict[str, JsonValue] = {}
        for item_id, item in candidates_by_id.items():
            tag_names: list[JsonValue] = [
                tag_name for tag_name in item.catalog_tag_names
            ]
            item_criteria: dict[str, JsonValue] = {
                "id": item.id,
                "name": item.name,
                "bilateral": item.bilateral,
                "catalog_tag_names": tag_names,
            }
            criteria[item_id] = item_criteria

        state = {
            "request": request,
            "rules": [
                *self._RULES,
                f"The sequence must contain exactly {elements_count} poses.",
            ],
            "sequence_progress": {
                "elements_count": elements_count,
                "selected_elements_count": len(selected_items),
                "next_element_position": next_element_position,
                "is_selecting_last_element": (
                    next_element_position == elements_count
                ),
            },
            "selected_catalog_items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "bilateral": item.bilateral,
                    "catalog_tag_names": item.catalog_tag_names,
                }
                for item in selected_items
            ],
        }

        try:
            response = self._classifier.invoke(
                {
                    "state": state,
                    "questions": {
                        self._QUESTION_ID: Choice(
                            instructions=(
                                "Which catalog item is the best next yoga pose for "
                                "the requested sequence and its rules?"
                            ),
                            criteria=criteria,
                        )
                    },
                }
            )
        except TypeSafeError as exc:
            raise TypeSafeSelectionError(
                "Could not select the next catalog item"
            ) from exc

        answer = response.choices.get(self._QUESTION_ID)
        if answer is None or answer.choice not in candidates_by_id:
            raise TypeSafeSelectionError(
                "Jev returned an invalid catalog item selection"
            )

        return candidates_by_id[answer.choice]

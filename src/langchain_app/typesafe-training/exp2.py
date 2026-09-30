from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()
client = TypeSafeClient()

ticket = """
    You are composing a yoga workout sequence.
    There are rules to follow to select what element should be next.
    The rules are as follows:
    1. Each element must be a valid yoga pose.
    2. No pose should be repeated consecutively.
    3. The sequence should alternate between flexion and extension poses.
    4. The sequence should include a balance of strength, flexibility, and relaxation poses.

    There are elements already in the sequence that must be considered when adding the next element.
    Elements are listed below.
    [
        "catalog_item": {
            "order": 1,
            "name": "Salabhasana",
            "catalog_tag_names": [
                "Arms/at the chest",
                "Arms/bent",
                "Back/arch",
                "Contact/abdomen",
                "Contact/feet",
                "Contact/pelvis",
                "Feet/on insteps",
                "Head/neutral",
                "Knees/together",
                "Legs/straight",
                "Legs/together",
                "Pelvis/on the floor",
                "Position/Lying on the stomach",
                "Torso/horizontal"
            ]
        },
        "catalog_item": {
            "name": "Paschimottanasana",
            "catalog_tag_names": [
                "Arms/bent",
                "Arms/down",
                "Back/rounding",
                "Contact/feet",
                "Contact/pelvis",
                "Feet/flexed",
                "Knees/together",
                "Legs/forward",
                "Legs/straight",
                "Legs/together",
                "Pelvis/on the floor",
                "Position/Sitting",
                "Torso/forward bend"
            ]
        },
        "catalog_item": {
            "name": "Setu Bandhasana",
            "catalog_tag_names": [
                "Arms/down",
                "Arms/holding the legs",
                "Arms/straight",
                "Back/arch",
                "Contact/elbows",
                "Contact/feet",
                "Contact/head",
                "Contact/shoulders",
                "Head/forward bend",
                "Legs/bent",
                "Pelvis/lifted",
                "Position/Support",
                "Torso/horizontal"
            ]
        }
    ]
"""

response = client.system_one(
    state=ticket,
    questions={
        "next_yoga_pose": Choice(
            instructions="""
                What item is the most valid next yoga pose in the sequence according to the rules?
            """,
            criteria={
                "Halasana": {
                    "catalog_tag_names": [
                        "Arms/holding the legs",
                        "Arms/straight",
                        "Arms/up",
                        "Back/straight",
                        "Contact/back",
                        "Contact/hands",
                        "Contact/head",
                        "Contact/shoulders",
                        "Feet/together",
                        "Head/forward bend",
                        "Knees/together",
                        "Legs/back",
                        "Legs/straight",
                        "Legs/together",
                        "Pelvis/lifted",
                        "Position/Inverted",
                        "Position/Lying on the back",
                        "Torso/forward bend"
                    ]
                },
                "Matsyasana": {
                    "catalog_tag_names": [
                        "Arms/bent",
                        "Back/arch",
                        "Contact/back",
                        "Contact/elbows",
                        "Contact/hands",
                        "Contact/head",
                        "Contact/pelvis",
                        "Feet/pointed",
                        "Feet/together",
                        "Head/backward bend",
                        "Knees/together",
                        "Legs/forward",
                        "Legs/straight",
                        "Legs/together",
                        "Pelvis/on the floor",
                        "Position/Lying on the back",
                        "Torso/horizontal"
                    ]
                }
            }
        )
    }
)

print(response.answers["next_yoga_pose"])



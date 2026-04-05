from vln_perception.plugins.mock_grounding import MockGroundingBackend


def load_grounding_backend(name: str, landmarks=None):
    # The default implementation is intentionally deterministic.
    # Real backends can be added later without changing node contracts.
    return MockGroundingBackend(landmarks=landmarks)


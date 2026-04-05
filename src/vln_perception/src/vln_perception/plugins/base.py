class GroundingBackend:
    def observe_mission(self, mission):
        raise NotImplementedError

    def debug_boxes_for_mission(self, mission):
        return []

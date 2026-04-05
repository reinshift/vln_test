from vln_sim.vehicle_adapter import VehicleAdapter


class RoverAdapterStub(VehicleAdapter):
    def reset_episode(self, episode_name: str, hard_reset: bool) -> None:
        del episode_name, hard_reset


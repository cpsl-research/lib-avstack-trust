from typing import TYPE_CHECKING, Dict, List, Tuple


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import Position, Shape

import numpy as np
from avstack.geometry.fov import box_in_fov
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .config import MATE
from .distributions import TrustDistribution


class Psm:
    def __init__(
        self,
        timestamp: float,
        target: str,
        value: float,
        confidence: float,
        source: str,
    ):
        self.timestamp = timestamp
        self.target = target
        self.value = value
        self.confidence = confidence
        self.source = source

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Pseudomeasurement: ({self.value}, {self.confidence})"


class PsmArray:
    def __init__(self, timestamp: float, psms: List[Psm]):
        self.timestamp = timestamp
        self.psms = psms

    def __add__(self, other: "PsmArray"):
        return PsmArray(self.timestamp, self.psms + other.psms)

    def __iter__(self):
        return iter(self.psms)

    def __getitem__(self, key: int) -> Psm:
        return self.psms[key]

    def __len__(self) -> int:
        return len(self.psms)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"PsmArray at time {self.timestamp}, {self.psms}"

    def append(self, other: "Psm"):
        self.psms.append(other)

    def extend(self, other: "PsmArray"):
        self.psms.extend(other.psms)

    def reduce_by_target(self) -> Dict[int, "PsmArray"]:
        psms_target = {}
        for psm in self.psms:
            if psm.target not in psms_target:
                psms_target[psm.target] = PsmArray(self.timestamp, [])
            psms_target[psm.target].append(psm)
        return psms_target


class PsmDiagnostics:
    def __init__(self):
        self.diag_agent = {}
        self.diag_track = {}

    def __getitem__(self, key: str):
        if key == "agent":
            return self.diag_agent
        elif key == "track":
            return self.diag_track
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key == "agent":
            self.diag_agent = value
        elif key == "track":
            self.diag_track = value
        else:
            raise KeyError(key)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        out_str = f"PSM Diagnostic Data Structure:\n"
        for i, ident in enumerate(["Agent", "Track"]):
            diag = self.diag_agent if i == 0 else self.diag_track
            header = "-" * 12 + "\n" + " " * 5 + f"{ident} Diags:\n"
            content = "\n".join(
                [
                    f"{ident} {d_ID}: [{', '.join([f'{d_k}: ' + d_v['identifier'] for d_k, d_v in diag[d_ID].items()])}]"
                    for d_ID in diag
                ]
            )
            out_str += header + content
        return out_str


class PsmGenerator:
    def __call__(self, *args, **kwargs) -> Tuple[PsmArray, PsmArray]:
        raise NotImplementedError

    def psms_agents(self):
        raise NotImplementedError

    def psms_agent(self):
        raise NotImplementedError

    def psms_tracks(self):
        raise NotImplementedError

    def psms_track(self):
        raise NotImplementedError


@MATE.register_module()
class ViewBasedPsm(PsmGenerator):
    """Simple overlap-based PSM function"""

    def __init__(self, assign_radius: float = 1.0, verbose: bool = False):
        self.assign_radius = assign_radius
        self.verbose = verbose
        self._diagnostics = PsmDiagnostics()
        self._assign_diagnostics = {}

    def __call__(
        self,
        position_agents: Dict[int, "Position"],
        fov_agents: Dict[int, "Shape"],
        tracks_agents: Dict[int, "DataContainer"],
        tracks_cc: "DataContainer",
        trust_agents: Dict[int, TrustDistribution],
        trust_tracks: Dict[int, TrustDistribution],
        d_self_thresh: float = 1.0,
    ) -> Tuple[PsmArray, PsmArray]:
        # -- reset the diagnostics
        self._diagnostics["agent"] = {
            id_agent: {track.ID: {} for track in tracks_cc}
            for id_agent in position_agents
        }
        self._diagnostics["track"] = {
            track.ID: {id_agent: {} for id_agent in position_agents}
            for track in tracks_cc
        }
        self._assign_diagnostics = {}

        # -- allocate trust dict space
        psms_agents = PsmArray(timestamp=tracks_cc.timestamp, psms=[])
        psms_tracks = PsmArray(timestamp=tracks_cc.timestamp, psms=[])

        # -- perform agent-wise assignment
        cc_tracks_pos = [t.x[:2] for t in tracks_cc]
        timestamp = tracks_cc.timestamp
        for i_agent in position_agents:
            position_agent = position_agents[i_agent]
            fov_agent = fov_agents[i_agent]

            # -- assignment for this agent
            A = build_A_from_distance(
                [t.x[:2] for t in tracks_agents[i_agent]],
                cc_tracks_pos,
                check_reference=False,
            )
            assign = gnn_single_frame_assign(A, cost_threshold=self.assign_radius)
            self._assign_diagnostics[i_agent] = A

            # ===============================================
            # -- PSMs for assignments
            # ===============================================

            for i_trk_agent, j_trk_central in assign.iterate_over(
                "rows", with_cost=False
            ).items():
                # get ID of the central track
                j_trk_central = j_trk_central[0]  # assumes one assignment
                ID_central = tracks_cc[j_trk_central].ID

                # -- agent PSM for assignment
                psms_agents.append(
                    Psm(
                        timestamp=timestamp,
                        target=i_agent,
                        value=trust_tracks[ID_central].mean,
                        confidence=1 - trust_tracks[ID_central].variance,
                        source=ID_central,
                    )
                )
                self._diagnostics["agent"][i_agent][ID_central] = {
                    "type": "assigned",
                    "in_fov": True,
                    "distance": A[i_trk_agent, j_trk_central],
                    "track_is_agent": False,
                    "identifier": "agent_assigned",
                }

                # -- track PSM for assignment
                psms_tracks.append(
                    Psm(
                        timestamp=timestamp,
                        target=ID_central,
                        value=1.0,
                        confidence=trust_agents[i_agent].mean,
                        source=i_agent,
                    )
                )
                self._diagnostics["track"][ID_central][i_agent] = {
                    "type": "assigned",
                    "in_fov": True,
                    "distance": A[i_trk_agent, j_trk_central],
                    "track_is_agent": False,
                    "identifier": "track_assigned",
                }

            # ===============================================
            # -- PSMs for local not in central
            # ===============================================

            # ===============================================
            # -- PSMs for central not in local
            # ===============================================

            for j_trk_central in assign.unassigned_cols:
                ID_central = tracks_cc[j_trk_central].ID
                track_central = tracks_cc[j_trk_central]
                d_self = np.linalg.norm(position_agent.x[:3] - track_central.x[:3])
                if d_self < d_self_thresh:
                    # agent can't see itself - track improves, agent neutral
                    self._diagnostics["agent"][i_agent][ID_central] = {
                        "type": "unassigned",
                        "in_fov": True,
                        "distance": d_self,
                        "track_is_agent": True,
                        "identifier": "agent_central_track_is_agent",
                    }
                    psms_tracks.append(
                        Psm(
                            timestamp=timestamp,
                            target=ID_central,
                            value=1.0,
                            confidence=1.0,
                            source=i_agent,
                        )
                    )
                    self._diagnostics["track"][ID_central][i_agent] = {
                        "type": "unassigned",
                        "in_fov": True,
                        "distance": d_self,
                        "track_is_agent": True,
                        "identifier": "agent_central_track_is_agent",
                    }
                else:
                    # it's not the agent - check if in fov
                    if box_in_fov(track_central.box3d, fov_agent):
                        # if in fov, negative PSMs
                        psms_agents.append(
                            Psm(
                                timestamp=timestamp,
                                target=i_agent,
                                value=1 - trust_tracks[ID_central].mean,
                                confidence=1 - trust_tracks[ID_central].variance,
                                source=ID_central,
                            )
                        )
                        self._diagnostics["agent"][i_agent][ID_central] = {
                            "type": "unassigned",
                            "in_fov": True,
                            "distance": None,
                            "track_is_agent": False,
                            "identifier": "agent_unassigned_central_in_fov",
                        }
                        psms_tracks.append(
                            Psm(
                                timestamp=timestamp,
                                target=ID_central,
                                value=0.0,
                                confidence=trust_agents[i_agent].mean,
                                source=i_agent,
                            )
                        )
                        self._diagnostics["track"][ID_central][i_agent] = {
                            "type": "unassigned",
                            "in_fov": True,
                            "distance": None,
                            "track_is_agent": False,
                            "identifier": "track_unassigned_central_in_fov",
                        }
                    else:
                        # if not in fov, pass
                        self._diagnostics["agent"][i_agent][ID_central] = {
                            "type": "unassigned",
                            "in_fov": False,
                            "distance": None,
                            "track_is_agent": False,
                            "identifier": "agent_unassigned_central_not_in_fov",
                        }
                        self._diagnostics["track"][ID_central][i_agent] = {
                            "type": "unassigned",
                            "in_fov": False,
                            "distance": None,
                            "track_is_agent": False,
                            "identifier": "track_unassigned_central_not_in_fov",
                        }

        return psms_agents, psms_tracks

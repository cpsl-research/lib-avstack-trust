from mate.measurement import AgentScorer


def test_agent_scorer_trivial():
    scorer = AgentScorer(
        connective={"type": "StandardFuzzy", "operator": {"type": "ZadehOperator"}}
    )
    tracks = {}
    poses = {}
    fovs = {}
    # score = scorer(tracks=tracks, poses=poses, fovs=fovs, agents=[])

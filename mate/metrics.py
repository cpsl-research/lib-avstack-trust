from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign


def classify(trust):
    if trust < 0.4:
        return -1
    elif trust > 0.7:
        return 1
    else:
        return 0


def run_metrics(trust_estimator, fovs, objects):
    # assign last tracks to truths
    tracks = trust_estimator.tracks
    A = build_A_from_distance(tracks, objects)
    assigns = gnn_single_frame_assign(A, cost_threshold=0.1)
    IDs_assign = [tracks[j_track].ID for j_track in assigns._row_to_col]

    # make track trust predictions
    means = {
        ID: ps.alpha / (ps.alpha + ps.beta)
        for ID, ps in trust_estimator.track_trust.items()
    }
    predictions = {ID: classify(mean) for ID, mean in means.items()}

    # get number of observers per track/lone-object
    n_obs_unassign = {
        i: sum([fov.check_point(objects[idx][:2]) for fov in fovs])
        for i, idx in enumerate(assigns.unassigned_cols)
    }
    n_obs_trks = {
        obj.ID: sum([fov.check_point(obj.x[:2]) for fov in fovs]) for obj in tracks
    }

    # get metrics based on num of observers
    cases_all = {}
    for i_obs in range(1, len(fovs) + 1, 1):
        cases = {}
        # case 0: false negative of true object (no track)
        cases[0] = sum([True for i in n_obs_unassign if n_obs_unassign[i] == i_obs])

        # case 1: true track predicts as valid
        cases[1] = sum(
            [
                pred == 1
                for ID, pred in predictions.items()
                if (ID in IDs_assign) and (n_obs_trks[ID] == i_obs)
            ]
        )

        # case 2: true track predicts as unknown
        cases[2] = sum(
            [
                pred == 0
                for ID, pred in predictions.items()
                if (ID in IDs_assign) and (n_obs_trks[ID] == i_obs)
            ]
        )

        # case 3: true track predicts as invalid
        cases[3] = sum(
            [
                pred == -1
                for ID, pred in predictions.items()
                if (ID in IDs_assign) and (n_obs_trks[ID] == i_obs)
            ]
        )

        # case 4: false track predicts as valid
        cases[4] = sum(
            [
                pred == 1
                for ID, pred in predictions.items()
                if (ID not in IDs_assign) and (n_obs_trks[ID] == i_obs)
            ]
        )

        # case 5: false track predicts as unknown
        cases[5] = sum(
            [
                pred == 0
                for ID, pred in predictions.items()
                if (ID not in IDs_assign) and (n_obs_trks[ID] == i_obs)
            ]
        )

        # case 6: false track predicts as invalid
        cases[6] = sum(
            [
                pred == -1
                for ID, pred in predictions.items()
                if (ID not in IDs_assign) and (n_obs_trks[ID] == i_obs)
            ]
        )

        # add
        cases_all[i_obs] = cases

    n_outcomes = sum(v for cases in cases_all.values() for v in list(cases.values()))
    if n_outcomes != len(tracks) + len(assigns.unassigned_cols):
        raise RuntimeError

    metrics = {"cases_by_observer": cases_all}
    return metrics

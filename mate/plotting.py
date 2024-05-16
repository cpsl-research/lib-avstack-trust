import os

import matplotlib.pyplot as plt
import numpy as np
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign
from labellines import labelLine
from matplotlib.patches import Polygon
from scipy.stats import beta


marker_large = 10
marker_medium = 8
marker_small = 5
linewidth = 3
agent_colors = "rgbmc"
det_markers = [str(i + 1) for i in range(len(agent_colors))]


def set_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()


def _add_agents_fovs(ax, agents, fovs, swap_axes=False):
    idxs = [1, 0] if swap_axes else [0, 1]
    for i, color in zip(fovs, agent_colors):
        ax.plot(
            *agents[i][idxs],
            "*",
            markersize=marker_medium,
            color=color,
            label=f"Agent {i}",
        )
        try:
            # for FOV as a shape
            circle = plt.Circle(agents[i], fovs[i].radius, color=color, alpha=0.4)
            ax.add_patch(circle)
        except AttributeError:
            # for FOV as a hull
            polygon = Polygon(
                fovs[i][:, idxs], closed=True, facecolor=color, alpha=0.25
            )
            ax.add_patch(polygon)
    # if swap_axes:
    #     ax.invert_xaxis()
    # else:
    #     ax.invert_yaxis()
    return ax


def _add_objects(ax, objects, swap_axes=False):
    if objects:
        idxs = [1, 0] if swap_axes else [0, 1]
        for i, object in enumerate(objects):
            ax.plot(
                *object[idxs],
                "o",
                markersize=marker_medium,
                alpha=0.5,
                color="black",
                label="True Object" if i == 0 else "",
            )
    return ax


def _add_detections(ax, dets, swap_axes=False):
    idxs = [1, 0] if swap_axes else [0, 1]
    for i_agent, ds in dets.items():
        for j, det in enumerate(ds):
            try:
                center = det.x
            except AttributeError:
                center = det.box.t
            ax.plot(
                *center[idxs],
                marker=det_markers[i_agent],
                markersize=marker_large + 4,
                alpha=1,
                color=agent_colors[i_agent],
                label=f"Detection, Agent {i_agent}" if j == 0 else "",
            )
    return ax


def _add_clusters(ax, clusters, swap_axes=False):
    idxs = [1, 0] if swap_axes else [0, 1]
    for j, clust in enumerate(clusters):
        pos = clust.centroid().x[idxs]
        ax.plot(
            *pos,
            "x",
            markersize=marker_large,
            color="orange",
            label="Cluster" if j == 0 else "",
            alpha=0.8,
        )
        ax.text(pos[0] + 0.02, pos[1] + 0.02, j)
    return ax


def _add_tracks(ax, tracks, swap_axes=False):
    idxs = [1, 0] if swap_axes else [0, 1]
    for j, track in enumerate(tracks):
        pos = track.position.x[idxs]
        ax.plot(
            *pos,
            "x",
            markersize=marker_large,
            color="orange",
            label="Track" if j == 0 else "",
            alpha=0.8,
        )
        ax.text(pos[0] + 0.02, pos[1] + 0.02, track.ID)
    return ax


def plot_agents(agents, fovs, swap_axes=False, save=False, fig_dir="figures", **kwargs):
    fig, ax = plt.subplots()
    _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes, **kwargs)
    set_axes(ax)
    if save:
        plt.savefig(os.path.join(fig_dir, "agents.pdf"))
    plt.show()


def plot_agents_objects(
    agents, fovs, objects, fps, fns, swap_axes=False, save=False, fig_dir="figures"
):
    fig, ax = plt.subplots()
    idxs = [1, 0] if swap_axes else [0, 1]
    ax = _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes)
    ax = _add_objects(ax, objects, swap_axes=swap_axes)

    # plot false positives
    for i, fp in enumerate(fps):
        ax.plot(
            *fp[1][idxs],
            "x",
            markersize=marker_medium,
            color=agent_colors[fp[0]],
            label="FP" if i == 0 else "",
        )

    # plot false negatives
    for i, fn in enumerate(fns):
        ax.plot(
            *objects[fn[1]][idxs],
            "+",
            markersize=marker_large,
            color=agent_colors[fn[0]],
            label="FN" if i == 0 else "",
        )

    set_axes(ax)
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_truth.pdf"))
    plt.show()


def plot_agents_detections(
    agents, fovs, dets, objects=None, swap_axes=False, save=False, fig_dir="figures"
):
    fig, ax = plt.subplots()
    _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes)
    _add_objects(ax, objects, swap_axes=swap_axes)
    _add_detections(ax, dets, swap_axes=swap_axes)
    set_axes(ax)
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_detections.pdf"))
    plt.show()


def plot_agents_clusters(
    agents, fovs, clusters, objects=None, swap_axes=False, save=False, fig_dir="figures"
):
    fig, ax = plt.subplots()
    _add_agents_fovs(ax, agents, fovs)
    _add_objects(ax, objects, swap_axes=swap_axes)
    _add_clusters(ax, clusters, swap_axes=swap_axes)
    set_axes(ax)
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_clusters.pdf"))
    plt.show()


def plot_agents_tracks(
    agents, fovs, tracks, objects=None, swap_axes=False, save=False, fig_dir="figures"
):
    fig, ax = plt.subplots()
    _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes)
    _add_objects(ax, objects, swap_axes=swap_axes)
    _add_tracks(ax, tracks, swap_axes=swap_axes)
    set_axes(ax)
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_tracks.pdf"))
    plt.show()


def plot_trust(
    trust_estimator,
    objects=None,
    show_legend=False,
    use_labellines=False,
    save=False,
    fig_dir="figures",
):
    tracks_sorted = sorted(trust_estimator.tracks, key=lambda x: x.ID, reverse=False)

    # assign last tracks to truths, if possible
    if objects is not None:
        A = build_A_from_distance(tracks_sorted, objects)
        assigns = gnn_single_frame_assign(A, cost_threshold=0.1)
    else:
        assigns = None

    # plot all track trust distributions
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(0, 1.0, 10000)
    for i_track, track in enumerate(tracks_sorted):
        if assigns is not None:
            label = (
                f"{track.ID}: True Object"
                if assigns.has_assign(row=i_track)
                else f"{track.ID}: False Pos."
            )
        else:
            label = f"Track {track.ID}"
        linestyle = "-" if "True" in label else "--"
        y = beta.pdf(
            x,
            trust_estimator.track_trust[track.ID].alpha,
            trust_estimator.track_trust[track.ID].beta,
        )
        ax.plot(x, y, linewidth=linewidth, linestyle=linestyle, label=label)

    if use_labellines:
        lines = ax.get_lines()
        xs = np.linspace(0.3, 0.7, len(lines))
        for i, line in enumerate(lines):
            labelLine(
                line,
                xs[i],
                label=r"{}".format(line.get_label().split(":")[0]),
                ha="left",
                va="bottom",
                align=False,
                backgroundcolor="none",
            )

    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 10])
    if show_legend:
        ax.legend(loc="upper right")
    ax.set_xlabel("Track Trust Score")
    ax.set_ylabel("PDF")
    ax.set_yticklabels([])
    ax.set_title("Track Trusts")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_track_trusts.pdf"))
    plt.show()

    # plot all agent trust distributions
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.linspace(0, 1.0, 10000)
    for ID, color in zip(trust_estimator.agent_trust, agent_colors):
        y = beta.pdf(
            x,
            trust_estimator.agent_trust[ID].alpha,
            trust_estimator.agent_trust[ID].beta,
        )
        ax.plot(
            x,
            y,
            color=color,
            alpha=0.5,
            linewidth=linewidth,
            linestyle="--",
            label=f"Agent {ID}",
        )

    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 10])
    ax.legend(loc="upper right")
    ax.set_xlabel("Agent Trust Score")
    ax.set_ylabel("PDF")
    ax.set_yticklabels([])
    ax.set_title("Agent Trusts")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_agent_trusts.pdf"))
    plt.show()

    # plot track trusts in bar format
    fig, ax = plt.subplots(figsize=(5, 4))
    for i_track, track in enumerate(tracks_sorted):
        if assigns is not None:
            label = (
                f"{track.ID}: True Object"
                if assigns.has_assign(row=i_track)
                else f"{track.ID}: False Pos."
            )
        else:
            label = f"Track {track.ID}"
        ax.barh(
            y=i_track, width=trust_estimator.track_trust[track.ID].mean, label=label
        )
    ax.set_xlim([0, 1.0])
    ax.set_xlabel("Track Trust Score")
    ax.set_ylabel("PDF")
    ax.set_yticks(list(range(len(trust_estimator.tracks))))
    ax.set_yticklabels([f"Track {track.ID}" for track in tracks_sorted])
    ax.set_title("Track Trusts")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_track_trusts_bar.pdf"))
    plt.show()


def plot_metrics(df, save=False, fig_dir="figures"):
    # aggregate and normalize by number of occurrences
    df_agg = df.groupby("n_observers")[list(range(7))].sum().T.astype(float)
    df_agg[1:4] /= df_agg[1:4].sum().astype(float)
    df_agg[4:7] /= df_agg[4:7].sum().astype(float)

    # plot cases 1 - 6
    ax = df_agg[1:].plot.bar(figsize=(6, 3))
    xticklabels = [
        "TT\nTrusted",
        "TT\nUnknown",
        "TT\nDistrusted",
        "FT\nTrusted",
        "FT\nUnknown",
        "FT\nDistrusted",
    ]
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(["1 Observer", "2 Observers", "3 Observers"], loc="lower right")
    ax.tick_params(axis="x", which="major", labelsize=11)
    ax.axvline(x=2.5, color="k", linestyle="--", linewidth=linewidth, label="")

    # plt.xlabel('Case Index')
    plt.ylabel("Fraction of Cases")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(fig_dir, "experiment_metrics.pdf"))
    plt.show()

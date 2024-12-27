import os

import matplotlib.colors as mcolors
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
agent_colors = list(mcolors.XKCD_COLORS.keys())
track_colors = list(mcolors.XKCD_COLORS.keys())
det_markers = [str(i + 1) for i in range(len(agent_colors))]


def get_agent_color(ID_agent):
    return agent_colors[ID_agent % len(agent_colors)]


def get_track_color(ID_track):
    return track_colors[ID_track % len(track_colors)]


def set_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    plt.legend(loc="lower left")
    plt.axis("off")
    plt.tight_layout()


def get_quad_trust_axes(font_family: str = "serif"):
    # Initialize figure and axes and save to class
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for i, txt in enumerate(["Agent", "Track"]):
        # -- attributes for bar plot
        axs[i, 0].set_title(f"{txt} Trust Mean", family=font_family)
        axs[i, 0].set_xlim([0, 1])
        axs[i, 0].set_xlabel("Mean Trust Value", family=font_family)
        axs[i, 0].set_ylabel("Identifier", family=font_family)
        axs[i, 0].xaxis.grid()

        # -- attributes for distribution plot
        axs[i, 1].set_title(f"{txt} Trust Distributions", family=font_family)
        axs[i, 1].set_xlim([0, 1])
        axs[i, 1].set_ylim([0, 10])
        axs[i, 1].set_xlabel("Trust Value", family=font_family)
        axs[i, 1].set_ylabel("PDF", family=font_family)
        axs[i, 1].grid()
    return axs


def plot_trust_on_quad(axs, trust_agents, trust_tracks, font_family: str = "serif"):
    x_trust = np.linspace(0, 1, 1000)
    for i, trusts in enumerate([trust_agents, trust_tracks]):
        yticks = []
        ytick_labels = []
        for j, (j_ID, trust) in enumerate(trusts.items()):
            color = get_agent_color(j_ID) if i == 0 else get_track_color(j_ID)
            label = f"Agent {j_ID}" if i == 0 else f"Track {j_ID}"
            yticks.append(j)
            ytick_labels.append(label)

            # -- update bars
            y = j
            w = trust.mean
            height = 0.6
            axs[i, 0].barh(
                y,
                w,
                height=height,
                left=0.0,
                color=color,
                label=label,
            )

            # -- update distributions
            pdfs = trust.pdf(x_trust)
            axs[i, 1].plot(
                x_trust,
                pdfs,
                color=color,
                label=label,
            )

        # set the ticks for the identifiers
        axs[i, 0].set_yticks(yticks)
        axs[i, 0].set_yticklabels(ytick_labels, family=font_family)
    plt.tight_layout()


def _add_agents_fovs(ax, agents, fovs, swap_axes=False):
    idxs = [1, 0] if swap_axes else [0, 1]
    for ID_agent, fov in fovs.items():
        color = get_agent_color(ID_agent)
        ax.plot(
            *agents[ID_agent][idxs],
            "*",
            markersize=marker_medium,
            color=color,
            label=f"Agent {ID_agent}",
        )
        try:
            # for FOV as a shape
            circle = plt.Circle(agents[ID_agent], fov.radius, color=color, alpha=0.4)
            ax.add_patch(circle)
        except AttributeError:
            # for FOV as a hull
            polygon = Polygon(
                fov.boundary[:, idxs], closed=True, facecolor=color, alpha=0.25
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
    for ID_agent, ds in dets.items():
        color = get_agent_color(ID_agent)
        for j, det in enumerate(ds):
            try:
                center = det.x
            except AttributeError:
                center = det.box.t
            ax.plot(
                *center[idxs],
                marker=det_markers[ID_agent],
                markersize=marker_large + 4,
                alpha=1,
                color=color,
                label=f"Detection, Agent {ID_agent}" if j == 0 else "",
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
    for j, track in enumerate(tracks):
        try:
            # avstack tracks
            idxs = [1, 0] if swap_axes else [0, 1]
            pos = track.position.x[idxs]
        except AttributeError:
            # stonesoup tracks
            idxs = [2, 0] if swap_axes else [0, 2]
            pos = track.state_vector[idxs]
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


def _plot_preliminaries(figsize=(5, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _plot_post(
    title,
    show=True,
    save=False,
    fig_dir="figures",
    subfig_dir="",
    suffix="",
    extension="pdf",
):
    plt.tight_layout()
    if save:
        os.makedirs(os.path.join(fig_dir, subfig_dir), exist_ok=True)
        plt.savefig(os.path.join(fig_dir, subfig_dir, f"{title}{suffix}.{extension}"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_agents(agents, fovs, swap_axes=False, **kwds):
    fig, ax = _plot_preliminaries()
    _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes, **kwds)
    set_axes(ax)
    _plot_post(title="agents", **kwds)


def plot_agents_objects(agents, fovs, objects, fps, fns, swap_axes=False, **kwds):
    fig, ax = _plot_preliminaries()

    idxs = [1, 0] if swap_axes else [0, 1]
    ax = _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes)
    ax = _add_objects(ax, objects, swap_axes=swap_axes)

    # plot false positives
    for i, fp in enumerate(fps):
        ax.plot(
            *fp[1][idxs],
            "x",
            markersize=marker_medium,
            color=get_agent_color([fp[0]]),
            label="FP" if i == 0 else "",
        )

    # plot false negatives
    for i, fn in enumerate(fns):
        ax.plot(
            *objects[fn[1]][idxs],
            "+",
            markersize=marker_large,
            color=get_agent_color([fn[0]]),
            label="FN" if i == 0 else "",
        )
    set_axes(ax)
    _plot_post(title="agents-objects", **kwds)


def plot_agents_detections(
    agents,
    fovs,
    dets,
    objects=None,
    swap_axes=False,
    **kwds,
):
    fig, ax = _plot_preliminaries()
    _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes)
    _add_objects(ax, objects, swap_axes=swap_axes)
    _add_detections(ax, dets, swap_axes=swap_axes)
    set_axes(ax)
    _plot_post(title="experiment_detections", **kwds)


def plot_agents_clusters(
    agents,
    fovs,
    clusters,
    objects=None,
    swap_axes=False,
    **kwds,
):
    fig, ax = _plot_preliminaries()
    _add_agents_fovs(ax, agents, fovs)
    _add_objects(ax, objects, swap_axes=swap_axes)
    _add_clusters(ax, clusters, swap_axes=swap_axes)
    set_axes(ax)
    _plot_post(title="experiment_clusters", **kwds)


def plot_agents_tracks(
    agents,
    fovs,
    tracks,
    objects=None,
    swap_axes=False,
    **kwds,
):
    fig, ax = _plot_preliminaries()
    _add_agents_fovs(ax, agents, fovs, swap_axes=swap_axes)
    _add_objects(ax, objects, swap_axes=swap_axes)
    _add_tracks(ax, tracks, swap_axes=swap_axes)
    set_axes(ax)
    _plot_post(title="experiment_tracks", **kwds)


def plot_trust(
    tracks,
    track_trust,
    agent_trust,
    objects=None,
    show_legend=False,
    use_labellines=False,
    use_subfolders=False,
    **kwds,
):
    tracks_sorted = sorted(tracks, key=lambda x: x.ID, reverse=False)

    # assign last tracks to truths, if possible
    if objects is not None:
        A = build_A_from_distance(tracks_sorted, objects)
        assigns = gnn_single_frame_assign(A, cost_threshold=0.1)
    else:
        assigns = None

    # plot all track trust distributions
    fig, ax = _plot_preliminaries(figsize=(5, 4))
    x = np.linspace(0, 1.0, 10000)
    for i_track, track in enumerate(tracks_sorted):
        t_color = get_track_color(track.ID)
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
            track_trust[track.ID].alpha,
            track_trust[track.ID].beta,
        )
        ax.plot(
            x, y, color=t_color, linewidth=linewidth, linestyle=linestyle, label=label
        )

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
    title = "experiment_track_trusts"
    subfig_dir = "track_trust_dist" if use_subfolders else ""
    _plot_post(title=title, subfig_dir=subfig_dir, **kwds)

    # plot all agent trust distributions
    fig, ax = _plot_preliminaries(figsize=(5, 4))
    x = np.linspace(0, 1.0, 10000)
    for ID_agent in agent_trust:
        a_color = get_agent_color(ID_agent)
        y = beta.pdf(
            x,
            agent_trust[ID_agent].alpha,
            agent_trust[ID_agent].beta,
        )
        ax.plot(
            x,
            y,
            color=a_color,
            alpha=0.5,
            linewidth=linewidth,
            linestyle="--",
            label=f"Agent {ID_agent}",
        )

    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 10])
    ax.legend(loc="upper right")
    ax.set_xlabel("Agent Trust Score")
    ax.set_ylabel("PDF")
    ax.set_yticklabels([])
    ax.set_title("Agent Trusts")
    title = "experiment_agent_trusts"
    subfig_dir = "agent_trust_dist" if use_subfolders else ""
    _plot_post(title=title, subfig_dir=subfig_dir, **kwds)

    # plot track trusts in bar format
    fig, ax = _plot_preliminaries(figsize=(5, 4))
    for i_track, track in enumerate(tracks_sorted):
        t_color = get_track_color(track.ID)
        if assigns is not None:
            label = (
                f"{track.ID}: True Object"
                if assigns.has_assign(row=i_track)
                else f"{track.ID}: False Pos."
            )
        else:
            label = f"Track {track.ID}"
        a = track_trust[track.ID].alpha
        b = track_trust[track.ID].beta
        mean = track_trust[track.ID].mean
        x_below = beta.ppf(0.025, a, b)
        x_above = beta.ppf(0.975, a, b)
        ax.barh(
            y=i_track,
            width=mean,
            xerr=np.array([[mean - x_below], [x_above - mean]]),
            capsize=4,
            label=label,
            color=t_color,
        )
    ax.set_xlim([0, 1.0])
    ax.set_xlabel("Track Trust Score")
    ax.set_ylabel("PDF")
    ax.set_yticks(list(range(len(tracks))))
    ax.set_yticklabels([f"Track {track.ID}" for track in tracks_sorted])
    ax.set_title("Track Trusts")
    title = "experiment_track_trusts_bar"
    subfig_dir = "track_trust_bar" if use_subfolders else ""
    _plot_post(title=title, subfig_dir=subfig_dir, **kwds)

    # plot agent trusts in bar format
    fig, ax = _plot_preliminaries(figsize=(5, 4))
    for i_agent, ID_agent in enumerate(agent_trust):
        label = f"Agent {ID_agent}"
        color = get_agent_color(ID_agent)
        a = agent_trust[ID_agent].alpha
        b = agent_trust[ID_agent].beta
        mean = agent_trust[ID_agent].mean
        x_below = beta.ppf(0.025, a, b)
        x_above = beta.ppf(0.975, a, b)
        ax.barh(
            y=i_agent,
            width=agent_trust[ID_agent].mean,
            xerr=np.array([[mean - x_below], [x_above - mean]]),
            capsize=7,
            label=label,
            color=color,
        )
    ax.set_xlim([0, 1.0])
    ax.set_xlabel("Agent Trust Score")
    ax.set_ylabel("PDF")
    ax.set_yticks(list(range(len(agent_trust))))
    ax.set_yticklabels([f"Agent {ID_agent}" for ID_agent in agent_trust])
    ax.set_title("Agent Trusts")
    title = "experiment_agent_trusts_bar"
    subfig_dir = "agent_trust_bar" if use_subfolders else ""
    _plot_post(title=title, subfig_dir=subfig_dir, **kwds)


def plot_metrics(df, **kwds):
    fig, ax = _plot_preliminaries()

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
    _plot_post(title="experiment_metrics", **kwds)

import matplotlib.pyplot as plt


def get_metrics_axes():
    # Initialize figure and axes and save to class
    fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey="row")

    # left column is for baseline, right for attacked
    # top row for OSPA, middle for F1-score, bottom trust metric
    loop_tuples = [
        ("{}", [0, 1], 1),
        ("OSPA Metric", [0, 10], 0),
        # ("Trust Metric", [0, 1], 1),
    ]
    for i_row, (ylabel, ylim, goal) in enumerate(loop_tuples):
        for j_col, case in enumerate(["Benign", "Attacked"]):
            ax = axs[i_row, j_col]
            title_txt = f"{case}, {ylabel} (Goal: {goal})"
            ax.set_title(title_txt)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)
            ax.grid()
    return fig, axs


def get_trust_metrics_axes():
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)
    ylim = [0, 1]
    goal = 1
    for j_col, case in enumerate(["Benign", "Attacked"]):
        ax = axs[j_col]
        title_txt = f"{case}, (Goal: {goal})"
        ax.set_title(title_txt)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trust Metric")
        ax.set_ylim(ylim)
        ax.grid()
    return fig, axs

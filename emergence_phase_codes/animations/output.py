# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,protected-access,duplicate-code
"""Animation classes for output through time animations."""

import jax.numpy as jnp


class OutputAnimator:
    """
    Animate RNN output through time.
    Assumes model_output is a dictionary keyed by trajectory index with values of shape (n_time,)
    (or (n_time, 1), which will be squeezed).
    """

    def __init__(
        self,
        ax,
        model_output,
        trajectory_colors,
        title,
        time_end=1,
        stimulus_colors=None,
    ):
        self.ax = ax
        self.title = title
        self.trajectories = model_output
        self.trajectory_colors = trajectory_colors
        self.stimulus_colors = stimulus_colors

        # Determine axis limits from the concatenated trajectories.
        all_data = jnp.concatenate(list(model_output.values()), axis=0)
        y_min, y_max = float(jnp.min(all_data)), float(jnp.max(all_data))
        margin_y = (y_max - y_min) * 0.1 if y_max != y_min else 1
        self.ylim = (y_min - margin_y, y_max + margin_y)

        # Set up the axis.
        self.ax.set_xlim((-0.05, time_end + 0.05))
        self.ax.set_ylim(self.ylim)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Model output")
        self.ax.axhline(y=0, ls="--", lw=2, alpha=0.5, color="tab:grey")

        # Number of time points (assumed same for all trajectories).
        self.n_time = list(self.trajectories.values())[0].shape[0]
        self.time = jnp.linspace(0, time_end, self.n_time)

        # Create dictionaries for line segments and scatter markers.
        self.line_dict = {}  # key: trajectory index, value: list of line objects
        self.scatter_dict = {}  # key: trajectory index, value: scatter object

        # For convenience, let trajectory_indices be all keys in model_output.
        self.trajectory_indices = list(self.trajectories.keys())

        # Create scatter markers and empty line objects.
        for idx in self.trajectory_indices:
            self.line_dict[idx] = []
            # Create a scatter marker at time 0.
            init_pt = jnp.squeeze(self.trajectories[idx][0])
            scatter = self.ax.scatter(
                0,
                init_pt,
                color=self.trajectory_colors[idx][-1],
                marker="D",
                s=150,
                edgecolors="black",
                zorder=3,
            )
            self.scatter_dict[idx] = scatter

        # Create one line per segment (between time t and t+1) for each trajectory.
        for _ in range(self.n_time - 1):
            for idx in self.trajectory_indices:
                (line,) = self.ax.plot([], [], linewidth=2)
                self.line_dict[idx].append(line)

    def color_integer_bars(self, integer_array):
        """
        Plots color bars at time steps where integer stimuli were presented.

        Args:
            integer_array (list): List of decoded integers for each time step.
        """
        if self.stimulus_colors is None:
            raise ValueError(
                "Stimulus colors must be provided when using color_integer_bars."
            )

        time_steps = self.n_time
        skip = False
        for t, integer in enumerate(integer_array):
            if skip:
                skip = False
                continue
            if integer is not None:
                self.ax.axvspan(
                    self.time[t - 1] / self.time[-1],  # Normalize time
                    self.time[min(t + 1, time_steps - 1)] / self.time[-1],
                    facecolor=self.stimulus_colors[integer],
                    edgecolor="black",
                    alpha=0.5,
                )
                skip = True

    def update(self, frame):
        """
        Update the animation to the given frame.

        Parameters:
          frame: current frame index (0-indexed)

        Returns a list of updated artists for blitting.
        """
        frame = int(jnp.clip(frame, a_max=self.n_time - 1).item())
        time_range = self.time[frame - 1 : frame + 1] if frame > 0 else self.time[0:2]
        updated_artists = []
        for idx in self.trajectory_indices:
            traj = jnp.squeeze(self.trajectories[idx])  # shape (n_time,)
            colors = self.trajectory_colors[idx]

            if frame > 0 and (frame - 1) < len(self.line_dict[idx]):
                segment = traj[frame - 1 : frame + 1]
                self.line_dict[idx][frame - 1].set_data(time_range, segment)
                self.line_dict[idx][frame - 1].set_color(colors[frame])
                updated_artists.append(self.line_dict[idx][frame - 1])

            pt = traj[frame]
            t = self.time[frame]
            self.scatter_dict[idx].set_offsets([t, pt])
            updated_artists.append(self.scatter_dict[idx])
        return updated_artists

    def figure(self):
        """Invoke update over all frames to produce the final figure."""
        for i in range(self.n_time):
            _ = self.update(i)

    def add_stimulus_legend(self, legend_loc="lower right"):
        """
        Adds a legend to the plot for stimulus colors.

        Args:
            legend_loc (str): Location of the legend (default: "lower right").
        """
        if self.stimulus_colors is None:
            raise ValueError(
                "Stimulus colors must be provided when using add_stimulus_legend."
            )

        for label in self.stimulus_colors:
            self.ax.scatter(
                [], [], color=self.stimulus_colors[label], label=label, s=100, alpha=0.8
            )

        self.ax.legend(loc=legend_loc, fontsize=12, frameon=True)

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,protected-access
"""Animation classes for phase difference animations."""

import jax.numpy as jnp


class PhaseTrajectoryAnimator:
    """
    Animate phase differences of example trajectories relative to a null (resting) trajectory.

    For each example trajectory (selected via trajectory_indices from model_behavior),
    the phase difference between its phase and the null phase is computed at every time step.
    This phase difference is then converted into a phasor and animated on a unit circle.

    Parameters:
      ax: matplotlib axis on which to plot (should be 2D).
      model_behavior: dictionary containing at least:
          - "rates_pc": an array of shape (n_trials, n_time, n_components)
      trajectory_indices: list of integers indicating which trajectories (trials) to animate.
      null_trajectory: a jnp.array of shape (n_time, 2) representing the resting (null) trajectory.
      title: Title for the axis.
    """

    def __init__(
        self,
        ax,
        model_behavior,
        trajectory_indices,
        trajectory_colors,
        null_trajectory,
        title,
    ):
        self.ax = ax
        self.title = title
        self.trajectory_indices = trajectory_indices
        self.trajectory_colors = trajectory_colors

        # Extract the selected trajectories
        rates_pc = model_behavior["rates_pc"]
        self.trajectories = rates_pc[self.trajectory_indices, :, :][
            :, :, [0, 1]
        ]  # Shape (n_selected, n_time, 2)

        # Compute phasors once and store them in an array
        self.phasors = self.compute_phasors(
            self.trajectories, null_trajectory
        )  # Shape (n_selected, n_time, 2)

        # Store the null trajectory (shape: (n_time, 2))
        self.null_trajectory = null_trajectory  # already 2D

        # Determine number of time steps (assumed the same for all trajectories).
        self.n_time = self.phasors.shape[1]

        # Set up the axis for a unit circle.
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_title(title)
        self.ax.set_xlabel("Cosine of Phase Difference")
        self.ax.set_ylabel("Sine of Phase Difference")

        # Draw a background unit circle.
        theta = jnp.linspace(0, 2 * jnp.pi, 200)
        self.ax.plot(
            jnp.cos(theta),
            jnp.sin(theta),
            color="grey",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        # Create dictionaries for line segments (to show the trajectory path)
        # and scatter markers (to show current position).
        self.line_dict = (
            {}
        )  # key: trajectory index, value: list of line objects (one per segment)
        self.scatter_dict = {}  # key: trajectory index, value: scatter object
        for idx in self.trajectory_indices:
            self.line_dict[idx] = []
            # Create a scatter marker at time 0.
            init_pt = self.phasors[idx, 0, :]  # shape (2,)
            scatter = self.ax.scatter(
                init_pt[0],
                init_pt[1],
                color=self.trajectory_colors[idx][-1],
                marker="o",
                s=150,
                edgecolors="black",
                zorder=3,
            )
            self.scatter_dict[idx] = scatter

        # Create one line segment per consecutive time step for each trajectory.
        for _ in range(self.n_time - 1):
            for idx in self.trajectory_indices:
                (line,) = self.ax.plot([], [], linewidth=2)
                self.line_dict[idx].append(line)

    def compute_phasors(self, trajectories, null_trajectory):
        """
        Compute phase differences and convert them to phasor coordinates.

        Args:
            trajectories (jnp.ndarray): Shape (n_trials, n_time, 2), trajectory in PCA space.
            null_trajectory (jnp.ndarray): Shape (n_time, 2), resting trajectory.

        Returns:
            jnp.ndarray: Shape (n_trials, n_time, 2), phasor coordinates.
        """
        phase_example = jnp.arctan2(trajectories[:, :, 1], trajectories[:, :, 0])
        phase_null = jnp.arctan2(null_trajectory[:, 1], null_trajectory[:, 0])
        phase_diff = phase_example - phase_null
        phasor_x = jnp.cos(phase_diff)
        phasor_y = jnp.sin(phase_diff)
        return jnp.stack([phasor_x, phasor_y], axis=2)  # Shape (n_trials, n_time, 2)

    def update(self, frame_pre_clip):
        """
        Update the animation to the given frame.

        Parameters:
          frame_pre_clip: current frame index (0-indexed, can be non-integer; will be clipped)

        Returns:
          A list of updated matplotlib artist objects for blitting.
        """
        frame = int(jnp.clip(frame_pre_clip, a_max=self.n_time - 1).item())
        updated_artists = []
        phasor = self.phasors  # Shape (n_selected, n_time, 2)
        if frame > 0 and (frame - 1) < self.phasors.shape[1]:
            segment = phasor[:, frame - 1 : frame + 1, :]
            for idx, line in enumerate(self.line_dict.values()):
                line[frame - 1].set_data(segment[idx, :, 0], segment[idx, :, 1])
                line[frame - 1].set_color(
                    self.trajectory_colors[self.trajectory_indices[idx]][frame]
                )
                updated_artists.append(line[frame - 1])

        # Update scatter points
        scatter_pts = phasor[:, frame, :]
        for idx, scatter in enumerate(self.scatter_dict.values()):
            scatter.set_offsets([scatter_pts[idx, 0], scatter_pts[idx, 1]])
            updated_artists.append(scatter)
        return updated_artists

    def figure(self):
        """Invokes update over all frames to produce the final figure."""
        for i in range(self.n_time):
            _ = self.update(i)


class PhasePopulationAnimator:
    """
    Animate phase differences for a population of trials on a unit circle.

    For each trial in the population (all trials from model_behavior are used),
    the phase difference between its phase and a supplied null trajectory is computed
    at every time step. The phase differences are then converted into phasor coordinates
    (cosine, sine) and a scatter plot is animated over time.

    Parameters:
      ax: matplotlib axis on which to plot (should be 2D).
      model_behavior: dictionary containing at least:
          - "rates_pc": an array of shape (n_trials, n_time, n_components)
          - "outputs": an array of shape (n_trials, n_time, output_dim)
      classification_color_map: dict mapping classification values (e.g. -1, 1) to colors.
      null_trajectory: a jnp.array of shape (n_time, 2) representing the resting (null) trajectory.
      title: Title for the axis.
      highlight_indices: Optional list of trial indices to mark specially.
    """

    def __init__(
        self,
        ax,
        model_behavior,
        classification_color_map,
        null_trajectory,
        title,
        highlight_indices=None,
    ):
        self.ax = ax
        self.title = title
        self.classification_color_map = classification_color_map

        # Extract the population PCA data.
        # Assume rates_pc has shape (n_trials, n_time, n_components).
        # For phase computation we assume that the relevant 2D data are given by the first two PCs.
        rates_pc = model_behavior["rates_pc"]
        self.population = rates_pc[:, :, [0, 1]]  # shape: (n_trials, n_time, 2)

        # Determine number of trials and time steps.
        self.n_trials, self.n_time, _ = self.population.shape

        # Store the null trajectory (should be shape (n_time, 2)).
        self.null_trajectory = null_trajectory

        # Extract final outputs from ALL output channels
        final_outputs = jnp.sign(
            model_behavior["outputs"][:, -1, :]
        )  # (n_trials, n_outputs)

        # Convert outputs to tuples (e.g., (-1,1), (1,-1))
        self.classification_keys = [
            tuple(final_outputs[i].tolist()) for i in range(final_outputs.shape[0])
        ]

        # Assign colors using tuple keys
        self.trial_colors = [
            classification_color_map.get(
                self.classification_keys[i], "tab:gray"
            )  # Default gray if missing
            for i in range(len(self.classification_keys))
        ]

        # Compute phasors once and store them in an array
        self.phasors = self.compute_phasors(
            self.population, null_trajectory
        )  # Shape (n_trials, n_time, 2)

        # Set up the axis for a unit circle.
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_title(title)
        self.ax.set_xlabel("Cosine of Phase Difference")
        self.ax.set_ylabel("Sine of Phase Difference")
        # Draw the background unit circle.
        theta = jnp.linspace(0, 2 * jnp.pi, 200)
        self.ax.plot(
            jnp.cos(theta),
            jnp.sin(theta),
            color="grey",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        # Create scatter markers for the full population at time 0.
        init_positions = jnp.stack(
            [self.phasors[idx][0, :] for idx in range(self.n_trials)], axis=0
        )
        self.scatter = self.ax.scatter(
            init_positions[:, 0],
            init_positions[:, 1],
            c=self.trial_colors,
        )

        # Optionally, create a separate scatter for highlighted trials.
        if highlight_indices is not None:
            self.highlight_indices = jnp.array(highlight_indices)
            highlight_positions = jnp.stack(
                [self.phasors[idx.item()][0, :] for idx in self.highlight_indices],
                axis=0,
            )
            highlight_classifications = [
                self.classification_keys[idx] for idx in highlight_indices
            ]
            highlight_colors = [
                classification_color_map.get(c, "tab:gray")
                for c in highlight_classifications
            ]
            self.highlight_scatter = self.ax.scatter(
                highlight_positions[:, 0],
                highlight_positions[:, 1],
                color=highlight_colors,
                marker="o",
                s=150,
                edgecolors="black",
                zorder=3,
            )
        else:
            self.highlight_indices = None
            self.highlight_scatter = None

    def compute_phasors(self, trajectories, null_trajectory):
        """
        Compute phase differences and convert them to phasor coordinates.

        Args:
            trajectories (jnp.ndarray): Shape (n_trials, n_time, 2), trajectory in PCA space.
            null_trajectory (jnp.ndarray): Shape (n_time, 2), resting trajectory.

        Returns:
            jnp.ndarray: Shape (n_trials, n_time, 2), phasor coordinates.
        """
        phase_trial = jnp.arctan2(trajectories[:, :, 1], trajectories[:, :, 0])
        phase_null = jnp.arctan2(null_trajectory[:, 1], null_trajectory[:, 0])
        phase_diff = phase_trial - phase_null
        phasor_x = jnp.cos(phase_diff)
        phasor_y = jnp.sin(phase_diff)
        return jnp.stack([phasor_x, phasor_y], axis=2)  # Shape (n_trials, n_time, 2)

    def update(self, frame_pre_clip):
        """
        Update the phase population scatter plot to the given frame.

        Parameters:
          frame_pre_clip: current frame index (0-indexed, can be non-integer; will be clipped)

        Returns:
          A list of updated matplotlib artist objects for blitting.
        """
        frame = int(jnp.clip(frame_pre_clip, a_max=self.n_time - 1).item())
        updated_artists = []
        # Update the scatter positions for the full population.
        all_positions = jnp.stack(
            [self.phasors[idx][frame, :] for idx in range(self.n_trials)], axis=0
        )
        self.scatter.set_offsets(all_positions)
        updated_artists.append(self.scatter)

        # Update the highlighted trials scatter, if any.
        if self.highlight_scatter is not None:
            h_positions = jnp.stack(
                [self.phasors[idx.item()][frame, :] for idx in self.highlight_indices],
                axis=0,
            )
            self.highlight_scatter.set_offsets(h_positions)
            updated_artists.append(self.highlight_scatter)

        return updated_artists

    def figure(self):
        """Invokes update over all frames to produce the final figure."""
        for i in range(self.n_time):
            _ = self.update(i)

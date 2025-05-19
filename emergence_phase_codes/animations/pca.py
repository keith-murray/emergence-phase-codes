# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,protected-access,duplicate-code
"""Animation classes for PCA Trajectory and Population animations."""

import jax.numpy as jnp


class PCATrajectoryAnimator:
    """
    Animate PCA trajectories in 2D or 3D.
    """

    # pylint: disable=too-many-branches,too-many-statements
    def __init__(
        self,
        ax,
        model_behavior,
        pc_x,
        pc_y,
        trajectory_indices,
        trajectory_colors,
        title,
        pc_z=None,
        stimulus_colors=None,
    ):
        """
        Parameters:
          ax: matplotlib axis to plot on. If pc_z is provided, ax should be a 3D axis.
          model_behavior: dictionary from compute_pca containing a key (e.g., "rates_pc")
                          whose value is an array of shape (n_trials, n_time, n_components).
          pc_x: int, principal component for the x-axis (PC number, starting at 1).
          pc_y: int, principal component for the y-axis.
          pc_z: int or None, if provided, principal component for the z-axis.
          trajectory_indices: list of integers corresponding to which trajectories to animate.
          trajectory_colors: dictionary of color arrays (each a sequence of colors for time points)
                             keyed by trajectory index.
          title: Title for the axis.
        stimulus_colors: dictionary mapping integers to colors.
        """
        self.ax = ax
        self.title = title
        self.pc_x = pc_x
        self.pc_y = pc_y
        self.pc_z = pc_z
        self.trajectory_indices = trajectory_indices
        self.trajectory_colors = trajectory_colors
        self.stimulus_colors = stimulus_colors

        # Extract the PCA trajectories for the chosen indices.
        # Assume model_behavior has key "rates_pc" with shape (n_trials, n_time, n_components)
        all_rates = model_behavior["rates_pc"]
        self.trajectories = {}
        for idx in trajectory_indices:
            if self.pc_z is None:
                # For 2D: extract columns [pc_x-1, pc_y-1]
                traj = all_rates[idx, :, :][:, [pc_x - 1, pc_y - 1]]
            else:
                # For 3D: extract columns [pc_x-1, pc_y-1, pc_z-1]
                traj = all_rates[idx, :, :][:, [pc_x - 1, pc_y - 1, pc_z - 1]]
            self.trajectories[idx] = traj

        # Determine axis limits from the concatenated trajectories.
        all_data = jnp.concatenate(list(self.trajectories.values()), axis=0)
        dim = all_data.shape[1]
        x_min, x_max = float(jnp.min(all_data[:, 0])), float(jnp.max(all_data[:, 0]))
        y_min, y_max = float(jnp.min(all_data[:, 1])), float(jnp.max(all_data[:, 1]))
        if dim == 3:
            z_min, z_max = float(jnp.min(all_data[:, 2])), float(
                jnp.max(all_data[:, 2])
            )
        else:
            z_min, z_max = None, None

        # Add margin
        margin_x = (x_max - x_min) * 0.1 if x_max != x_min else 1
        margin_y = (y_max - y_min) * 0.1 if y_max != y_min else 1
        self.xlim = (x_min - margin_x, x_max + margin_x)
        self.ylim = (y_min - margin_y, y_max + margin_y)
        if dim == 3:
            margin_z = (z_max - z_min) * 0.1 if z_max != z_min else 1
            self.zlim = (z_min - margin_z, z_max + margin_z)

        # Set up the axis.
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        if self.pc_z is not None:
            self.ax.set_zlim(self.zlim)
        self.ax.set_title(title)
        self.ax.set_xlabel(f"Principal Component {pc_x}")
        self.ax.set_ylabel(f"Principal Component {pc_y}")
        if self.pc_z is not None:
            self.ax.set_zlabel(f"Principal Component {pc_z}")

        # Create dictionaries for line segments and scatter markers.
        self.line_dict = {}  # key: trajectory index, value: list of line objects
        self.scatter_dict = {}  # key: trajectory index, value: scatter object

        # Number of time points (assumed same for all trajectories).
        self.n_time = list(self.trajectories.values())[0].shape[0]

        # Create scatter markers and empty line objects.
        for idx in trajectory_indices:
            self.line_dict[idx] = []
            # Create a scatter marker at time 0.
            init_pt = self.trajectories[idx][0, :]
            if self.pc_z is None:
                scatter = self.ax.scatter(
                    init_pt[0],
                    init_pt[1],
                    color=self.trajectory_colors[idx][-1],
                    marker="D",
                    s=150,
                    edgecolors="black",
                    zorder=3,
                )
            else:
                scatter = self.ax.scatter(
                    init_pt[0],
                    init_pt[1],
                    init_pt[2],
                    color=self.trajectory_colors[idx][-1],
                    marker="D",
                    s=150,
                    edgecolors="black",
                    zorder=3,
                    depthshade=False,
                )
            self.scatter_dict[idx] = scatter

        # Create one line per segment (between time t and t+1) for each trajectory.
        for _ in range(self.n_time - 1):
            for idx in trajectory_indices:
                if self.pc_z is None:
                    (line,) = self.ax.plot([], [], linewidth=2)
                else:
                    # For 3D, use plot3D (or ax.plot with 3D axis)
                    (line,) = self.ax.plot3D([], [], [], linewidth=2)
                self.line_dict[idx].append(line)

    def update(self, frame):
        """
        Update the animation to the given frame.

        Parameters:
          frame: current frame index (0-indexed)

        Returns a list of updated artists for blitting.
        """
        # Clip frame to the final frame.
        frame = int(jnp.clip(frame, a_max=self.n_time - 1).item())
        updated_artists = []
        for idx in self.trajectory_indices:
            traj = self.trajectories[idx]  # shape (n_time, dim)
            colors = self.trajectory_colors[idx]
            # If frame > 0, update the line segment from frame-1 to frame.
            if frame > 0 and (frame - 1) < len(self.line_dict[idx]):
                # Extract x, y (and z if 3D) for the segment.
                segment = traj[frame - 1 : frame + 1, :]
                if self.pc_z is None:
                    self.line_dict[idx][frame - 1].set_data(
                        segment[:, 0], segment[:, 1]
                    )
                else:
                    # For 3D, use set_data_3d.
                    self.line_dict[idx][frame - 1].set_data_3d(
                        segment[:, 0], segment[:, 1], segment[:, 2]
                    )
                self.line_dict[idx][frame - 1].set_color(colors[frame])
                updated_artists.append(self.line_dict[idx][frame - 1])
            # Update the endpoint scatter.
            pt = traj[frame, :]
            if self.pc_z is None:
                self.scatter_dict[idx].set_offsets([pt[0], pt[1]])
            else:
                # For 3D scatter, update the _offsets3d attribute.
                self.scatter_dict[idx]._offsets3d = ([pt[0]], [pt[1]], [pt[2]])
            updated_artists.append(self.scatter_dict[idx])
        return updated_artists

    def color_integer_points(
        self,
        indx,
        integer_array,
    ):
        """
        Adds stimulus markers to PCA trajectories.

        Args:
            indx (int): Index of the trajectory in `model_behavior["rates_pc"]`.
            integer_array (list): List of decoded integers for each time step.
        """
        skip = False
        for time_step, stim in enumerate(integer_array):
            if skip:
                skip = False
                continue
            if stim is not None:
                self.ax.scatter(
                    self.trajectories[indx][time_step - 1, 0],
                    self.trajectories[indx][time_step - 1, 1],
                    color=self.stimulus_colors[stim],
                    marker="o",
                    s=200,
                    zorder=1,
                    alpha=0.5,
                )
                skip = True

    def figure(self):
        """Invokes update over all frames to produce the final figure."""
        for i in range(self.n_time):
            _ = self.update(i)


class PCAPopulationAnimator:
    """
    Animate a PCA population scatter plot in 2D or 3D.
    """

    # pylint: disable=too-many-branches,too-many-statements
    def __init__(
        self,
        ax,
        model_behavior,
        pc_x,
        pc_y,
        classification_color_map,
        title,
        highlight_indices=None,
        pc_z=None,
        example_trajectory=None,
        final_output_index=None,
    ):
        """
        Parameters:
        ax: matplotlib axis to plot on. If pc_z is provided, ax should be a 3D axis.
        model_behavior: dictionary from compute_pca containing at least:
            - "rates_pc": an array of shape (n_trials, n_time, n_components)
            - "outputs": an array of shape (n_trials, n_time, output_dim)
        pc_x: int, principal component for the x-axis (PC number, starting at 1).
        pc_y: int, principal component for the y-axis.
        pc_z: int or None, if provided, principal component for the z-axis.
        classification_color_map: dict mapping classification values (e.g. -1, 1) to colors.
        title: Title for the axis.
        highlight_indices: Optional list of trial indices to mark specially.
        example_trajectory: Optional trajectory (jnp.array) to outline dynamics.
        """
        self.ax = ax
        self.title = title
        self.pc_x = pc_x
        self.pc_y = pc_y
        self.pc_z = pc_z
        self.classification_color_map = classification_color_map

        # Extract population PCA data.
        # rates_pc has shape (n_trials, n_time, n_components)
        rates_pc = model_behavior["rates_pc"]
        if self.pc_z is None:
            self.population = rates_pc[
                :, :, [pc_x - 1, pc_y - 1]
            ]  # shape (n_trials, n_time, 2)
        else:
            self.population = rates_pc[
                :, :, [pc_x - 1, pc_y - 1, pc_z - 1]
            ]  # shape (n_trials, n_time, 3)

        # Determine number of trials and time steps.
        self.n_trials, self.n_time, self.dim = self.population.shape

        # Compute axis limits from all data.
        all_data = self.population.reshape(-1, self.dim)
        x_min, x_max = float(jnp.min(all_data[:, 0])), float(jnp.max(all_data[:, 0]))
        y_min, y_max = float(jnp.min(all_data[:, 1])), float(jnp.max(all_data[:, 1]))
        margin_x = (x_max - x_min) * 0.1 if x_max != x_min else 1
        margin_y = (y_max - y_min) * 0.1 if y_max != y_min else 1
        self.xlim = (x_min - margin_x, x_max + margin_x)
        self.ylim = (y_min - margin_y, y_max + margin_y)
        if self.dim == 3:
            z_min, z_max = float(jnp.min(all_data[:, 2])), float(
                jnp.max(all_data[:, 2])
            )
            margin_z = (z_max - z_min) * 0.1 if z_max != z_min else 1
            self.zlim = (z_min - margin_z, z_max + margin_z)

        # Set up the axis.
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        if self.dim == 3:
            self.ax.set_zlim(self.zlim)
        self.ax.set_title(title)
        self.ax.set_xlabel(f"Principal Component {pc_x}")
        self.ax.set_ylabel(f"Principal Component {pc_y}")
        if self.dim == 3:
            self.ax.set_zlabel(f"Principal Component {pc_z}")

        # Extract final outputs from ALL output channels
        if final_output_index is None:
            final_outputs = jnp.sign(
                model_behavior["outputs"][:, -1, :]
            )  # (n_trials, n_outputs)
        else:
            final_outputs = jnp.sign(
                model_behavior["outputs"][:, -1, final_output_index]
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

        # Create scatter for all trials at time 0.
        init_positions = self.population[:, 0, :]  # shape (n_trials, dim)
        if self.dim == 2:
            self.scatter = self.ax.scatter(
                init_positions[:, 0],
                init_positions[:, 1],
                c=self.trial_colors,
                zorder=2,
            )
        else:
            self.scatter = self.ax.scatter(
                init_positions[:, 0],
                init_positions[:, 1],
                init_positions[:, 2],
                c=self.trial_colors,
                zorder=2,
            )

        # Optionally, create a separate scatter for highlighted trials.
        if highlight_indices is not None:
            self.highlight_indices = jnp.array(highlight_indices)
            highlight_positions = self.population[self.highlight_indices, :, :][:, 0, :]
            highlight_classifications = [
                self.classification_keys[idx] for idx in highlight_indices
            ]
            highlight_colors = [
                classification_color_map.get(c, "tab:gray")
                for c in highlight_classifications
            ]
            if self.dim == 2:
                self.highlight_scatter = self.ax.scatter(
                    highlight_positions[:, 0],
                    highlight_positions[:, 1],
                    color=highlight_colors,
                    marker="D",
                    s=150,
                    edgecolors="black",
                    zorder=3,
                )
            else:
                self.highlight_scatter = self.ax.scatter(
                    highlight_positions[:, 0],
                    highlight_positions[:, 1],
                    highlight_positions[:, 2],
                    color=highlight_colors,
                    marker="D",
                    s=150,
                    edgecolors="black",
                    zorder=3,
                    depthshade=False,
                )
        else:
            self.highlight_scatter = None

        if example_trajectory is not None:
            assert (
                example_trajectory.shape == model_behavior["rates_pc"][0, :, :].shape
            ), "Unexpected example_trajectory shape"
            if self.dim == 2:
                self.example_trajectory = example_trajectory[
                    :, [pc_x - 1, pc_y - 1]
                ]  # shape (n_time, 2)
                (line,) = self.ax.plot(
                    [], [], color="tab:gray", zorder=1.5, linewidth=2
                )
            else:
                self.example_trajectory = example_trajectory[
                    :, [pc_x - 1, pc_y - 1, pc_z - 1]
                ]  # shape (n_time, 3)
                (line,) = self.ax.plot3D(
                    [], [], color="tab:gray", zorder=1.5, linewidth=2
                )
            self.example_line = line
        else:
            self.example_trajectory = None

    def update(self, frame_pre_clip):
        """
        Update the population scatter plot to the given frame.

        Parameters:
          frame: current frame index (0-indexed)

        Returns a list of updated artists for blitting.
        """
        frame = int(jnp.clip(frame_pre_clip, a_max=self.n_time - 1).item())
        positions = self.population[:, frame, :]  # shape (n_trials, dim)
        if self.dim == 2:
            self.scatter.set_offsets(positions)
        else:
            # For 3D scatter, update the _offsets3d attribute.
            self.scatter._offsets3d = (
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
            )
        updated_artists = [self.scatter]

        if self.highlight_scatter is not None:
            hpos = self.population[self.highlight_indices, :, :][:, frame, :]
            if self.dim == 2:
                self.highlight_scatter.set_offsets(hpos)
            else:
                self.highlight_scatter._offsets3d = (hpos[:, 0], hpos[:, 1], hpos[:, 2])
            updated_artists.append(self.highlight_scatter)

        if self.example_trajectory is not None:
            line_segment = self.example_trajectory[:frame_pre_clip, :]
            if self.dim == 2:
                self.example_line.set_data(line_segment[:, 0], line_segment[:, 1])
            else:
                self.example_line.set_data_3d(
                    line_segment[:, 0], line_segment[:, 1], line_segment[:, 2]
                )
            updated_artists.append(self.example_line)
        return updated_artists

    def figure(self):
        """Invokes update over all frames to produce the final figure."""
        for i in range(self.n_time):
            _ = self.update(i)

"""Stimple utility functions to build animations."""

import jax.numpy as jnp


def create_update_func(animators):
    """Function to create an aggregate `update` function for animations."""

    def update(frame):
        artists = []
        for anim in animators:
            artists.extend(anim.update(frame))
        return artists

    return update


def hex_to_rgb(hex_color):
    """Function to convert hex colors to RGB."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# pylint: disable=consider-using-f-string
def rgb_to_hex(rgb_tuple):
    """Function to convert RGB colors to hex."""
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in rgb_tuple])


def interpolate_colors(color1, color2, n_steps=10):
    """Function to interpolate between two hex colors."""
    c1 = jnp.array(hex_to_rgb(color1))
    c2 = jnp.array(hex_to_rgb(color2))
    gradient = [rgb_to_hex(c1 + (c2 - c1) * t) for t in jnp.linspace(0, 1, n_steps)]
    return gradient

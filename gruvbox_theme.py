
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

########################
# GRUVBOX THEME CONFIG #
########################
GRUVBOX = {
    # Dark Backgrounds
    "bg0_h": "#1d2021",
    "bg0": "#282828",
    "bg": "#282828",
    "bg0_s": "#32302f",
    "bg1": "#3c3836",
    "bg2": "#504945",
    "bg3": "#665c54",
    "bg4": "#7c6f64",
    
    # Foregrounds
    "fg0": "#fbf1c7",
    "fg": "#ebdbb2",
    "fg1": "#ebdbb2",
    "fg2": "#d5c4a1",
    "fg3": "#bdae93",
    "fg4": "#a89984",
    
    # Normal / Dark Accents
    "red_dark": "#cc241d",
    "green_dark": "#98971a",
    "yellow_dark": "#d79921",
    "blue_dark": "#458588",
    "purple_dark": "#b16286",
    "aqua_dark": "#689d6a",
    "orange_dark": "#d65d0e",
    "gray_dark": "#928374",
    
    # Bright / Light Accents
    "red": "#fb4934",
    "green": "#b8bb26",
    "yellow": "#fabd2f",
    "blue": "#83a598",
    "purple": "#d3869b",
    "aqua": "#8ec07c",
    "orange": "#fe8019",
    "gray": "#a89984",
}

# Color cycle for line plots & scatter plots
GRUVBOX_CYCLE = [
    GRUVBOX["yellow"],
    GRUVBOX["orange"],
    GRUVBOX["blue"],
    GRUVBOX["purple"],
    GRUVBOX["green"],
    GRUVBOX["aqua"],
    GRUVBOX["red"],
]

# Custom Gruvbox Colormaps
# Sequential colormap: dark background -> blue -> aqua -> green -> yellow -> orange -> red
gruvbox_rainbow_colors = [
    GRUVBOX["bg0"],
    GRUVBOX["blue_dark"],
    GRUVBOX["aqua"],
    GRUVBOX["green"],
    GRUVBOX["yellow"],
    GRUVBOX["orange"],
    GRUVBOX["red"],
]
gruvbox_rainbow_cmap = LinearSegmentedColormap.from_list("gruvbox_rainbow", gruvbox_rainbow_colors)

# Heat colormap: dark background -> dark orange -> orange -> yellow -> bright white/fg
gruvbox_heat_colors = [
    GRUVBOX["bg0"],
    GRUVBOX["red_dark"],
    GRUVBOX["orange_dark"],
    GRUVBOX["orange"],
    GRUVBOX["yellow"],
    GRUVBOX["fg0"],
]
gruvbox_heat_cmap = LinearSegmentedColormap.from_list("gruvbox_heat", gruvbox_heat_colors)

mpl.colormaps.register(cmap=gruvbox_rainbow_cmap, name="gruvbox_rainbow")
mpl.colormaps.register(cmap=gruvbox_heat_cmap, name="gruvbox_heat")

def apply_gruvbox_theme(bg="bg0", fg="fg1"):
    """
    Applies the custom Gruvbox theme to Matplotlib and Seaborn parameters.
    """
    bg_color = GRUVBOX.get(bg, GRUVBOX["bg0"])
    fg_color = GRUVBOX.get(fg, GRUVBOX["fg1"])
    
    theme_params = {
        "figure.facecolor": bg_color,
        "axes.facecolor": bg_color,
        "savefig.facecolor": bg_color,
        "text.color": fg_color,
        "axes.labelcolor": fg_color,
        "axes.titlecolor": fg_color,
        "xtick.color": fg_color,
        "ytick.color": fg_color,
        "axes.edgecolor": GRUVBOX["bg4"],
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRUVBOX["bg2"],
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=GRUVBOX_CYCLE),
        "legend.facecolor": GRUVBOX["bg1"],
        "legend.edgecolor": GRUVBOX["bg3"],
        "legend.labelcolor": fg_color,
        "legend.framealpha": 0.8,
    }
    plt.rcParams.update(theme_params)
    sns.set_theme(style="darkgrid", rc=theme_params)

# Auto-apply Gruvbox theme on import
apply_gruvbox_theme()
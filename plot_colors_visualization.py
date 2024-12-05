import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import matplotlib.pyplot as plt
import numpy as np

def create_color_batch(value_idx, batch_size):
    # Define color dictionary, mapping color names to their corresponding channel intensity
    color_dict = {
        0: torch.tensor([1.0, 0.0, 0.0]),   # red
        1: torch.tensor([0.0, 1.0, 0.0]),   # green
        2: torch.tensor([0.0, 0.0, 1.0]),   # blue
        3: torch.tensor([1.0, 1.0, 0.0]),   # yellow
        4: torch.tensor([1.0, 0.0, 1.0]),   # magenta
        5: torch.tensor([0.0, 1.0, 1.0]),   # cyan
        6: torch.tensor([0.5, 0.0, 0.5]),   # purple
        7: torch.tensor([0.5, 0.5, 0.5]),   # grey
        8: torch.tensor([1.0, 0.5, 0.0]),   # orange
        9: torch.tensor([0.5, 0.5, 0.0]),   # olive
        10: torch.tensor([0.75, 0.0, 0.25]), # maroon
        11: torch.tensor([0.25, 0.75, 0.0]), # lime
        12: torch.tensor([0.0, 0.25, 0.75]), # navy
        13: torch.tensor([0.75, 0.75, 0.75]), # silver
        14: torch.tensor([0.75, 0.75, 0.0]),  # gold
        15: torch.tensor([0.0, 0.75, 0.75]),  # teal
        16: torch.tensor([0.75, 0.0, 0.75]),  # fuchsia
        17: torch.tensor([0.75, 0.25, 0.0]),  # rust
        18: torch.tensor([0.25, 0.75, 0.25]), # forest green
        19: torch.tensor([0.25, 0.25, 0.75]), # royal blue
        20: torch.tensor([0.5, 0.25, 0.0]),   # sienna
        21: torch.tensor([0.25, 0.5, 0.0]),   # moss
        22: torch.tensor([0.0, 0.5, 0.25]),   # sea green
        23: torch.tensor([0.5, 0.0, 0.25]),   # plum
        24: torch.tensor([0.25, 0.0, 0.5]),   # indigo
        25: torch.tensor([0.0, 0.25, 0.5]),   # slate blue
        26: torch.tensor([0.25, 0.0, 0.75]),  # violet
        27: torch.tensor([0.75, 0.0, 0.25]),  # crimson
        28: torch.tensor([0.0, 0.75, 0.25]),  # mint
        29: torch.tensor([0.25, 0.75, 0.5]),  # pastel green
        30: torch.tensor([0.75, 0.75, 0.25]), # pale yellow
        31: torch.tensor([0.75, 0.25, 0.75]), # pink
        32: torch.tensor([0.25, 0.75, 0.75]), # turquoise
        33: torch.tensor([0.25, 0.25, 0.25]), # dark grey
        34: torch.tensor([0.75, 0.0, 0.0]),   # dark red
        35: torch.tensor([0.0, 0.75, 0.0]),   # dark green
        36: torch.tensor([0.0, 0.0, 0.75]),   # dark blue
        37: torch.tensor([0.5, 0.5, 1.0]),    # light blue
        38: torch.tensor([1.0, 0.5, 0.5]),    # light red
        39: torch.tensor([0.5, 1.0, 0.5]),    # light green
        40: torch.tensor([0.75, 0.5, 0.25]),  # bronze
        41: torch.tensor([0.25, 0.5, 0.75]),  # sky blue
        42: torch.tensor([0.75, 0.25, 0.5]),  # raspberry
        43: torch.tensor([0.5, 0.75, 0.25]),  # lime green
        44: torch.tensor([0.25, 0.5, 0.5]),   # steel blue
        45: torch.tensor([0.5, 0.25, 0.75]),  # amethyst
        46: torch.tensor([0.25, 0.75, 0.25]), # spring green
        47: torch.tensor([0.75, 0.25, 0.25]), # salmon
        48: torch.tensor([0.25, 0.25, 0.75]), # cobalt blue
        49: torch.tensor([0.75, 0.25, 0.0]),  # terracotta
        50: torch.tensor([0.25, 0.75, 0.0]),  # apple green
        51: torch.tensor([0.0, 0.25, 0.75]),  # cerulean
        52: torch.tensor([0.75, 0.75, 0.5]),  # sand
        53: torch.tensor([0.75, 0.5, 0.75]),  # mauve
        54: torch.tensor([0.5, 0.75, 0.75]),  # aquamarine
        55: torch.tensor([0.5, 0.5, 0.5]),    # medium grey
        56: torch.tensor([0.5, 0.0, 0.0]),    # brick red
        57: torch.tensor([0.0, 0.5, 0.0]),    # emerald green
        58: torch.tensor([0.0, 0.0, 0.5]),    # navy blue
        59: torch.tensor([0.5, 0.25, 0.25]),  # rosewood
        60: torch.tensor([0.25, 0.5, 0.25]),  # fern green
        61: torch.tensor([0.25, 0.25, 0.5]),  # denim
        62: torch.tensor([0.75, 0.75, 0.25]), # mustard
        63: torch.tensor([0.75, 0.25, 0.75]), # orchid
        64: torch.tensor([0.25, 0.75, 0.75]), # ice blue
        65: torch.tensor([0.75, 0.5, 0.5]),   # coral
        66: torch.tensor([0.5, 0.75, 0.5]),   # moss green
        67: torch.tensor([0.5, 0.5, 0.75]),   # periwinkle
        68: torch.tensor([0.75, 0.5, 0.25]),  # copper
        69: torch.tensor([0.25, 0.5, 0.75]),  # peacock blue
        70: torch.tensor([0.75, 0.25, 0.5]),  # rose
        71: torch.tensor([0.5, 0.75, 0.25]),  # chartreuse
        72: torch.tensor([0.25, 0.5, 0.5]),   # slate grey
        73: torch.tensor([0.5, 0.25, 0.75]),  # lilac
        74: torch.tensor([0.25, 0.75, 0.25]), # mint green
        75: torch.tensor([0.75, 0.25, 0.25]), # sunset
        76: torch.tensor([0.25, 0.25, 0.75]), # midnight blue
        77: torch.tensor([0.75, 0.25, 0.0]),  # burnt orange
        78: torch.tensor([0.25, 0.75, 0.0]),  # lime yellow
        79: torch.tensor([0.0, 0.25, 0.75]),  # azure
        80: torch.tensor([0.75, 0.75, 0.5]),  # flax
        81: torch.tensor([0.75, 0.5, 0.75]),  # blush
        82: torch.tensor([0.5, 0.75, 0.75]),  # crystal blue
        83: torch.tensor([0.5, 0.5, 0.5]),    # granite
        84: torch.tensor([0.5, 0.0, 0.0]),    # burgundy
        85: torch.tensor([0.0, 0.5, 0.0]),    # forest green
        86: torch.tensor([0.0, 0.0, 0.5]),    # midnight blue
        87: torch.tensor([0.5, 0.25, 0.25]),  # mahogany
        88: torch.tensor([0.25, 0.5, 0.25]),  # sage green
        89: torch.tensor([0.25, 0.25, 0.5]),  # blueberry
        90: torch.tensor([0.75, 0.75, 0.25]), # brass
        91: torch.tensor([0.75, 0.25, 0.75]), # heather
        92: torch.tensor([0.25, 0.75, 0.75]), # glacial blue
        93: torch.tensor([0.75, 0.5, 0.5]),   # peach
        94: torch.tensor([0.5, 0.75, 0.5]),   # sage
        95: torch.tensor([0.5, 0.5, 0.75]),   # wisteria
        96: torch.tensor([0.75, 0.5, 0.25]),  # chestnut
        97: torch.tensor([0.25, 0.5, 0.75]),  # lagoon blue
        98: torch.tensor([0.75, 0.25, 0.5]),  # rose pink
        99: torch.tensor([0.5, 0.75, 0.25])   # lime
    }

    # Choose color
    color = color_dict.get(value_idx % 100, torch.tensor([0.0, 0.0, 0.0]))  # 默认为黑色

    # Create batch
    color_batch = color.unsqueeze(0).repeat(batch_size, 1)
    return color_batch

# Visualize 100 colors
def visualize_colors():
    color_dict = {
        'Red': [1.0, 0.0, 0.0],
        'Green': [0.0, 1.0, 0.0],
        'Blue': [0.0, 0.0, 1.0],
        'Yellow': [1.0, 1.0, 0.0],
        'Magenta': [1.0, 0.0, 1.0],
        'Cyan': [0.0, 1.0, 1.0],
        'Purple': [0.5, 0.0, 0.5],
        'Grey': [0.5, 0.5, 0.5],
        'Orange': [1.0, 0.5, 0.0],
        'Olive': [0.5, 0.5, 0.0],
        'Maroon': [0.75, 0.0, 0.25],
        'Lime': [0.25, 0.75, 0.0],
        'Navy': [0.0, 0.25, 0.75],
        'Silver': [0.75, 0.75, 0.75],
        'Gold': [0.75, 0.75, 0.0],
        'Teal': [0.0, 0.75, 0.75],
        'Fuchsia': [0.75, 0.0, 0.75],
        'Rust': [0.75, 0.25, 0.0],
        'Forest Green': [0.25, 0.75, 0.25],
        'Royal Blue': [0.25, 0.25, 0.75],
        'Sienna': [0.5, 0.25, 0.0],
        'Moss': [0.25, 0.5, 0.0],
        'Sea Green': [0.0, 0.5, 0.25],
        'Plum': [0.5, 0.0, 0.25],
        'Indigo': [0.25, 0.0, 0.5],
        'Slate Blue': [0.0, 0.25, 0.5],
        'Violet': [0.25, 0.0, 0.75],
        'Crimson': [0.75, 0.0, 0.25],
        'Mint': [0.0, 0.75, 0.25],
        'Pastel Green': [0.25, 0.75, 0.5],
        'Pale Yellow': [0.75, 0.75, 0.25],
        'Pink': [0.75, 0.25, 0.75],
        'Turquoise': [0.25, 0.75, 0.75],
        'Dark Grey': [0.25, 0.25, 0.25],
        'Dark Red': [0.75, 0.0, 0.0],
        'Dark Green': [0.0, 0.75, 0.0],
        'Dark Blue': [0.0, 0.0, 0.75],
        'Light Blue': [0.5, 0.5, 1.0],
        'Light Red': [1.0, 0.5, 0.5],
        'Light Green': [0.5, 1.0, 0.5],
        'Bronze': [0.75, 0.5, 0.25],
        'Sky Blue': [0.25, 0.5, 0.75],
        'Raspberry': [0.75, 0.25, 0.5],
        'Lime Green': [0.5, 0.75, 0.25],
        'Steel Blue': [0.25, 0.5, 0.5],
        'Amethyst': [0.5, 0.25, 0.75],
        'Spring Green': [0.25, 0.75, 0.25],
        'Salmon': [0.75, 0.25, 0.25],
        'Cobalt Blue': [0.25, 0.25, 0.75],
        'Terracotta': [0.75, 0.25, 0.0],
        'Apple Green': [0.25, 0.75, 0.0],
        'Cerulean': [0.0, 0.25, 0.75],
        'Sand': [0.75, 0.75, 0.5],
        'Mauve': [0.75, 0.5, 0.75],
        'Aquamarine': [0.5, 0.75, 0.75],
        'Medium Grey': [0.5, 0.5, 0.5],
        'Brick Red': [0.5, 0.0, 0.0],
        'Emerald Green': [0.0, 0.5, 0.0],
        'Navy Blue': [0.0, 0.0, 0.5],
        'Rosewood': [0.5, 0.25, 0.25],
        'Fern Green': [0.25, 0.5, 0.25],
        'Denim': [0.25, 0.25, 0.5],
        'Mustard': [0.75, 0.75, 0.25],
        'Orchid': [0.75, 0.25, 0.75],
        'Ice Blue': [0.25, 0.75, 0.75],
        'Coral': [0.75, 0.5, 0.5],
        'Moss Green': [0.5, 0.75, 0.5],
        'Periwinkle': [0.5, 0.5, 0.75],
        'Copper': [0.75, 0.5, 0.25],
        'Peacock Blue': [0.25, 0.5, 0.75],
        'Rose': [0.75, 0.25, 0.5],
        'Chartreuse': [0.5, 0.75, 0.25],
        'Slate Grey': [0.25, 0.5, 0.5],
        'Lilac': [0.5, 0.25, 0.75],
        'Mint Green': [0.25, 0.75, 0.25],
        'Sunset': [0.75, 0.25, 0.25],
        'Midnight Blue': [0.25, 0.25, 0.75],
        'Burnt Orange': [0.75, 0.25, 0.0],
        'Lime Yellow': [0.25, 0.75, 0.0],
        'Azure': [0.0, 0.25, 0.75],
        'Flax': [0.75, 0.75, 0.5],
        'Blush': [0.75, 0.5, 0.75],
        'Crystal Blue': [0.5, 0.75, 0.75],
        'Granite': [0.5, 0.5, 0.5],
        'Burgundy': [0.5, 0.0, 0.0],
        'Forest Green': [0.0, 0.5, 0.0],
        'Midnight Blue': [0.0, 0.0, 0.5],
        'Mahogany': [0.5, 0.25, 0.25],
        'Sage Green': [0.25, 0.5, 0.25],
        'Blueberry': [0.25, 0.25, 0.5],
        'Brass': [0.75, 0.75, 0.25],
        'Heather': [0.75, 0.25, 0.75],
        'Glacial Blue': [0.25, 0.75, 0.75],
        'Peach': [0.75, 0.5, 0.5],
        'Sage': [0.5, 0.75, 0.5],
        'Wisteria': [0.5, 0.5, 0.75],
        'Chestnut': [0.75, 0.5, 0.25],
        'Lagoon Blue': [0.25, 0.5, 0.75],
        'Rose Pink': [0.75, 0.25, 0.5],
        'Lime': [0.5, 0.75, 0.25]
    }

    fig, ax = plt.subplots(10, 10, figsize=(20, 20))
    for idx, (color_name, color_value) in enumerate(color_dict.items()):
        ax[idx // 10, idx % 10].imshow(np.ones((10, 10, 3)) * color_value)
        ax[idx // 10, idx % 10].set_title(color_name, fontsize=8)
        ax[idx // 10, idx % 10].axis('off')

    plt.show()

visualize_colors()

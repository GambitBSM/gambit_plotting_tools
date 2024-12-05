import matplotlib


# 
# The standard GAMBIT color map for 2D profile likelihood plots
# 

# Color map from pippi-generated ctioga2 script: #000--#000(0.04599999999999993)--#33f(0.31700000000000006)--#0ff(0.5)--#ff0
hex_colors = ["#000", "#33f", "#0ff", "#ff0"]
rgb_colors = [matplotlib.colors.hex2color(hex_color) for hex_color in hex_colors]
cdict = {
         "red":   [(0.00000,  rgb_colors[0][0], rgb_colors[0][0]),
                   (0.04599,  rgb_colors[0][0], rgb_colors[0][0]),
                   (0.31700,  rgb_colors[1][0], rgb_colors[1][0]),
                   (0.50000,  rgb_colors[2][0], rgb_colors[2][0]),
                   (1.00000,  rgb_colors[3][0], rgb_colors[3][0])],

         "green": [(0.00000,  rgb_colors[0][1], rgb_colors[0][1]),
                   (0.04599,  rgb_colors[0][1], rgb_colors[0][1]),
                   (0.31700,  rgb_colors[1][1], rgb_colors[1][1]),
                   (0.50000,  rgb_colors[2][1], rgb_colors[2][1]),
                   (1.00000,  rgb_colors[3][1], rgb_colors[3][1])],

         "blue":  [(0.00000,  rgb_colors[0][2], rgb_colors[0][2]),
                   (0.04599,  rgb_colors[0][2], rgb_colors[0][2]),
                   (0.31700,  rgb_colors[1][2], rgb_colors[1][2]),
                   (0.50000,  rgb_colors[2][2], rgb_colors[2][2]),
                   (1.00000,  rgb_colors[3][2], rgb_colors[3][2])]
}
gambit_std_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gambit_cmap", rgb_colors)


# 
# Add more color maps below
# 


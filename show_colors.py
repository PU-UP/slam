import pandas as pd
import matplotlib.colors as mcolors
import colorsys

# Prepare list of CSS4 colors
colors = []
for name, hexcode in mcolors.CSS4_COLORS.items():
    r, g, b = mcolors.to_rgb(hexcode)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    colors.append({'name': name, 'hex': hexcode, 'lightness': l, 'hue': h})

df = pd.DataFrame(colors)

# Define hue ranges for each target color
hue_ranges = {
    'Red': [(0, 0.05), (0.95, 1.0)],
    'Yellow': [(0.10, 0.17)],
    'Blue': [(0.55, 0.75)],
    'Green': [(0.25, 0.45)]
}

# Filter and sort for each color category
results = {}
for color_name, ranges in hue_ranges.items():
    df_filtered = pd.DataFrame()
    for low, high in ranges:
        df_filtered = pd.concat([df_filtered, df[(df['hue'] >= low) & (df['hue'] <= high)]])
    df_filtered = df_filtered.drop_duplicates().sort_values('lightness').reset_index(drop=True)
    results[color_name] = df_filtered[['name', 'hex', 'lightness']]

# Write to txt file
file_path = 'colors_list.txt'
with open(file_path, 'w', encoding='utf-8') as f:
    for category, table in results.items():
        f.write(f"{category} colors (dark to light):\n")
        for _, row in table.iterrows():
            f.write(f"{row['name']}\t{row['hex']}\t{row['lightness']:.3f}\n")
        f.write("\n")

file_path

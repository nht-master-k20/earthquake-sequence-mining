"""
Earthquake Data Analysis - Meaningful Visualizations
Focus on insights that matter for earthquake analysis
All charts are separate, not combined
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

# Set style for professional appearance
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

# Read data
df = pd.read_csv('/home/haind/Desktop/earthquake-sequence-mining/dongdat.csv')
df['time'] = pd.to_datetime(df['time'])

# ============================================================================
# 1. TEMPORAL ANALYSIS - Time Trends
# ============================================================================

# 1.1 Monthly earthquake counts
fig, ax = plt.subplots(figsize=(14, 6))
df_monthly = df.set_index('time').resample('ME').size()
ax.plot(df_monthly.index, df_monthly.values, linewidth=1.5, color='#2c3e50')
ax.fill_between(df_monthly.index, df_monthly.values, alpha=0.3, color='#3498db')
ax.set_title('Earthquake Frequency Over Time (Monthly)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Earthquakes', fontsize=11)
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/01_temporal_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# 1.2 Moving average to show trends
fig, ax = plt.subplots(figsize=(14, 6))
df_monthly_ma = df_monthly.rolling(window=6, center=True).mean()
ax.plot(df_monthly.index, df_monthly.values, alpha=0.4, label='Actual Count', color='#95a5a6')
ax.plot(df_monthly_ma.index, df_monthly_ma.values, linewidth=2.5, label='6-Month Moving Average', color='#e74c3c')
ax.set_title('Earthquake Trend Analysis (Moving Average)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Earthquakes', fontsize=11)
ax.legend()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/02_temporal_moving_average.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== TEMPORAL STATISTICS ===")
print(f"Total earthquakes: {len(df):,}")
print(f"Period: {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
print(f"Average: {df_monthly.mean():.1f} earthquakes/month")
print(f"Peak: {df_monthly.max()} earthquakes in {df_monthly.idxmax().strftime('%Y-%m')}")
print(f"Lowest: {df_monthly.min()} earthquakes in {df_monthly.idxmin().strftime('%Y-%m')}")
print()

# ============================================================================
# 2. MAGNITUDE DISTRIBUTION - Frequency-Magnitude Distribution
# ============================================================================

# 2.1 Magnitude histogram (Gutenberg-Richter Law)
fig, ax = plt.subplots(figsize=(12, 6))
bins = np.arange(df['mag'].min(), df['mag'].max() + 0.5, 0.5)
counts, edges, patches = ax.hist(df['mag'], bins=bins, edgecolor='black', alpha=0.7, color='#3498db')
ax.set_title('Magnitude Distribution (Gutenberg-Richter Law)', fontsize=14, fontweight='bold')
ax.set_xlabel('Magnitude', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add trend line for b-value
from scipy import stats
bin_centers = (edges[:-1] + edges[1:]) / 2
valid_mask = counts > 0
if np.sum(valid_mask) > 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(bin_centers[valid_mask], np.log(counts[valid_mask]))
    x_fit = np.linspace(df['mag'].min(), df['mag'].max(), 100)
    y_fit = np.exp(slope * x_fit + intercept)
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'b-value = {-slope:.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/03_magnitude_histogram.png', dpi=150, bbox_inches='tight')
plt.close()

# 2.2 Cumulative distribution
fig, ax = plt.subplots(figsize=(12, 6))
sorted_mag = np.sort(df['mag'])
cumulative = np.arange(1, len(sorted_mag) + 1)[::-1]
ax.plot(sorted_mag, cumulative, linewidth=2, color='#e74c3c')
ax.set_title('Cumulative Distribution: Count ≥ Magnitude', fontsize=14, fontweight='bold')
ax.set_xlabel('Magnitude', fontsize=11)
ax.set_ylabel('Number of Earthquakes', fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/04_magnitude_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== MAGNITUDE STATISTICS ===")
print(f"Min: {df['mag'].min():.2f}")
print(f"Max: {df['mag'].max():.2f}")
print(f"Mean: {df['mag'].mean():.2f}")
print(f"Median: {df['mag'].median():.2f}")
print(f"Std Dev: {df['mag'].std():.2f}")
print(f"M ≥ 4.0: {(df['mag'] >= 4.0).sum():,}")
print(f"M ≥ 5.0: {(df['mag'] >= 5.0).sum():,}")
print(f"M ≥ 6.0: {(df['mag'] >= 6.0).sum():,}")
print()

# ============================================================================
# 3. DEPTH ANALYSIS - Depth Distribution
# ============================================================================

# 3.1 Depth histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['depth'], bins=50, edgecolor='black', alpha=0.7, color='#27ae60')
ax.axvline(df['depth'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df["depth"].median():.1f} km')
ax.set_title('Earthquake Depth Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Depth (km)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/05_depth_histogram.png', dpi=150, bbox_inches='tight')
plt.close()

# 3.2 Magnitude vs Depth relationship
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(df['depth'], df['mag'], c=df['mag'], cmap='RdYlBu_r', alpha=0.5, s=20)
ax.set_title('Magnitude vs Depth Relationship', fontsize=14, fontweight='bold')
ax.set_xlabel('Depth (km)', fontsize=11)
ax.set_ylabel('Magnitude', fontsize=11)
plt.colorbar(scatter, ax=ax, label='Magnitude')
ax.grid(True, alpha=0.3)

corr = df[['mag', 'depth']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/06_magnitude_vs_depth.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== DEPTH STATISTICS ===")
print(f"Min: {df['depth'].min():.2f} km")
print(f"Max: {df['depth'].max():.2f} km")
print(f"Mean: {df['depth'].mean():.2f} km")
print(f"Median: {df['depth'].median():.2f} km")
print(f"Correlation (Mag vs Depth): {corr:.3f}")
print()

# ============================================================================
# 4. SPATIAL DISTRIBUTION - Geographic Distribution
# ============================================================================

# 4.1 Spatial scatter plot with world map
import urllib.request
import shapefile

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor('#e6f3ff')
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# Draw world map using Natural Earth admin 0 countries
sf_countries = shapefile.Reader('/tmp/naturalearth/ne_110m_admin_0_countries.shp')

for shape_rec in sf_countries.shapeRecords():
    points = np.array(shape_rec.shape.points)
    parts = list(shape_rec.shape.parts) + [len(points)]

    for i in range(len(parts) - 1):
        part_points = points[parts[i]:parts[i+1]]
        if len(part_points) > 2:
            poly = plt.Polygon(part_points, facecolor='#d9d9d9', edgecolor='#999999', linewidth=0.5, alpha=0.7)
            ax.add_patch(poly)

# Grid lines
ax.set_xticks(np.arange(-180, 181, 30))
ax.set_yticks(np.arange(-90, 91, 30))
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#666666', zorder=1)

# Plot earthquake data
scatter1 = ax.scatter(df['longitude'], df['latitude'], c=df['mag'], cmap='RdYlBu_r',
                      s=df['mag']*2, alpha=0.7, edgecolors='black', linewidth=0.1, zorder=3)

ax.set_title('Global Earthquake Distribution (World Map)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_aspect('equal')

# Colorbar
cbar1 = plt.colorbar(scatter1, ax=ax, orientation='horizontal', pad=0.08, aspect=40)
cbar1.set_label('Magnitude', fontsize=11)

plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/07_spatial_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# 4.2 Density heatmap with world map
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor('#e6f3ff')
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# Draw world map
for shape_rec in sf_countries.shapeRecords():
    points = np.array(shape_rec.shape.points)
    parts = list(shape_rec.shape.parts) + [len(points)]

    for i in range(len(parts) - 1):
        part_points = points[parts[i]:parts[i+1]]
        if len(part_points) > 2:
            poly = plt.Polygon(part_points, facecolor='#d9d9d9', edgecolor='#999999', linewidth=0.5, alpha=0.7)
            ax.add_patch(poly)

# Draw hexbin heatmap
hb = ax.hexbin(df['longitude'], df['latitude'], gridsize=100, cmap='YlOrRd', bins='log',
               alpha=0.7, edgecolors='none', zorder=3)

ax.set_title('Earthquake Density Heatmap (World Map)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_aspect('equal')

# Colorbar
cbar2 = plt.colorbar(hb, ax=ax, orientation='horizontal', pad=0.08, aspect=40)
cbar2.set_label('Log10(Count)', fontsize=11)

plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/08_spatial_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("=== SPATIAL STATISTICS ===")
print(f"Latitude: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
print(f"Longitude: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
print(f"Coverage area: ~{(df['latitude'].max()-df['latitude'].min())*(df['longitude'].max()-df['longitude'].min()):.1f} deg²")
print()

# ============================================================================
# 5. ADDITIONAL INSIGHTS - Supplementary Analysis
# ============================================================================

# 5.1 Top 10 locations with most earthquakes
fig, ax = plt.subplots(figsize=(12, 6))
top_places = df['place'].value_counts().head(10)
bars = ax.barh(range(len(top_places)), top_places.values, color='#3498db', alpha=0.7)
ax.set_yticks(range(len(top_places)))
ax.set_yticklabels(top_places.index, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Number of Earthquakes', fontsize=11)
ax.set_title('Top 10 Locations with Most Earthquakes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, (bar, value) in enumerate(zip(bars, top_places.values)):
    ax.text(value + max(top_places.values)*0.01, i, f'{value:,}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/09_top_locations.png', dpi=150, bbox_inches='tight')
plt.close()

# 5.2 Top 10 locations with strongest earthquakes (M ≥ 5.0)
fig, ax = plt.subplots(figsize=(12, 6))
strong_quakes = df[df['mag'] >= 5.0]
top_strong_places = strong_quakes['place'].value_counts().head(10)
bars = ax.barh(range(len(top_strong_places)), top_strong_places.values, color='#e74c3c', alpha=0.7)
ax.set_yticks(range(len(top_strong_places)))
ax.set_yticklabels(top_strong_places.index, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Number of Earthquakes (M ≥ 5.0)', fontsize=11)
ax.set_title('Top 10 Locations with Strongest Earthquakes (M ≥ 5.0)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, (bar, value) in enumerate(zip(bars, top_strong_places.values)):
    ax.text(value + max(top_strong_places.values)*0.01, i, f'{value:,}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/10_top_strong_earthquakes.png', dpi=150, bbox_inches='tight')
plt.close()

# 5.3 Seasonal pattern
fig, ax = plt.subplots(figsize=(10, 6))
df['month'] = df['time'].dt.month
seasonal = df.groupby('month').size()
ax.bar(seasonal.index, seasonal.values, color='#27ae60', alpha=0.7)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Number of Earthquakes', fontsize=11)
ax.set_title('Seasonal Pattern of Earthquakes', fontsize=14, fontweight='bold')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/11_seasonal_pattern.png', dpi=150, bbox_inches='tight')
plt.close()

# 5.4 Magnitude categories (USGS classification)
fig, ax = plt.subplots(figsize=(10, 6))
mag_categories = pd.cut(df['mag'], bins=[0, 2, 3, 4, 5, 10],
                        labels=['Micro (<2)', 'Minor (2-3)', 'Light (3-4)', 'Moderate (4-5)', 'Strong (≥5)'])
mag_counts = mag_categories.value_counts().sort_index()
bars = ax.bar(range(len(mag_counts)), mag_counts.values, color='#e74c3c', alpha=0.7)
ax.set_xticks(range(len(mag_counts)))
ax.set_xticklabels(mag_counts.index, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Number of Earthquakes', fontsize=11)
ax.set_title('Magnitude Classification (USGS Standard)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, value) in enumerate(zip(bars, mag_counts.values)):
    ax.text(i, value + max(mag_counts.values)*0.01, f'{value:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/12_magnitude_categories.png', dpi=150, bbox_inches='tight')
plt.close()

# 5.5 Depth categories
fig, ax = plt.subplots(figsize=(10, 6))
depth_categories = pd.cut(df['depth'], bins=[0, 10, 35, 70, 300, 1000],
                          labels=['Shallow (<10km)', 'Intermediate (10-35km)', 'Medium (35-70km)', 'Deep (70-300km)', 'Very Deep (>300km)'])
depth_counts = depth_categories.value_counts().sort_index()
bars = ax.bar(range(len(depth_counts)), depth_counts.values, color='#9b59b6', alpha=0.7)
ax.set_xticks(range(len(depth_counts)))
ax.set_xticklabels(depth_counts.index, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Number of Earthquakes', fontsize=11)
ax.set_title('Depth Classification', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, value) in enumerate(zip(bars, depth_counts.values)):
    ax.text(i, value + max(depth_counts.values)*0.01, f'{value:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/haind/Desktop/earthquake-sequence-mining/haind/13_depth_categories.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n=== COMPLETE ===")
print("Created 13 individual charts:")
print("1. 01_temporal_trend.png - Monthly earthquake frequency")
print("2. 02_temporal_moving_average.png - Moving average trend")
print("3. 03_magnitude_histogram.png - Magnitude distribution (Gutenberg-Richter)")
print("4. 04_magnitude_cumulative.png - Cumulative distribution")
print("5. 05_depth_histogram.png - Depth distribution")
print("6. 06_magnitude_vs_depth.png - Magnitude vs depth relationship")
print("7. 07_spatial_scatter.png - Global spatial distribution (world map)")
print("8. 08_spatial_heatmap.png - Density heatmap (world map)")
print("9. 09_top_locations.png - Top 10 locations with most earthquakes")
print("10. 10_top_strong_earthquakes.png - Top 10 locations with strongest earthquakes (M ≥ 5.0)")
print("11. 11_seasonal_pattern.png - Seasonal pattern")
print("12. 12_magnitude_categories.png - Magnitude classification (USGS)")
print("13. 13_depth_categories.png - Depth classification")

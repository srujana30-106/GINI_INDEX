import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#load csv data set
file_path = r'C:\Users\RGUKT\Desktop\GINI-index\input_raw_data.csv'
df = pd.read_csv(file_path)

# consider only certain columns from input file
quintile_columns = ['q1', 'q2', 'q3', 'q4', 'q5']
columns_to_keep = ['id', 'country', 'c3', 'year'] + quintile_columns + ['region_un', 'region_un_sub']

# Filter data such that duplicate and null values will be deleted
df_2010_complete = df[
    (df['year'] == 2010) & 
    df[quintile_columns].notnull().all(axis=1)
]
final_df = df_2010_complete[columns_to_keep].drop_duplicates(subset='country')

# Calculate Gini_index
def calculate_gini(row):
    shares = [row[q] for q in quintile_columns]
    cum_shares = [0]
    for s in shares:
        cum_shares.append(cum_shares[-1] + s)
    cum_shares = [s / 100 for s in cum_shares]  # convert to proportions
    gini = 1 - sum((cum_shares[i] + cum_shares[i-1]) * 0.2 for i in range(1, 6))
    return round(gini, 4)

final_df['gini_index'] = final_df.apply(calculate_gini, axis=1)

#classify countries into equally distributed and unequal distribution
equality_top10 = final_df[['country', 'gini_index']].sort_values(by='gini_index').head(10)
inequality_top10 = final_df[['country', 'gini_index']].sort_values(by='gini_index', ascending=False).head(10)


print("\n Top 10 Most Equal Countries (Lowest Gini):")
print(equality_top10.to_string(index=False))

print("\n Top 10 Most Unequal Countries (Highest Gini):")
print(inequality_top10.to_string(index=False))
print(final_df.to_string(index=False))
final_df.to_csv('gini_index_filtered.csv', index=False)

# Sort by Gini index (optional, for better visualization)
plot_df = final_df[['country', 'gini_index']].sort_values(by='gini_index', ascending=False)
region_stats = final_df.groupby('region_un')['gini_index'].mean().sort_values()
print(region_stats)

# Plot
plt.figure(figsize=(14, 8))
plt.barh(plot_df['country'], plot_df['gini_index'], color='skyblue')
plt.xlabel('Gini Index')
plt.title('Gini Index by Country (2010)')
plt.gca().invert_yaxis()  # Highest Gini on top
plt.tight_layout()
plt.show()

fig = px.choropleth(
    final_df,
    locations="c3",                #country codes
    color="gini_index",            # Column to color by
    hover_name="country",          # Hover info
    color_continuous_scale="RdYlGn_r",  # Red = high inequality, Green = low
    title="Gini Index by Country "
)

fig.write_html("gini_map.html")
print("Map saved as gini_map.html. Open it in browser.")

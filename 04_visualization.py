import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# This script aggrefates and visualizes the results from the evaluation

evaluation_path = 'Evaluation\\full_evaluation.json'

with open(evaluation_path, 'r') as file:
    data = json.load(file)



data_df = pd.DataFrame(data)
data_df = data_df.T
print(data_df)

od_metrics = ["accuracy od1","precision od1","recall od1","f1_score od1","iou od1"]
is_metrics = ["accuracy is","precision is","recall is","f1_score is","iou is"]
gpt_metrics = ["accuracy gpt1","precision gpt1","recall gpt1","f1_score gpt1","iou gpt1"]

metrics = [od_metrics,is_metrics,gpt_metrics]


for metric in metrics:
    mean = data_df[metric].mean()
    median = data_df[metric].median()
    percentile25 = data_df[metric].quantile(0.25)
    percentile75 = data_df[metric].quantile(0.75)
    metrics_table = pd.DataFrame([mean, percentile25, median, percentile75], index=['Mean', '25%','Median','75%']).T
    print(metrics_table.to_latex())




df_category = data_df.groupby("category").mean()
print(df_category)

print(df_category [od_metrics].to_latex())
print(df_category [is_metrics].to_latex())
print(df_category [gpt_metrics].to_latex())

ar_data_df = data_df.copy()
s_data_df = data_df.copy()
ar_data_df['aspect ratio bin'] = pd.cut(data_df['aspect ratio'], bins=[0.6,1.0,1.4,1.8,2.4], include_lowest=False)
s_data_df["sizes bin"] = pd.cut(data_df['size10000'], bins=[0,100,500,1000,5000], include_lowest=False)
print(ar_data_df['aspect ratio bin'].value_counts())
print(s_data_df['sizes bin'].value_counts())
# Group by the binned aspect ratio and calculate mean for each performance metric
grouped_by_aspect_ratio_bin = ar_data_df.drop(['category'],axis=1).groupby('aspect ratio bin').mean()
grouped_by_sizes_bin = s_data_df.drop(['category'],axis=1).groupby('sizes bin').mean()

print(grouped_by_aspect_ratio_bin[od_metrics])
print(grouped_by_aspect_ratio_bin[is_metrics])
print(grouped_by_aspect_ratio_bin[gpt_metrics])

print(grouped_by_sizes_bin[od_metrics])
print(grouped_by_sizes_bin[is_metrics])
print(grouped_by_sizes_bin[gpt_metrics])


boxplot_labels = ["Object Detection 0.3", "Object Detection 0.6","Image Segmentation", "GPT Prompt 1", "GPT Prompt 2"]

metric_order = ["accuracy", "precision", "recall", "f1_score", "iou"]  # Define custom row order
approach_order = ["od1", "od2", "is", "gpt1", "gpt2"]  # Define custom column order

# Assuming data_df is your DataFrame
accuracy_df = data_df[["accuracy od1", "accuracy od2", "accuracy is", "accuracy gpt1", "accuracy gpt2"]]
precision_df = data_df[["precision od1","precision od2", "precision is", "precision gpt1","precision gpt2"]]
recall_df = data_df[["recall od1", "recall od2", "recall is", "recall gpt1", "recall gpt2"]]
f1_score_df = data_df[["f1_score od1","f1_score od2", "f1_score is", "f1_score gpt1", "f1_score gpt2"]]
iou_df = data_df[["iou od1","iou od2", "iou is", "iou gpt1", "iou gpt2"]]

# Calculate the mean for each column
mean_accuracy = accuracy_df.mean()
mean_precision = precision_df.mean()
mean_recall = recall_df.mean()
mean_f1_score = f1_score_df.mean()
mean_iou = iou_df.mean()



# Create a figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(26, 16)) 

# Flatten the axes array for easy iteration
axes = axes.flatten()

# List of DataFrames and their corresponding titles and means
dataframes = [accuracy_df, precision_df, recall_df, f1_score_df, iou_df]
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']
means = [mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_iou]

# Plot each metric in a subplot
for i, (df, title, mean) in enumerate(zip(dataframes, titles, means)):
    ax = axes[i]
    sns.violinplot(data=df, ax=ax, inner="box", inner_kws=dict(box_width=10,whis_width=2))  # Thicker boxplot lines
    ax.set_xticklabels(approach_order, fontsize = 20)
    ax.set_ylabel(title, fontsize = 20)
    ax.set_ylim(0,1)
    # Annotate with mean values
    for j, m in enumerate(mean):
        ax.text(j, m, f' ⌀{m:.2f}', horizontalalignment='left', verticalalignment='bottom', fontsize = 20)

# Remove the empty subplot (if any)
if len(dataframes) < len(axes):
    fig.delaxes(axes[-1])

# Adjust layout and display
plt.subplots_adjust(hspace=0.25)  # Increase spacing between rows
plt.savefig("performance_violinplot.png", dpi=300, bbox_inches="tight") 
plt.show()

accuracy_df = df_category[["accuracy od1", "accuracy od2", "accuracy is", "accuracy gpt1", "accuracy gpt2"]]
precision_df = df_category[["precision od1","precision od2", "precision is", "precision gpt1","precision gpt2"]]
recall_df = df_category[["recall od1", "recall od2", "recall is", "recall gpt1", "recall gpt2"]]
f1_score_df = df_category[["f1_score od1","f1_score od2", "f1_score is", "f1_score gpt1", "f1_score gpt2"]]
iou_df = df_category[["iou od1","iou od2", "iou is", "iou gpt1", "iou gpt2"]]






df = data_df
metric_columns = [col for col in df.columns if col not in ["category", "size10000", "aspect ratio"]]

# Melt the DataFrame
df_melted = df.melt(id_vars=["category", "size10000", "aspect ratio"], 
                     value_vars=metric_columns, 
                     var_name="Metric_Approach", 
                     value_name="Value")

df_melted["Metric"] = df_melted["Metric_Approach"].apply(lambda x: x.split()[0])
df_melted["Approach"] = df_melted["Metric_Approach"].apply(lambda x: x.split()[1])

# Convert 'Value' to numeric (handles possible non-numeric values)
df_melted["Value"] = pd.to_numeric(df_melted["Value"], errors="coerce")




df_melted = df_melted.drop(['Metric_Approach'],axis = 1)
print(df_melted)

categories = ["single-person", "pair", "group", "animal", "other", "none"]



# Create heatmaps for each category
num_categories = len(categories)
fig, axes = plt.subplots(2,3, figsize=(28,20))
axes = axes.flatten()

if num_categories == 1:
    axes = [axes]  # Ensure axes is iterable for a single category

for i, category in enumerate(categories):
    sub_df = df_melted[df_melted["category"] == category]
    pivoted = sub_df.groupby(["Metric", "Approach"])["Value"].mean().unstack()
    pivoted = pivoted.astype(float).fillna(0)  # Replace NaN values with 0
    pivoted = pivoted.reindex(index=metric_order, columns=approach_order)
    # Plot heatmap
    heatmap = sns.heatmap(pivoted, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=axes[i],vmin=0,vmax=1, annot_kws={"size": 22})  
    axes[i].set_title(f"Category: {category}", fontsize=20)
    axes[i].set_xlabel("", fontsize=16)
    axes[i].set_ylabel("", fontsize=16)
    
    # Rotate labels for better readability
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha="right",fontsize = 18)
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0,fontsize = 18)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("", fontsize=12)  # Set colorbar label font size
    colorbar.ax.tick_params(labelsize=20)  # Set colorbar tick label font size

plt.subplots_adjust(hspace=0.25)  # Increase spacing between heatmaps
plt.subplots_adjust(wspace=0.1)  # Increase spacing between heatmaps
plt.savefig("category_heatmap.png", dpi=300, bbox_inches="tight")  # Save before plt.show()
plt.show()


df = ar_data_df
metric_columns = [col for col in df.columns if col not in ["category", "size10000", "aspect ratio","aspect ratio bin"]]

# Melt the DataFrame
df_melted = df.melt(id_vars=["category", "size10000", "aspect ratio", "aspect ratio bin" ], 
                     value_vars=metric_columns, 
                     var_name="Metric_Approach", 
                     value_name="Value")

df_melted["Metric"] = df_melted["Metric_Approach"].apply(lambda x: x.split()[0])
df_melted["Approach"] = df_melted["Metric_Approach"].apply(lambda x: x.split()[1])

# Convert 'Value' to numeric (handles possible non-numeric values)
df_melted["Value"] = pd.to_numeric(df_melted["Value"], errors="coerce")




df_melted = df_melted.drop(['Metric_Approach'],axis = 1)
print(df_melted)

aspect_ratios = df_melted.sort_values(by='aspect ratio bin')['aspect ratio bin'].unique()

# Create heatmaps for each category
num_categories = len(aspect_ratios)
fig, axes = plt.subplots(2,2, figsize=(28,20))
axes = axes.flatten()

if num_categories == 1:
    axes = [axes]  # Ensure axes is iterable for a single category

for i, ratio in enumerate(aspect_ratios):
    sub_df = df_melted[df_melted['aspect ratio bin'] == ratio]
    pivoted = sub_df.groupby(["Metric", "Approach"])["Value"].mean().unstack()
    pivoted = pivoted.astype(float).fillna(0)  # Replace NaN values with 0
    pivoted = pivoted.reindex(index=metric_order, columns=approach_order)
    # Plot heatmap
    heatmap = sns.heatmap(pivoted, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=axes[i],vmin=0,vmax=1, annot_kws={"size": 22})  
    axes[i].set_title(f"width/height: {ratio}", fontsize=20)
    axes[i].set_xlabel("", fontsize=16)
    axes[i].set_ylabel("", fontsize=16)
    
    # Rotate labels for better readability
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha="right",fontsize = 18)
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0,fontsize = 18)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("", fontsize=12)  # Set colorbar label font size
    colorbar.ax.tick_params(labelsize=20)  # Set colorbar tick label font size

plt.subplots_adjust(hspace=0.25)  # Increase spacing between heatmaps
plt.subplots_adjust(wspace=0.1)  # Increase spacing between heatmaps
plt.savefig("aspect_heatmap.png", dpi=300, bbox_inches="tight")  # Save before plt.show()
plt.show()



df = s_data_df
metric_columns = [col for col in df.columns if col not in ["category", "size10000", "aspect ratio","sizes bin"]]

# Melt the DataFrame
df_melted = df.melt(id_vars=["category", "size10000", "aspect ratio", "sizes bin" ], 
                     value_vars=metric_columns, 
                     var_name="Metric_Approach", 
                     value_name="Value")

df_melted["Metric"] = df_melted["Metric_Approach"].apply(lambda x: x.split()[0])
df_melted["Approach"] = df_melted["Metric_Approach"].apply(lambda x: x.split()[1])

# Convert 'Value' to numeric (handles possible non-numeric values)
df_melted["Value"] = pd.to_numeric(df_melted["Value"], errors="coerce")




df_melted = df_melted.drop(['Metric_Approach'],axis = 1)

sizes = df_melted.sort_values(by='sizes bin')['sizes bin'].unique()

# Create heatmaps for each category
num_categories = len(sizes)
fig, axes = plt.subplots(2,2, figsize=(28,20))
axes = axes.flatten()

if num_categories == 1:
    axes = [axes]  # Ensure axes is iterable for a single category

for i, size in enumerate(sizes):
    sub_df = df_melted[df_melted['sizes bin'] == size]
    pivoted = sub_df.groupby(["Metric", "Approach"])["Value"].mean().unstack()
    pivoted = pivoted.astype(float).fillna(0)  # Replace NaN values with 0
    pivoted = pivoted.reindex(index=metric_order, columns=approach_order)
    # Plot heatmap
    heatmap = sns.heatmap(pivoted, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=axes[i],vmin=0,vmax=1, annot_kws={"size": 22})  
    axes[i].set_title(f"size in 10k px: {size}", fontsize=20)
    axes[i].set_xlabel("", fontsize=16)
    axes[i].set_ylabel("", fontsize=16)
    
    # Rotate labels for better readability
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha="right",fontsize = 18)
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0,fontsize = 18)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("", fontsize=12)  # Set colorbar label font size
    colorbar.ax.tick_params(labelsize=20)  # Set colorbar tick label font size

plt.subplots_adjust(hspace=0.25)  # Increase spacing between heatmaps
plt.subplots_adjust(wspace=0.1)  # Increase spacing between heatmaps
plt.savefig("size_heatmap.png", dpi=300, bbox_inches="tight")  # Save before plt.show()
plt.show()
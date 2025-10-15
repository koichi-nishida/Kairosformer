import pandas as pd

# Load your merged CSV from the previous step
df = pd.read_csv("summary_all_datasets.csv")

# (optional) normalize dataset naming once here
df["Dataset"] = df["Dataset"].replace({"Exchange": "Exchange Rate"})

records = []
for (dataset, pred_len), g in df.groupby(["Dataset", "PredLen"]):
    # Best by MSE
    mse_row = g.loc[g["MSE"].idxmin()]
    # Best by MAE
    mae_row = g.loc[g["MAE"].idxmin()]
    # Fastest (shortest) total time
    t_row = g.loc[g["TotalTime(s)"].idxmin()]

    records.append({
        "Dataset": dataset,
        "PredLen": int(pred_len),

        "Best_MSE_Model": mse_row["Model"],
        "Best_MSE": round(float(mse_row["MSE"]), 6),

        "Best_MAE_Model": mae_row["Model"],
        "Best_MAE": round(float(mae_row["MAE"]), 6),

        "Best_Time_Model": t_row["Model"],
        "Best_Time(s)": round(float(t_row["TotalTime(s)"]), 2),
    })

out = pd.DataFrame(records).sort_values(["Dataset", "PredLen"])
out.to_csv("best_results_by_dataset_mse_mae_time.csv", index=False)

print("✅ Saved to best_results_by_dataset_mse_mae_time.csv")
print(out.head(12))

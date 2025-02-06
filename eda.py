import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(data_path):
    return pd.read_csv(data_path)

def data_cleaning(data):
    data["Model"] = data["Model"].str.replace("5-Sep", "9-5")
    data["Model"] = data["Model"].str.replace("3-Sep", "9-3")
    return data

if __name__ == "__main__":
    data_path = "data/car_data.csv"
    data = load_data(data_path)
    data = data_cleaning(data)

    os.makedirs("results/sales_distribution", exist_ok=True)
    os.makedirs("results/price_distribution", exist_ok=True)
    os.makedirs("results/price_engine", exist_ok=True)

    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y")

    brand_sales = data.groupby("Company")["Car_id"].count().reset_index().rename(columns={"Car_id": "Sales Count"})
    model_sales = data.groupby("Model")["Car_id"].count().reset_index().rename(columns={"Car_id": "Sales Count"})

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Sales Count", y="Company", data=brand_sales, hue="Company", dodge=False)
    plt.xlabel("Sales Count")
    plt.ylabel("Brand (Company)")
    plt.title("Sales Count per Brand")
    plt.legend([], [], frameon=False)
    plt.show()

    for model in model_sales["Model"].unique():
        model_data = data[data["Model"] == model]
        plt.figure(figsize=(10, 5))
        sns.histplot(model_data["Date"], bins=10, kde=True)
        plt.xlabel("Date")
        plt.ylabel("Frequency")
        plt.title(f"Sales Distribution for {model}")
        plt.savefig(f"results/sales_distribution/{model}.png")
        plt.close()

    for model in data["Model"].unique():
        model_data = data[data["Model"] == model]
        plt.figure(figsize=(10, 5))
        sns.histplot(model_data["Price ($)"], bins=10, kde=True)
        plt.xlabel("Price ($)")
        plt.ylabel("Frequency")
        plt.title(f"Price Distribution for {model}")
        plt.savefig(f"results/price_distribution/{model}.png")
        plt.close()

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["Weekday"] = data["Date"].dt.weekday
    data["Quarter"] = data["Date"].dt.quarter

    monthly_sales_trend = data.groupby(["Year", "Month"])["Car_id"].count().reset_index().rename(columns={"Car_id": "Sales Count"})
    monthly_sales_trend["Date"] = pd.to_datetime(monthly_sales_trend[["Year", "Month"]].assign(Day=1))

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Date", y="Sales Count", data=monthly_sales_trend, marker="o")
    plt.xlabel("Date")
    plt.ylabel("Sales Count")
    plt.title("Seasonal Trends in Car Sales")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=data["Price ($)"], ax=axes[0])
    axes[0].set_title("Price Outliers")
    sns.boxplot(y=data["Annual Income"], ax=axes[1])
    axes[1].set_title("Annual Income Outliers")
    plt.show()

    corr_features = ["Price ($)", "Annual Income", "Month", "Sales Count"]

    sales_per_model = data.groupby("Model")["Car_id"].count().reset_index().rename(columns={"Car_id": "Sales Count"})
    data = data.merge(sales_per_model, on="Model", how="left")

    corr_matrix = data[corr_features].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Key Features")
    plt.show()

    for model in data["Model"].unique():
        model_data = data[data["Model"] == model]
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x="Engine", y="Price ($)", data=model_data, hue="Transmission")
        plt.xlabel("Engine")
        plt.ylabel("Price ($)")
        plt.title(f"Price vs Engine for {model}")
        plt.savefig(f"results/price_engine/{model}.png")
        plt.close()

    
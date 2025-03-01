import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

plt.figure(figsize=(10, 5))
for a in df.columns[1:]:
    print(a)
    plt.plot(df["price"], df[a], marker="o", linestyle="-", color="b", label="Value")
    plt.xlabel("price")
    plt.ylabel(a)
    plt.title("CSV Data Plot")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.show()


# plt.xticks(rotation=45)  # Rotate x-axis labels for readability
# plt.show()


# # plt.plot(df["price"], df["area"], marker="o", linestyle="-", color="b", label="Value")
# # plt.plot(df["price"], df["bedrooms"], marker="o", linestyle="-", color="b", label="Value")


# plt.xlabel("price")
# plt.ylabel("area")
# plt.title("CSV Data Plot")
# plt.legend()
# plt.grid(True)


# plt.xticks(rotation=45)  # Rotate x-axis labels for readability
# plt.show()

corr_mat = df.corr()

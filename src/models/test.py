df = None
df["home"].groupby(
    df["home"]
).count(
).sort_values(
    ascending=False
).plot(
    kind="bar", color="#4C72B0", edgecolor="white"
)
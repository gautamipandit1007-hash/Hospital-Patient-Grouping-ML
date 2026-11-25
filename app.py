import joblib, pandas as pd, streamlit as st
b = joblib.load("bundle.joblib")
pre, km, caps, num_cols, cat_cols = b["pre"], b["km"], b["caps"], b["num_cols"], b["cat_cols"]
st.title("Hospital Patient Grouping")
st.warning("Analytical similarity grouping only. This is not a medical assessment.")
p = {"age": st.selectbox("Age group", b["age_groups"])}
for c in num_cols:
    if c != "age_mid":
        lo, hi, med = b["ranges"][c]
        p[c] = st.number_input(c, min_value=lo, max_value=hi, value=med)
for c in cat_cols:
    p[c] = st.selectbox(c, b["cat_options"][c])
if st.button("Find nearest cluster"):
    row = dict(p); row["age_mid"] = int(row.pop("age").strip("[)").split("-")[0]) + 5
    r = pd.DataFrame([row])
    for c in num_cols: r[c] = r[c].clip(upper=caps[c])
    x = pre.transform(r[num_cols + cat_cols])
    st.subheader(f"Nearest cluster: {int(km.predict(x)[0])}")
    st.write("Distances to cluster centres:", km.transform(x).round(2))

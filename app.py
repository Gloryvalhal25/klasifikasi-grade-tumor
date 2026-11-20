import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Klasifikasi Grade Tumor Otak", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load("model_grade_tumor.joblib")


bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]

st.title("Klasifikasi Grade Tumor Otak (HGG vs LGG)")
st.caption(f"Model: {bundle['model_name']} — ROC-AUC uji: {bundle['test_auc']:.3f}. "
           "Alat bantu penelitian, bukan alat diagnosis medis.")

tab_form, tab_csv = st.tabs(["Input manual", "Unggah CSV"])

with tab_form:
    with st.form("input"):
        row = {}
        for col in features:
            default = bundle["defaults"][col]
            if col in bundle["categories"]:
                opts = bundle["categories"][col]
                row[col] = st.selectbox(col, opts, index=opts.index(default) if default in opts else 0)
            else:
                row[col] = st.number_input(col, value=float(default), format="%.4f")
        submitted = st.form_submit_button("Prediksi")
    if submitted:
        x = pd.DataFrame([row], columns=features)
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        st.subheader(f"Prediksi: {pred}")
        st.write(pd.DataFrame({"Kelas": model.classes_, "Probabilitas": proba}))

with tab_csv:
    up = st.file_uploader("CSV dengan kolom fitur yang sama seperti data latih", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"Kolom tidak ditemukan: {missing}")
        else:
            out = df.copy()
            out["prediksi_grade"] = model.predict(df[features])
            proba = model.predict_proba(df[features])
            for i, cls in enumerate(model.classes_):
                out[f"prob_{cls}"] = proba[:, i].round(4)
            st.dataframe(out)
            st.download_button("Unduh hasil", out.to_csv(index=False), "hasil_prediksi.csv", "text/csv")

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import openai

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

############################################
# AI integration function
############################################
def get_ai_recommendation(prompt_text):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful business consultant."},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error obtaining AI recommendation: {e}"

############################################
# Main Streamlit app
############################################
def main():
    st.title("Enhanced Causal Inference App with AI-Powered Recommendations")
    st.write(""" 
    This app demonstrates three causal inference methods for pricing A/B testing and 
    provides **dynamic AI-generated** recommendations for each model:

    1. Difference-in-Differences (DiD)
    2. Synthetic Control (Naive)
    3. Propensity Score Matching (PSM)

    Simply upload your CSV, pick a method, and see the analysis + AI insights!
    """)

    # 1. Upload Data
    st.header("1. Upload Your CSV Data")
    uploaded_file = st.file_uploader("Upload a CSV file (format depends on the chosen method)", type="csv")
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
        return

    df = pd.read_csv(uploaded_file)
    # Attempt to parse date columns
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                st.error(f"Error converting '{col}' to datetime: {e}")

    st.subheader("Preview of Uploaded Data")
    st.write(df.head(10))
    st.write(f"Data shape: {df.shape}")
    
    # 2. Select a Causal Inference Method
    st.header("2. Select a Causal Inference Method")
    model_choice = st.selectbox(
        "Choose a method:",
        ["Difference-in-Differences (DiD)", "Synthetic Control (Naive)", "Propensity Score Matching (PSM)"]
    )

    ############################################
    # A) DIFFERENCE-IN-DIFFERENCES (DiD)
    ############################################
    if model_choice == "Difference-in-Differences (DiD)":
        st.info("**Expected Columns**: 'date', 'treatment' (0/1), 'post' (0/1), and an outcome (e.g., 'quantity_sold').")
        numeric_cols = [c for c in df.columns if c not in ["date", "treatment", "post"] and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            st.error("No numeric columns found for outcome. Please check your dataset.")
            return

        outcome_var = st.selectbox("Select Outcome Variable:", numeric_cols)
        st.subheader("Difference-in-Differences Model")

        formula = f"{outcome_var} ~ treatment + post + treatment:post"
        st.markdown(f"**Using formula:** `{formula}`")

        try:
            model = smf.ols(formula, data=df).fit()
            st.text(model.summary())
        except Exception as e:
            st.error(f"Error fitting DiD model: {e}")
            return
        
        # Build AI prompt from DiD results
        coefs = model.params
        pvals = model.pvalues
        did_coef = coefs.get("treatment:post", np.nan)
        did_p = pvals.get("treatment:post", 1.0)

        # Use triple quotes to avoid any line-break issues
        prompt_did = f"""
Here are the key DiD results:
Intercept = {coefs.get('Intercept', 0):.2f},
treatment = {coefs.get('treatment', 0):.2f},
post = {coefs.get('post', 0):.2f},
treatment:post = {did_coef:.2f} (p-value = {did_p:.3f}).

Please provide a non-technical business recommendation about whether to continue or change the price strategy, 
and suggest next steps.
"""
        ai_did = get_ai_recommendation(prompt_did)
        st.markdown(f"**AI Recommendation (DiD):** {ai_did}")

        # Parallel Trends Plot
        st.subheader("Parallel Trends Visualization")
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if not date_cols:
            st.error("No date column found for plotting.")
            return

        date_col = st.selectbox("Select the date column:", date_cols)
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        chosen_date = st.date_input("Select Intervention Date:", value=min_date, min_value=min_date, max_value=max_date)

        pre_df = df[df[date_col] < pd.to_datetime(chosen_date)]
        if pre_df.empty:
            st.warning("No pre-intervention data available to plot.")
        else:
            if "treatment" in df.columns:
                grouped_pre = pre_df.groupby([date_col, "treatment"])[outcome_var].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8,5))
                sns.lineplot(data=grouped_pre, x=date_col, y=outcome_var, hue="treatment", ax=ax)
                ax.set_title(f"Pre-Intervention Trends in {outcome_var}")
                ax.set_xlabel("Date")
                ax.set_ylabel(f"Average {outcome_var}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        ############################################
    # B) SYNTHETIC CONTROL (Naive)
    ############################################
    elif model_choice == "Synthetic Control (Naive)":
        st.info("**Expected Columns**: A 'unit' identifier, 'date', 'post' (0/1), and 'outcome'.")
        
        # Check if 'unit' column is present, otherwise allow the user to choose a column
        if "unit" not in df.columns:
            possible_unit_cols = [c for c in df.columns if (df[c].dtype == object or df[c].nunique() < 20)]
            if len(possible_unit_cols) == 0:
                st.error("No suitable column found for 'unit' identifier. Please check your dataset.")
                return
            unit_col = st.selectbox("Select the column to use as the unit identifier:", possible_unit_cols)
        else:
            unit_col = "unit"

        required_cols = [unit_col, "date", "post", "outcome"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Dataset is missing required columns: {missing}")
            return
        
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            st.error(f"Error converting 'date' to datetime: {e}")
            return

        all_units = df[unit_col].unique().tolist()
        treated_unit = st.selectbox("Select the treated unit:", all_units)

        pre_df = df[df["post"] == 0]
        treated_avg = pre_df.loc[pre_df[unit_col] == treated_unit, "outcome"].mean()

        donor_units = [u for u in all_units if u != treated_unit]
        donors_info = {}
        for u in donor_units:
            donors_info[u] = pre_df.loc[pre_df[unit_col] == u, "outcome"].mean()

        best_match_unit = None
        best_diff = float("inf")
        for u in donor_units:
            diff = abs(donors_info[u] - treated_avg)
            if diff < best_diff:
                best_diff = diff
                best_match_unit = u

        st.subheader("Naive Synthetic Control: Best-Matching Donor")
        if best_match_unit is None:
            st.error("No donor unit found. Please check your data.")
            return
        
        st.write(f"Treated unit: **{treated_unit}**")
        st.write(f"Best-matching donor (by pre-period average): **{best_match_unit}**")

        post_df = df[df["post"] == 1].copy()
        treated_post = post_df[post_df[unit_col] == treated_unit].copy()
        donor_post = post_df[post_df[unit_col] == best_match_unit].copy()
        merged = pd.merge(treated_post, donor_post, on="date", suffixes=("_treated", "_donor"))
        merged["gap"] = merged["outcome_treated"] - merged["outcome_donor"]

        # Date Range Filter
        st.subheader("Select Date Range for Synthetic Control Plots")
        min_date_val, max_date_val = merged["date"].min(), merged["date"].max()
        start_date, end_date = st.date_input(
            "Pick a date range:",
            value=(min_date_val, max_date_val),
            min_value=min_date_val,
            max_value=max_date_val
        )
        mask = (merged["date"] >= pd.to_datetime(start_date)) & (merged["date"] <= pd.to_datetime(end_date))
        plot_df = merged[mask].copy()

        st.subheader("Actual vs. Synthetic Outcome (Post-Intervention)")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(plot_df["date"], plot_df["outcome_treated"], label=f"{treated_unit} (Treated)", marker="o", linestyle="-", linewidth=2)
        ax.plot(plot_df["date"], plot_df["outcome_donor"], label=f"{best_match_unit} (Synthetic)", marker="s", linestyle="--", linewidth=2)
        ax.set_title("Actual Outcome vs. Synthetic Control Outcome", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Outcome", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Outcome Gap Over Time")
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(plot_df["date"], plot_df["gap"], label="Gap", color="red", marker="^", linestyle="-", linewidth=2)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1.5)
        ax2.set_title("Outcome Gap Over Time", fontsize=14)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Gap (Treated - Synthetic)", fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # AI Prompt for Synthetic Control
        avg_gap = merged["gap"].mean()
        prompt_sc = f"""
Synthetic Control Analysis:
- Treated unit: {treated_unit}
- Best-match donor: {best_match_unit}
- Average gap (treated - synthetic): {avg_gap:.2f}

Please provide a non-technical recommendation about whether the price change was beneficial,
and suggest next steps.
"""
        sc_recommendation = get_ai_recommendation(prompt_sc)
        st.markdown(f"**AI Recommendation (Synthetic Control):** {sc_recommendation}")

    ############################################
    # C) PROPENSITY SCORE MATCHING (PSM)
    ############################################
    elif model_choice == "Propensity Score Matching (PSM)":
        st.info("**Expected Columns**: 'treatment' (0/1), 'outcome', and at least one covariate (e.g., age, income).")
        if "treatment" not in df.columns or "outcome" not in df.columns:
            st.error("Dataset must include 'treatment' and 'outcome' columns.")
            return

        possible_covs = [c for c in df.columns if c not in ["treatment", "outcome"]]
        covariates = st.multiselect("Select covariates for matching:", possible_covs)
        if len(covariates) == 0:
            st.warning("No covariates selected. We'll do a direct comparison without matching.")
            do_direct = True
        else:
            do_direct = False

        treated_df = df[df["treatment"] == 1].copy()
        control_df = df[df["treatment"] == 0].copy()
        st.write(f"Treated count: {len(treated_df)} | Control count: {len(control_df)}")

        if do_direct:
            avg_treated = treated_df["outcome"].mean()
            avg_control = control_df["outcome"].mean()
            st.write(f"Avg Outcome (Treated): {avg_treated:.2f}")
            st.write(f"Avg Outcome (Control): {avg_control:.2f}")
            st.write(f"Difference: {(avg_treated - avg_control):.2f}")
            st.markdown(""" 
            **Business Interpretation:**
            This direct comparison shows the raw difference in outcomes between treated and control groups,
            but we haven't matched on covariates.
            """)
        else:
            X = df[covariates].values
            y = df["treatment"].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logit = LogisticRegression()
            logit.fit(X_scaled, y)
            df["propensity_score"] = logit.predict_proba(X_scaled)[:,1]

            treated_df = df[df["treatment"] == 1].copy()
            control_df = df[df["treatment"] == 0].copy()

            st.subheader("Propensity Score Distribution")
            fig, ax = plt.subplots(figsize=(8,5))
            sns.kdeplot(treated_df["propensity_score"], label="Treated", shade=True, ax=ax)
            sns.kdeplot(control_df["propensity_score"], label="Control", shade=True, ax=ax)
            ax.set_title("Propensity Score Distribution")
            st.pyplot(fig)

            control_scores = control_df["propensity_score"].values.reshape(-1,1)
            nn = NearestNeighbors(n_neighbors=1).fit(control_scores)
            distances, indices = nn.kneighbors(treated_df["propensity_score"].values.reshape(-1,1))

            matched_treated = []
            matched_control = []
            for i, row in treated_df.iterrows():
                t_outcome = row["outcome"]
                idx = indices[treated_df.index.get_loc(i)][0]
                control_index = control_df.iloc[idx].name
                c_outcome = control_df.loc[control_index, "outcome"]
                matched_treated.append(t_outcome)
                matched_control.append(c_outcome)

            matched_treated = np.array(matched_treated)
            matched_control = np.array(matched_control)
            ATT = matched_treated.mean() - matched_control.mean()

            st.subheader("Matching Results")
            st.write(f"Avg Outcome (Treated, matched): {matched_treated.mean():.2f}")
            st.write(f"Avg Outcome (Control, matched): {matched_control.mean():.2f}")
            st.write(f"Estimated ATT (Treated - Control): **{ATT:.2f}**")

            # AI Prompt for PSM
            prompt_psm = f"""
Propensity Score Matching result:
- ATT = {ATT:.2f}

Please provide a non-technical recommendation about whether to continue or change the price strategy,
based on this PSM result.
"""
            psm_recommendation = get_ai_recommendation(prompt_psm)
            st.markdown(f"**AI Recommendation (PSM):** {psm_recommendation}")

    st.markdown("---\n**Thank you for using the Enhanced Causal Inference App!**")

if __name__ == "__main__":
    main()

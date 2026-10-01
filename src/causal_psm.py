import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist

class PropensityScoreMatcher:
    def __init__(self, caliper: float = 0.25):
        self.caliper = caliper
        self.ps_model = LogisticRegression(C=1e3, max_iter=1000)

    def fit_predict_ps(self, df: pd.DataFrame, treatment_col: str, feature_cols: list) -> pd.DataFrame:
        df = df.copy()
        X = df[feature_cols]
        y = df[treatment_col]
        self.ps_model.fit(X, y)
        df['propensity_score'] = self.ps_model.predict_proba(X)[:, 1]
        eps = 1e-6
        ps_clipped = np.clip(df['propensity_score'], eps, 1 - eps)
        df['ps_logit'] = np.log(ps_clipped / (1 - ps_clipped))
        return df

    def match(self, df: pd.DataFrame, treatment_col: str, feature_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_ps = self.fit_predict_ps(df, treatment_col, feature_cols)
        treated = df_ps[df_ps[treatment_col] == 1].copy().reset_index(drop=True)
        control = df_ps[df_ps[treatment_col] == 0].copy().reset_index(drop=True)
        
        ps_logit_std = df_ps['ps_logit'].std()
        max_distance = self.caliper * ps_logit_std

        dist_matrix = cdist(treated[['ps_logit']].values, control[['ps_logit']].values, metric='euclidean')
        matched_t_idx, matched_c_idx = [], []
        available_control_idx = set(range(len(control)))

        for t_idx in range(len(treated)):
            distances = dist_matrix[t_idx, :].copy()
            mask = np.array([idx not in available_control_idx for idx in range(len(control))])
            distances[mask] = np.inf
            min_idx = np.argmin(distances)
            
            if distances[min_idx] <= max_distance:
                matched_t_idx.append(t_idx)
                matched_c_idx.append(min_idx)
                available_control_idx.remove(min_idx)

        matched_treated = treated.iloc[matched_t_idx]
        matched_control = control.iloc[matched_c_idx]
        matched_df = pd.concat([matched_treated, matched_control], axis=0).reset_index(drop=True)
        balance_df = self._compute_balance(df_ps, matched_df, treatment_col, feature_cols)
        return matched_df, balance_df

    def _compute_balance(self, unmatched_df: pd.DataFrame, matched_df: pd.DataFrame, treatment_col: str, feature_cols: list) -> pd.DataFrame:
        balance_records = []
        for col in feature_cols + ['propensity_score']:
            smd_unmatched = self._smd(
                unmatched_df[unmatched_df[treatment_col] == 1][col],
                unmatched_df[unmatched_df[treatment_col] == 0][col]
            )
            smd_matched = self._smd(
                matched_df[matched_df[treatment_col] == 1][col],
                matched_df[matched_df[treatment_col] == 0][col]
            )
            balance_records.append({
                'covariate': col,
                'smd_unmatched': smd_unmatched,
                'smd_matched': smd_matched,
                'balanced_post_match': abs(smd_matched) < 0.10
            })
        return pd.DataFrame(balance_records)

    @staticmethod
    def _smd(g1: pd.Series, g2: pd.Series) -> float:
        m_diff = g1.mean() - g2.mean()
        p_std = np.sqrt((g1.var() + g2.var()) / 2.0)
        return m_diff / p_std if p_std > 0 else 0.0

def run_psm_analysis(db_path: str):
    conn = sqlite3.connect(db_path)
    sql = """
    SELECT m.customer_id, m.treatment_group, m.pre_policy_activity, m.post_policy_spend, c.initial_credit_score
    FROM marketing_campaigns m JOIN customers c ON m.customer_id = c.customer_id;
    """
    df = pd.read_sql_query(sql, conn)
    conn.close()

    matcher = PropensityScoreMatcher(caliper=0.25)
    matched_df, balance_df = matcher.match(df, 'treatment_group', ['pre_policy_activity', 'initial_credit_score'])

    naive_att = df[df['treatment_group'] == 1]['post_policy_spend'].mean() - df[df['treatment_group'] == 0]['post_policy_spend'].mean()
    psm_att = matched_df[matched_df['treatment_group'] == 1]['post_policy_spend'].mean() - matched_df[matched_df['treatment_group'] == 0]['post_policy_spend'].mean()

    return {
        'naive_att': naive_att,
        'psm_att': psm_att,
        'matched_pairs': len(matched_df) // 2,
        'balance_diagnostics': balance_df
    }
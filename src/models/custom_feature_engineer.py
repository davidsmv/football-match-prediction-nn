import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, min_referee_matches=20):
        self.min_referee_matches = min_referee_matches
        self.frequent_referees_ = set()

    def _normalize_column_names(self, X):
        X = X.copy()
        X.columns = (
            X.columns.str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
            .str.strip("_")
        )
        return X

    def _add_season_feature(self, X):
        X = X.copy()
        date_part = pd.to_datetime(X["date"], errors="coerce")
        years = date_part.dt.year
        months = date_part.dt.month
        start_year = np.where(months >= 8, years, years - 1)
        X["season"] = start_year
        return X

    def _dates_to_numeric(self, X):
        X = X.copy()
        date_part = pd.to_datetime(X["date"], errors="coerce")
        X["weekday"] = date_part.dt.weekday + 1
        X["month"] = date_part.dt.month
        X["hour"] = pd.to_numeric(X["time"].str[:2], errors="coerce")
        return X

    def _one_hot_encode_column(self, X, column_name, prefix=None):
        X = X.copy()

        if column_name not in X.columns:
            return X

        dummies = pd.get_dummies(
            X[column_name],
            prefix=prefix or column_name,
            dtype=int
        )

        X = X.drop(columns=[column_name])
        X = pd.concat([X, dummies], axis=1)
        return X

    def _add_cyclical_feature(
        self, X, column_name, period, drop_original=True
    ):
        X = X.copy()

        if column_name not in X.columns:
            return X

        values = pd.to_numeric(X[column_name], errors="coerce")
        radians = 2 * np.pi * (values / period)

        X[f"{column_name}_sin"] = np.sin(radians)
        X[f"{column_name}_cos"] = np.cos(radians)

        if drop_original:
            X = X.drop(columns=[column_name])

        return X

    def _fit_referee_encoder(self, X):
        if "referee" not in X.columns:
            self.frequent_referees_ = set()
            return

        referee_counts = X["referee"].fillna("Other").value_counts()
        self.frequent_referees_ = set(
            referee_counts[referee_counts >= self.min_referee_matches].index
        )

    def _encode_referee(self, X):
        X = X.copy()

        if "referee" not in X.columns:
            return X

        # Rare referees (below min_referee_matches) are grouped into "Other"
        X["referee"] = X["referee"].fillna("Other").where(
            X["referee"].isin(self.frequent_referees_), other="Other"
        )
        X = self._one_hot_encode_column(X, "referee", prefix="referee")
        return X
    
    def add_venue_form_last_n(
        self,
        X: pd.DataFrame,
        n: int = 5,
        outcome_col: str = "home_outcome",
        season_col: str = "season",
        home_col: str = "home",
        away_col: str = "away",
    ) -> pd.DataFrame:
        """Add venue-specific rolling form columns to *df* and return it.

        Columns added:
        - home_team_home_matches_form_balance_last_n
        - away_team_away_matches_form_balance_last_n
        """
        out = X.copy()
        out[f"home_team_home_matches_form_balance_last_{n}"] = (
            out.groupby([season_col, home_col])[outcome_col]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
            .fillna(0)
        )
        out[f"away_team_away_matches_form_balance_last_{n}"] = (
            out.groupby([season_col, away_col])[outcome_col]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
            .fillna(0)
            .mul(-1)
        )
        return out
    
    def add_team_form_last_n(
        self,
        X: pd.DataFrame,
        n: int = 5,
        outcome_col: str = "home_outcome",
        season_col: str = "season",
        date_col: str = "date",
        time_col: str = "time",
        home_col: str = "home",
        away_col: str = "away",
    ) -> pd.DataFrame:
        out = X.copy()

        # Stable match id for merging features back
        out["_match_id"] = out.index

        # Build sortable kickoff datetime (date + time)
        date_part = pd.to_datetime(out[date_col], errors="coerce")
        time_part = pd.to_timedelta(
            out[time_col].fillna("00:00") + ":00",
            errors="coerce",
        )
        out["_kickoff_dt"] = date_part + time_part.fillna(pd.Timedelta(0))

        # Team-centric history table:
        # - home team keeps home_outcome
        # - away team gets inverse outcome

        home_hist_cols = [
            "_match_id", season_col, "_kickoff_dt", home_col, outcome_col
        ]
        away_hist_cols = [
            "_match_id", season_col, "_kickoff_dt", away_col, outcome_col
        ]

        home_hist = out[home_hist_cols].rename(
            columns={home_col: "team", outcome_col: "team_result"}
        )
        away_hist = out[away_hist_cols].rename(
            columns={away_col: "team", outcome_col: "team_result"}
        )
        away_hist["team_result"] = -away_hist["team_result"]

        team_hist = pd.concat([home_hist, away_hist], ignore_index=True)
        team_hist = team_hist.sort_values(
            [season_col, "team", "_kickoff_dt", "_match_id"]
        )

        # Overall team form: all venues (home and away matches),
        # excluding current match
        team_hist[f"overall_form_balance_last_{n}"] = (
            team_hist.groupby([season_col, "team"])["team_result"]
            .transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum())
            .fillna(0)
        )

        # Merge back: one overall value for home team and one for away team
        base_cols = ["_match_id", "team", f"overall_form_balance_last_{n}"]

        home_form = team_hist[base_cols].rename(
            columns={
                "team": home_col,
                f"overall_form_balance_last_{n}": f"home_team_overall_form_balance_last_{n}",
            }
        )
        away_form = team_hist[base_cols].rename(
            columns={
                "team": away_col,
                f"overall_form_balance_last_{n}": f"away_team_overall_form_balance_last_{n}",
            }
        )

        out = out.merge(home_form, on=["_match_id", home_col], how="left")
        out = out.merge(away_form, on=["_match_id", away_col], how="left")

        return out.drop(columns=["_match_id", "_kickoff_dt"])

    def _drop_unnecessary_columns(self, X):
        X = X.copy()
        # TODO check if one of them are neccesary for the other featues.
        columns_to_drop = [
            "weekday",
            "match_report",
            "venue",  #  Because the home team is included so it will be redundant and cause multicollinearity
            "time",
            "date",
        ]
        existing_cols_to_drop = [
            col for col in columns_to_drop if col in X.columns
        ]
        return X.drop(columns=existing_cols_to_drop)

    def fit(self, X, y=None):
        X = self._normalize_column_names(X)
        self._fit_referee_encoder(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = self._add_season_feature(X)
        X = self._dates_to_numeric(X)
        X = self._add_cyclical_feature(X, "month", period=12, drop_original=True)
        X = self._add_cyclical_feature(X, "hour", period=24, drop_original=True)
        X = self._one_hot_encode_column(X, "day")
        X = self._encode_referee(X)
        X = self._normalize_column_names(X)
        # X = self._drop_unnecessary_columns(X)
        return X

        # TODO create the previous features here

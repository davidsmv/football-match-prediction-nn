import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, rare_threshold_percentile=25):
        self.rare_threshold_percentile = rare_threshold_percentile
        self.frequent_referees_ = set()
        self.frequent_teams_ = set()
        self.min_referee_matches_ = None
        self.min_team_matches_ = None
        self.out = pd.DataFrame()  # Store intermediate results for debugging

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

    def _fit_team_encoder(self, X):
        # Count appearances across both home and away columns
        home_counts = X["home"].fillna("Other").value_counts()
        away_counts = X["away"].fillna("Other").value_counts()
        total_counts = home_counts.add(away_counts, fill_value=0)
        # Threshold = percentile of the count distribution
        self.min_team_matches_ = np.percentile(
            total_counts.values, self.rare_threshold_percentile
        )
        self.frequent_teams_ = set(
            total_counts[total_counts >= self.min_team_matches_].index
        )

    def _encode_teams(self, X):
        X = X.copy()
        # Rare teams (below min_team_matches) are grouped into "Other"
        X["home"] = X["home"].fillna("Other").where(
            X["home"].isin(self.frequent_teams_), other="Other"
        )
        X["away"] = X["away"].fillna("Other").where(
            X["away"].isin(self.frequent_teams_), other="Other"
        )
        X = self._one_hot_encode_column(X, "home", prefix="home_team")
        X = self._one_hot_encode_column(X, "away", prefix="away_team")
        return X

    def _fit_referee_encoder(self, X):
        if "referee" not in X.columns:
            self.frequent_referees_ = set()
            return

        referee_counts = X["referee"].fillna("Other").value_counts()
        # Threshold = percentile of the count distribution
        self.min_referee_matches_ = np.percentile(
            referee_counts.values, self.rare_threshold_percentile
        )
        self.frequent_referees_ = set(
            referee_counts[referee_counts >= self.min_referee_matches_].index
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

    def _label_home_outcome(self, score):
        if pd.isna(score):
            return None
        try:
            home_score, away_score = map(int, score.split("–"))
            if home_score > away_score:
                return 1
            elif home_score < away_score:
                return -1
            else:
                return 0
        except Exception as e:
            print(f"Error encoding score '{score}': {e}")
            return None

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
        out["home_outcome"] = out["score"].apply(self._label_home_outcome)
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
        setattr(self, f"df_last_{n}", out)

    def add_columns_for_venue_form_last_n(self, X):
        # Get all the dataframes created by add_venue_form_last_n for different n values
        venue_form_attrs = [
            attr_name
            for attr_name in dir(self)
            if attr_name.startswith("df_last_")
        ]

        # Merge all the venue form dataframes on the common columns
        merged_df = X.copy()
        for attr_name in venue_form_attrs:
            n = int(attr_name.split("_")[-1])
            df_with_form = getattr(self, attr_name)
            home_form_col = f"home_team_home_matches_form_balance_last_{n}"
            away_form_col = f"away_team_away_matches_form_balance_last_{n}"

            merged_df = merged_df.merge(
                df_with_form[["season", "home", "away", home_form_col, away_form_col]],
                on=["season", "home", "away"],
                how="left"
            )

            # For future/unseen matches, fill with the most recent known form
            # for that team in that season from the training data
            if merged_df[home_form_col].isna().any():
                latest_home = (
                    df_with_form.sort_values("date")
                    .groupby(["season", "home"])[home_form_col]
                    .last()
                    .reset_index()
                    .rename(columns={home_form_col: "_fill"})
                )
                fill_vals = merged_df[["season", "home"]].merge(
                    latest_home, on=["season", "home"], how="left"
                )["_fill"].values
                merged_df[home_form_col] = merged_df[home_form_col].fillna(fill_vals)

            if merged_df[away_form_col].isna().any():
                latest_away = (
                    df_with_form.sort_values("date")
                    .groupby(["season", "away"])[away_form_col]
                    .last()
                    .reset_index()
                    .rename(columns={away_form_col: "_fill"})
                )
                fill_vals = merged_df[["season", "away"]].merge(
                    latest_away, on=["season", "away"], how="left"
                )["_fill"].values
                merged_df[away_form_col] = merged_df[away_form_col].fillna(fill_vals)

        return merged_df

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
        out["home_outcome"] = out["score"].apply(self._label_home_outcome)

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

        out = out.drop(columns=["_match_id", "_kickoff_dt"])
        setattr(self, f"df_team_form_last_{n}", out)

    def add_columns_for_team_form_last_n(self, X):
        # Get all the dataframes created by add_team_form_last_n for different n values
        team_form_attrs = [
            attr_name
            for attr_name in dir(self)
            if attr_name.startswith("df_team_form_last_")
        ]

        merged_df = X.copy()
        for attr_name in team_form_attrs:
            n = int(attr_name.split("_")[-1])
            df_with_form = getattr(self, attr_name)
            home_form_col = f"home_team_overall_form_balance_last_{n}"
            away_form_col = f"away_team_overall_form_balance_last_{n}"

            merged_df = merged_df.merge(
                df_with_form[["season", "home", "away", home_form_col, away_form_col]],
                on=["season", "home", "away"],
                how="left"
            )

            # For future/unseen matches, fill with the most recent known form
            # for that team in that season from the training data
            if merged_df[home_form_col].isna().any():
                latest_home = (
                    df_with_form.sort_values("date")
                    .groupby(["season", "home"])[home_form_col]
                    .last()
                    .reset_index()
                    .rename(columns={home_form_col: "_fill"})
                )
                fill_vals = merged_df[["season", "home"]].merge(
                    latest_home, on=["season", "home"], how="left"
                )["_fill"].values
                merged_df[home_form_col] = merged_df[home_form_col].fillna(fill_vals)

            if merged_df[away_form_col].isna().any():
                latest_away = (
                    df_with_form.sort_values("date")
                    .groupby(["season", "away"])[away_form_col]
                    .last()
                    .reset_index()
                    .rename(columns={away_form_col: "_fill"})
                )
                fill_vals = merged_df[["season", "away"]].merge(
                    latest_away, on=["season", "away"], how="left"
                )["_fill"].values
                merged_df[away_form_col] = merged_df[away_form_col].fillna(fill_vals)

        return merged_df

    def _drop_unnecessary_columns(self, X):
        X = X.copy()
        # TODO check if one of them are neccesary for the other featues.
        columns_to_drop = [
            "weekday",
            "match_report",
            # Because the home team is included so it will be redundant
            # and cause multicollinearity
            "venue",
            "time",
            "date",
            "score",
            "attendance",
        ]
        existing_cols_to_drop = [
            col for col in columns_to_drop if col in X.columns
        ]
        return X.drop(columns=existing_cols_to_drop)

    def fit(self, X, y=None):
        X = self._normalize_column_names(X)
        X = self._add_season_feature(X)
        self._fit_referee_encoder(X)
        self._fit_team_encoder(X)
        self.add_venue_form_last_n(X, n=1)
        self.add_team_form_last_n(X, n=1)
        self.add_venue_form_last_n(X, n=3)
        self.add_team_form_last_n(X, n=3)
        self.add_venue_form_last_n(X, n=5)
        self.add_team_form_last_n(X, n=5)
        self.add_venue_form_last_n(X, n=10)
        self.add_team_form_last_n(X, n=10)
        return self

    def transform(self, X):
        X = X.copy()
        X = self._normalize_column_names(X)
        X = self._add_season_feature(X)
        X = self.add_columns_for_venue_form_last_n(X)
        X = self.add_columns_for_team_form_last_n(X)
        X = self._dates_to_numeric(X)
        X = self._add_cyclical_feature(
            X, "month", period=12, drop_original=True)
        X = self._add_cyclical_feature(
            X, "hour", period=24, drop_original=True)
        X = self._one_hot_encode_column(X, "day")
        X = self._encode_referee(X)
        X = self._drop_unnecessary_columns(X)
        X = self._encode_teams(X)
        X = self._normalize_column_names(X)
        return X

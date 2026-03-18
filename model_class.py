import os
import json
import pickle
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
@dataclass
class ModelConfig:
    ticker: str = "BTC-USD"
    lookback: int = 120
    horizon_days: int = 366
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 120
    seed: int = 42


# =========================================================
# MODEL
# =========================================================
class BitcoinLSTMModel:
    """
    Direct multi-step BTC predictor.

    Idee:
    - Input: letzte `lookback` Tage Features
    - Output: nächste `horizon_days` Log-Returns auf einmal

    Vorteile gegenüber vorher:
    - kein Teacher Forcing im Forecast-Zeitraum
    - kein "echte Vergangenheit aus dem Testzeitraum" als Input
    - keine rekursive Preis-auf-Preis-Kette, die oft linear/glatt wird

    Einschränkung:
    - forecast Länge darf nicht größer sein als horizon_days
    """

    FEATURE_COLUMNS = [
        "LogReturn",
        "Volatility_7",
        "Volatility_30",
        "Momentum_7",
        "Momentum_30",
        "SMA_7_Ratio",
        "SMA_30_Ratio",
        "EMA_7_Ratio",
        "EMA_30_Ratio",
        "RSI_14",
    ]

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = None
        self.history_df = None
        self.feature_df = None

        self._set_seed(self.config.seed)

    # =====================================================
    # BASICS
    # =====================================================
    def _set_seed(self, seed: int):
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    @staticmethod
    def _normalize_date(date_str: str) -> pd.Timestamp:
        return pd.Timestamp(date_str).normalize()

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_log_returns(close_series: pd.Series) -> pd.Series:
        return np.log(close_series / close_series.shift(1))

    # =====================================================
    # DATA
    # =====================================================
    def download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        end_date ist inklusiv gemeint
        """
        start_ts = self._normalize_date(start_date)
        end_ts = self._normalize_date(end_date)

        df = yf.download(
            self.config.ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            multi_level_index=False
        )

        if df.empty:
            raise ValueError("Keine Daten von Yahoo Finance erhalten.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            raise ValueError("Spalte 'Close' fehlt.")

        df = df[["Close"]].dropna().copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()

        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["LogReturn"] = self._compute_log_returns(out["Close"])

        out["Volatility_7"] = out["LogReturn"].rolling(7).std()
        out["Volatility_30"] = out["LogReturn"].rolling(30).std()

        out["Momentum_7"] = out["Close"] / out["Close"].shift(7) - 1.0
        out["Momentum_30"] = out["Close"] / out["Close"].shift(30) - 1.0

        out["SMA_7"] = out["Close"].rolling(7).mean()
        out["SMA_30"] = out["Close"].rolling(30).mean()
        out["EMA_7"] = out["Close"].ewm(span=7, adjust=False).mean()
        out["EMA_30"] = out["Close"].ewm(span=30, adjust=False).mean()

        out["SMA_7_Ratio"] = out["Close"] / out["SMA_7"] - 1.0
        out["SMA_30_Ratio"] = out["Close"] / out["SMA_30"] - 1.0
        out["EMA_7_Ratio"] = out["Close"] / out["EMA_7"] - 1.0
        out["EMA_30_Ratio"] = out["Close"] / out["EMA_30"] - 1.0

        out["RSI_14"] = self._compute_rsi(out["Close"], 14) / 100.0

        out = out.drop(columns=["SMA_7", "SMA_30", "EMA_7", "EMA_30"])
        return out

    def set_history_data(self, df: pd.DataFrame):
        if "Close" not in df.columns:
            raise ValueError("DataFrame braucht die Spalte 'Close'.")

        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df[["Close"]].dropna().sort_index()

        min_rows = self.config.lookback + self.config.horizon_days + 50
        if len(df) < min_rows:
            raise ValueError(
                f"Zu wenig Daten. Benötigt werden grob mindestens {min_rows} Zeilen."
            )

        self.history_df = df
        feature_df = self._add_features(df)
        feature_df = feature_df.dropna().copy()
        self.feature_df = feature_df

    # =====================================================
    # TRAINING DATA
    # =====================================================
    def _build_training_samples(self):
        if self.feature_df is None:
            raise ValueError("Keine History gesetzt. Erst set_history_data(...) aufrufen.")

        df = self.feature_df.copy()

        x_list = []
        y_list = []
        target_start_dates = []

        feature_values = df[self.FEATURE_COLUMNS].values
        log_returns = df["LogReturn"].values
        dates = df.index

        lookback = self.config.lookback
        horizon = self.config.horizon_days

        # i = Ende des Input-Fensters
        # Target = i+1 bis i+horizon
        for i in range(lookback - 1, len(df) - horizon):
            x_seq = feature_values[i - lookback + 1:i + 1]
            y_seq = log_returns[i + 1:i + 1 + horizon]

            if np.isnan(x_seq).any() or np.isnan(y_seq).any():
                continue

            x_list.append(x_seq)
            y_list.append(y_seq)
            target_start_dates.append(dates[i + 1])

        if not x_list:
            raise ValueError("Keine Trainingssamples erzeugt.")

        x = np.array(x_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        target_start_dates = pd.to_datetime(target_start_dates)

        return x, y, target_start_dates

    def prepare_training_data(self, train_end_date: str):
        """
        train_end_date = letzter Tag, den das Modell kennen darf.
        Danach darf nichts mehr als Input ins Training.
        """
        if self.feature_df is None:
            raise ValueError("Keine History gesetzt. Erst set_history_data(...) aufrufen.")

        train_end_ts = self._normalize_date(train_end_date)

        x_all, y_all, target_start_dates = self._build_training_samples()

        # Nur Samples behalten, deren kompletter Forecast-Horizont noch innerhalb Training liegt
        horizon = self.config.horizon_days
        target_end_dates = target_start_dates + pd.to_timedelta(horizon - 1, unit="D")

        mask = target_end_dates <= train_end_ts

        x_train = x_all[mask]
        y_train = y_all[mask]

        if len(x_train) == 0:
            raise ValueError("Keine Trainingsdaten nach train_end_date-Filter.")

        n_features = x_train.shape[2]

        self.scaler = StandardScaler()
        self.scaler.fit(x_train.reshape(-1, n_features))
        x_train_scaled = self.scaler.transform(
            x_train.reshape(-1, n_features)
        ).reshape(x_train.shape)

        return x_train_scaled, y_train

    # =====================================================
    # MODEL
    # =====================================================
    def build_model(self):
        n_features = len(self.FEATURE_COLUMNS)
        horizon = self.config.horizon_days

        model = Sequential([
            Input(shape=(self.config.lookback, n_features)),

            LSTM(128, return_sequences=True),
            Dropout(0.2),

            LSTM(96, return_sequences=True),
            Dropout(0.2),

            LSTM(64),
            Dropout(0.2),

            Dense(128, activation="relu"),
            Dropout(0.1),

            Dense(64, activation="relu"),
            Dense(horizon)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=Huber(),
            metrics=["mae"]
        )

        self.model = model
        return model

    def train(
        self,
        train_end_date: str,
        validation_split: float = 0.1,
        save_best_path: str | None = None
    ):
        x_train, y_train = self.prepare_training_data(train_end_date=train_end_date)

        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        if save_best_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_best_path,
                    monitor="val_loss",
                    save_best_only=True
                )
            )

        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=validation_split,
            shuffle=False,
            callbacks=callbacks,
            verbose=1
        )

        return history

    # =====================================================
    # SAVE / LOAD
    # =====================================================
    def save(self, folder_path: str):
        if self.model is None:
            raise ValueError("Kein Modell vorhanden.")
        if self.scaler is None:
            raise ValueError("Kein Scaler vorhanden.")

        os.makedirs(folder_path, exist_ok=True)

        model_path = os.path.join(folder_path, "model.keras")
        scaler_path = os.path.join(folder_path, "scaler.pkl")
        config_path = os.path.join(folder_path, "config.json")

        self.model.save(model_path)

        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, folder_path: str):
        model_path = os.path.join(folder_path, "model.keras")
        scaler_path = os.path.join(folder_path, "scaler.pkl")
        config_path = os.path.join(folder_path, "config.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(scaler_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        instance = cls(ModelConfig(**config_dict))
        instance.model = load_model(model_path)

        with open(scaler_path, "rb") as f:
            instance.scaler = pickle.load(f)

        return instance

    # =====================================================
    # INTERNAL FORECAST
    # =====================================================
    def _predict_horizon_from_history_end(self, history_end_date: str) -> pd.DataFrame:
        """
        history_end_date = letzter echter Tag, den das Modell kennen darf
        Forecast startet am Folgetag und geht horizon_days weit
        """
        if self.model is None:
            raise ValueError("Kein Modell geladen/trainiert.")
        if self.scaler is None:
            raise ValueError("Kein Scaler geladen/trainiert.")
        if self.feature_df is None or self.history_df is None:
            raise ValueError("Keine History gesetzt.")

        history_end_ts = self._normalize_date(history_end_date)

        feature_df = self.feature_df.loc[:history_end_ts].copy()
        close_df = self.history_df.loc[:history_end_ts].copy()

        if len(feature_df) < self.config.lookback:
            raise ValueError("Zu wenig History für Forecast.")

        last_input = feature_df[self.FEATURE_COLUMNS].tail(self.config.lookback).values
        last_input_scaled = self.scaler.transform(last_input).reshape(
            1, self.config.lookback, len(self.FEATURE_COLUMNS)
        )

        pred_log_returns = self.model.predict(last_input_scaled, verbose=0)[0]

        prev_close = float(close_df["Close"].iloc[-1])

        forecast_dates = pd.date_range(
            start=history_end_ts + pd.Timedelta(days=1),
            periods=self.config.horizon_days,
            freq="D"
        )

        predicted_closes = []
        current_close = prev_close

        for r in pred_log_returns:
            current_close = float(current_close * np.exp(r))
            predicted_closes.append(current_close)

        forecast_df = pd.DataFrame(
            {
                "Predicted_Log_Return": pred_log_returns,
                "Predicted_Close": predicted_closes
            },
            index=forecast_dates
        )

        return forecast_df

    # =====================================================
    # PUBLIC PREDICT
    # =====================================================
    def predict_range(self, forecast_start_date: str, forecast_end_date: str) -> pd.DataFrame:
        """
        Echte Forecast-Funktion:
        - nutzt nur Daten bis Tag vor forecast_start_date
        - prognostiziert dann direkt den ganzen Bereich
        - wenn echte Werte existieren, werden sie nur zum Vergleich angehängt
        """
        if self.history_df is None:
            raise ValueError("Keine History gesetzt. Erst set_history_data(...) aufrufen.")

        forecast_start_ts = self._normalize_date(forecast_start_date)
        forecast_end_ts = self._normalize_date(forecast_end_date)

        if forecast_end_ts < forecast_start_ts:
            raise ValueError("forecast_end_date muss nach forecast_start_date liegen.")

        forecast_len = (forecast_end_ts - forecast_start_ts).days + 1
        if forecast_len > self.config.horizon_days:
            raise ValueError(
                f"forecast Bereich zu lang ({forecast_len} Tage). "
                f"Maximal unterstützt: {self.config.horizon_days}."
            )

        history_end_ts = forecast_start_ts - pd.Timedelta(days=1)

        if self.history_df.index[-1] < history_end_ts:
            raise ValueError(
                f"History endet bei {self.history_df.index[-1].date()}, "
                f"benötigt wird mindestens {history_end_ts.date()}."
            )

        horizon_df = self._predict_horizon_from_history_end(history_end_ts.strftime("%Y-%m-%d"))
        pred_df = horizon_df.loc[forecast_start_ts:forecast_end_ts].copy()

        # echte Werte nur zum Vergleich hinzufügen, falls vorhanden
        actual_close = self.history_df["Close"].reindex(pred_df.index)
        if actual_close.notna().any():
            pred_df["Actual_Close"] = actual_close

        return pred_df


# =========================================================
# PLOTTER
# =========================================================
class BitcoinLSTMPlotter:
    @staticmethod
    def plot_training_history(history):
        plt.figure(figsize=(12, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_prediction_vs_actual(pred_df: pd.DataFrame, title: str = "Prediction vs Real Price"):
        if "Actual_Close" not in pred_df.columns:
            raise ValueError("pred_df braucht 'Actual_Close' für diesen Plot.")

        df = pred_df.dropna(subset=["Actual_Close"]).copy()

        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df["Actual_Close"], label="Real Price", linewidth=2)
        plt.plot(df.index, df["Predicted_Close"], label="Prediction", linewidth=2)

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("BTC Close Price (USD)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_history_prediction(
        history_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        plot_from_date: str,
        forecast_start_date: str,
        title: str = "BTC History + Prediction"
    ):
        plot_from_ts = pd.Timestamp(plot_from_date).normalize()
        forecast_start_ts = pd.Timestamp(forecast_start_date).normalize()

        history_part = history_df.loc[
            (history_df.index >= plot_from_ts) & (history_df.index < forecast_start_ts)
        ].copy()

        plt.figure(figsize=(14, 7))

        if not history_part.empty:
            plt.plot(
                history_part.index,
                history_part["Close"],
                label="History",
                color="blue",
                linewidth=2
            )

        plt.plot(
            pred_df.index,
            pred_df["Predicted_Close"],
            label="Prediction",
            color="red",
            linewidth=2
        )

        if "Actual_Close" in pred_df.columns and pred_df["Actual_Close"].notna().any():
            actual_df = pred_df.dropna(subset=["Actual_Close"]).copy()
            plt.plot(
                actual_df.index,
                actual_df["Actual_Close"],
                label="Real Price",
                color="green",
                linewidth=2
            )

        plt.axvline(
            x=forecast_start_ts,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="Prediction starts here"
        )

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("BTC Close Price (USD)")
        plt.xlim(left=plot_from_ts)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error as mape
import optuna
from prophet import Prophet
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
import warnings
import os
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
import matplotlib

# Set Japanese font for matplotlib
matplotlib.rcParams['font.family'] = 'IPAexGothic'

# Error messages
ERROR_MESSAGES = {
    'file_empty': 'ファイルが空です。データを確認してください。',
    'invalid_format': '無効なファイル形式です。Excelファイル(.xlsx)を使用してください。',
    'no_data': 'データが見つかりません。',
    'date_error': '日付の形式が正しくありません。',
    'quantity_error': '数量データが不正です。',
    'forecast_error': '予測処理中にエラーが発生しました。',
    'optimization_error': 'パラメータ最適化中にエラーが発生しました。'
}

# Configure logging
logging.basicConfig(filename='error_log.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Class to manage model configuration
class ModelConfig:
    def __init__(self, data_length, n_trials):
        self.n_trials = self._calculate_trials(data_length, n_trials)
    
    def _calculate_trials(self, data_length, n_trials):
        # Adjust the number of trials based on the data size and user input
        return min(n_trials, max(10, data_length // 10))

# Function to clean and prepare data
def clean_data(file):
    try:
        # File check
        if file is None:
            raise ValueError(ERROR_MESSAGES['file_empty'])
        
        # File size check
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError('ファイルサイズが大きすぎます。10MB以下のファイルを使用してください。')
        
        # Save as a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        try:
            df = pd.read_excel(tmp_path, header=None, usecols=[0, 1])
        finally:
            os.unlink(tmp_path)  # Ensure temporary file is deleted
        
        if df.empty:
            raise ValueError(ERROR_MESSAGES['no_data'])

        # Check column names and data types
        if df.shape[1] != 2:
            raise ValueError('データの列数が正しくありません。2列のデータを使用してください。')
        
        df.columns = ['Date', 'Quantity']

        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isna().any():
            raise ValueError(ERROR_MESSAGES['date_error'])

        # Convert quantity column
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        if df['Quantity'].isna().any():
            raise ValueError(ERROR_MESSAGES['quantity_error'])

        # Handle missing values - Provide options for imputation
        imputation_method = st.sidebar.selectbox("欠損値補完方法を選択", ['平均', '中央値', '線形補間'])
        if imputation_method == '平均':
            imputer = SimpleImputer(strategy='mean')
        elif imputation_method == '中央値':
            imputer = SimpleImputer(strategy='median')
        elif imputation_method == '線形補間':
            df['Quantity'] = df['Quantity'].interpolate(method='linear')
        else:
            raise ValueError("不明な補完方法です。")
        
        if imputation_method != '線形補間':
            df['Quantity'] = imputer.fit_transform(df[['Quantity']])

        # Remove outliers using IQR method
        Q1 = df['Quantity'].quantile(0.25)
        Q3 = df['Quantity'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['Quantity'] >= Q1 - 1.5 * IQR) & (df['Quantity'] <= Q3 + 1.5 * IQR)]
        
        # Aggregate data by date in case of duplicate dates
        df = df.groupby('Date').sum().reset_index()
        
        # Sort by date and reset index
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        logging.error(f'データクリーニング中にエラーが発生しました: {str(e)}')
        raise Exception(f'データクリーニング中にエラーが発生しました: {str(e)}')

# Function to analyze explanatory variables
def analyze_explanatory_variables(df):
    try:
        # Monthly sales trend
        df['Month'] = df['Date'].dt.month
        monthly_trend = df.groupby('Month')['Quantity'].mean()

        # Weekly sales trend
        df['Weekday'] = df['Date'].dt.weekday
        weekday_trend = df.groupby('Weekday')['Quantity'].mean()
        
        # Get holidays using USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df['Date'].min(), end=df['Date'].max())
        df['Holiday'] = df['Date'].isin(holidays).astype(int)
        
        return df, monthly_trend, weekday_trend
    
    except Exception as e:
        logging.error(f'説明変数の分析中にエラーが発生しました: {str(e)}')
        raise Exception(f'説明変数の分析中にエラーが発生しました: {str(e)}')

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    try:
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[-min_length:]
        y_pred = y_pred[-min_length:]
        metrics = {
            'MAPE': mape(y_true, y_pred),
            'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'MAE': np.mean(np.abs(y_true - y_pred))
        }
        return metrics
    except Exception as e:
        logging.error(f'モデル評価中にエラーが発生しました: {str(e)}')
        raise Exception(f'モデル評価中にエラーが発生しました: {str(e)}')

# Function to forecast using ARIMA, SARIMA, Prophet
def forecast(df, model_type, seasonal, forecast_period, config):
    warnings.filterwarnings('ignore')  # Suppress warnings
    
    try:
        if len(df) < 30:
            raise ValueError('予測には最低30日分のデータが必要です。')

        if model_type == 'SARIMA':
            # Hyperparameter tuning with Optuna for SARIMA
            def objective(trial):
                try:
                    p = trial.suggest_int('p', 1, 3)
                    d = trial.suggest_int('d', 0, 2)
                    q = trial.suggest_int('q', 1, 3)
                    P = trial.suggest_int('P', 0, 3)
                    D = trial.suggest_int('D', 0, 2)
                    Q = trial.suggest_int('Q', 0, 3)
                    s = trial.suggest_categorical('s', [7, 12])

                    model = SARIMAX(df['Quantity'], order=(p, d, q), seasonal_order=(P, D, Q, s),
                                    exog=df[['Month', 'Weekday', 'Holiday']])
                    model_fit = model.fit(disp=False)
                    forecast = model_fit.predict(start=0, end=len(df) - 1, exog=df[['Month', 'Weekday', 'Holiday']])
                    error = mape(df['Quantity'], forecast)
                    
                    # Error validity check
                    if np.isnan(error) or np.isinf(error) or error > 100:
                        return float('inf')
                    return error
                except:
                    return float('inf')

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=config.n_trials, catch=(ValueError,))

            if study.best_value == float('inf'):
                raise ValueError(ERROR_MESSAGES['optimization_error'])

            best_params = study.best_params
            model = SARIMAX(df['Quantity'], order=(best_params['p'], best_params['d'], best_params['q']),
                            seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], best_params['s']),
                            exog=df[['Month', 'Weekday', 'Holiday']])
        elif model_type == 'ARIMA':
            # Hyperparameter tuning with Optuna for ARIMA
            def objective(trial):
                try:
                    p = trial.suggest_int('p', 1, 3)
                    d = trial.suggest_int('d', 0, 2)
                    q = trial.suggest_int('q', 1, 3)

                    model = ARIMA(df['Quantity'], order=(p, d, q), exog=df[['Month', 'Weekday', 'Holiday']])
                    model_fit = model.fit()
                    forecast = model_fit.predict(start=0, end=len(df) - 1, exog=df[['Month', 'Weekday', 'Holiday']])
                    error = mape(df['Quantity'], forecast)
                    
                    if np.isnan(error) or np.isinf(error) or error > 100:
                        return float('inf')
                    return error
                except:
                    return float('inf')

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=config.n_trials, catch=(ValueError,))

            if study.best_value == float('inf'):
                raise ValueError(ERROR_MESSAGES['optimization_error'])

            best_params = study.best_params
            model = ARIMA(df['Quantity'], order=(best_params['p'], best_params['d'], best_params['q']),
                          exog=df[['Month', 'Weekday', 'Holiday']])
        elif model_type == 'Prophet':
            # Prophet model
            df_prophet = df.rename(columns={'Date': 'ds', 'Quantity': 'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.add_regressor('Month')
            model.add_regressor('Weekday')
            model.add_regressor('Holiday')
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=forecast_period)
            future['Month'] = future['ds'].dt.month
            future['Weekday'] = future['ds'].dt.weekday
            future['Holiday'] = future['ds'].isin(df['Date']).astype(int)
            forecast = model.predict(future)
            forecast_values = forecast['yhat'][-forecast_period:]
        else:
            raise ValueError("不明なモデルタイプです。")
        
        # Fit the model for SARIMA and ARIMA
        if model_type in ['SARIMA', 'ARIMA']:
            model_fit = model.fit()
            # Forecast
            future_exog = pd.DataFrame({
                'Month': [(df['Date'].iloc[-1] + datetime.timedelta(days=i)).month for i in range(1, forecast_period + 1)],
                'Weekday': [(df['Date'].iloc[-1] + datetime.timedelta(days=i)).weekday() for i in range(1, forecast_period + 1)],
                'Holiday': [0] * forecast_period  # Assume no holidays in forecast period
            })
            forecast_values = model_fit.forecast(steps=forecast_period, exog=future_exog)
        
        # Forecast validity check
        if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
            raise ValueError('無効な予測値が生成されました。')
        
        if np.any(forecast_values < 0):
            forecast_values[forecast_values < 0] = 0

        return forecast_values

    except Exception as e:
        logging.error(f'予測処理中にエラーが発生しました: {str(e)}')
        raise Exception(f'予測処理中にエラーが発生しました: {str(e)}')

# Streamlit Application
def main():
    st.set_page_config(page_title="需要予測アプリ", layout="wide")
    st.title("需要予測アプリ (ARIMA/SARIMA/Prophet)")
    
    try:
        with st.sidebar:
            st.header("設定")
            file = st.file_uploader("Excelファイルをアップロード", type=['xlsx'])
            
            if st.button("セッションリセット"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
            
            if file is not None:
                try:
                    df = clean_data(file)
                    df, monthly_trend, weekday_trend = analyze_explanatory_variables(df)
                    st.session_state['df'] = df
                    st.session_state['forecast'] = None
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        model_type = st.selectbox("モデル選択", ['ARIMA', 'SARIMA', 'Prophet'])
                    with col2:
                        seasonal = st.checkbox("季節性を考慮", value=True if model_type == 'SARIMA' else False)
                    with col3:
                        n_trials = st.slider("Optuna試行回数", 10, 100, 30)
                    
                    forecast_period = st.slider("予測期間 (日数)", 30, 365, 30)
                    
                    if st.button("予測実行"):
                        with st.spinner('予測を実行中...'):
                            config = ModelConfig(len(df), n_trials)
                            forecast_values = forecast(df, model_type, seasonal, forecast_period, config)
                            st.session_state['forecast'] = forecast_values
                            st.session_state['forecast_period'] = forecast_period
                            st.success('予測が完了しました！')
                            
                            # Evaluate model
                            train_true = df['Quantity']
                            train_pred = forecast_values[:len(train_true)]
                            metrics = evaluate_model(train_true, train_pred)
                            st.subheader("モデル評価結果")
                            st.write(f"MAPE: {metrics['MAPE']:.2f}% (10%未満が非常に良好です)")
                            st.write(f"RMSE: {metrics['RMSE']:.2f}")
                            st.write(f"MAE: {metrics['MAE']:.2f}")
                            
                            # Display acceptable MAPE range
                            st.info("一般的なMAPE評価基準: 10%未満で非常に良好、10%から20%で良好、20%から50%で許容範囲、50%以上で改善が必要です。")
                
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    return
        
        if 'df' in st.session_state and st.session_state['df'] is not None:
            st.subheader("データプレビュー")
            st.dataframe(st.session_state['df'])
            
            if 'forecast' in st.session_state and st.session_state['forecast'] is not None:
                st.subheader("予測結果")
                
                # Plot forecast
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(st.session_state['df']['Date'], st.session_state['df']['Quantity'], label='実績データ')
                
                future_dates = [st.session_state['df']['Date'].iloc[-1] + datetime.timedelta(days=i)
                                for i in range(1, st.session_state['forecast_period'] + 1)]
                
                ax.plot(future_dates, st.session_state['forecast'], label='予測', color='red')
                plt.xticks(rotation=45)
                plt.xlabel("日付")
                plt.ylabel("数量")
                plt.legend()
                st.pyplot(fig)
                
                # Output to Excel
                output_df = pd.DataFrame({
                    '日付': future_dates,
                    '予測数量': st.session_state['forecast']
                })
                
                buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                output_df.to_excel(buffer, index=False)
                
                with open(buffer.name, 'rb') as f:
                    st.download_button(
                        label="予測結果をダウンロード",
                        data=f,
                        file_name="forecast_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                os.unlink(buffer.name)  # Delete temporary file

    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {str(e)}")

if __name__ == '__main__':
    main()

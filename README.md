# Tennis Match Prediction & Betting Strategy Model

A comprehensive machine learning project that predicts tennis match outcomes and implements profitable betting strategies using ensemble methods and advanced feature engineering.

## 🎯 Project Overview

This project demonstrates end-to-end data science capabilities by building a predictive model for ATP tennis matches, combining multiple data sources, advanced feature engineering, and quantitative betting strategies. The model achieved **73.2% accuracy** and **81.1% AUC** on test data, with backtested betting strategies showing positive ROI.

## 🔑 Key Technical Features

### 📊 Data Engineering & Integration
- **Multi-source data fusion**: Combined Jeff Sackmann's ATP dataset, Infosys API data, and tennis-odds datasets
- **Advanced web scraping**: Built custom scrapers using the Infosys API to collect real-time match statistics
- **Data standardization**: Normalized tournament IDs, player names, and match formats across 20+ years of data
- **Feature-rich dataset**: Created 153 engineered features from 43,000+ matches (2002-2025)

### 🧠 Feature Engineering
- **Dynamic Elo ratings**: Implemented surface-specific and overall Elo calculations with tournament-weighted updates
- **Rolling statistics**: Generated moving averages for player performance across multiple time windows (5, 50, 100, 200 matches)
- **Head-to-head metrics**: Computed historical matchup statistics with surface and tournament-specific adjustments
- **Form indicators**: Created recent form metrics (1-month, 3-month performance tracking)
- **Advanced match parsing**: Extracted set-by-set scores, tiebreak statistics, and margin analysis

### 🤖 Machine Learning Pipeline
- **Feature selection**: Reduced dimensionality from 153 to 31 features using importance-based selection (top 20%)
- **Hyperparameter optimization**: Used Bayesian optimization (Optuna) with 5-fold cross-validation over 200 trials
- **Model validation**: Implemented stratified cross-validation with temporal splitting to prevent data leakage

### 📈 Quantitative Strategy Implementation
- **Kelly Criterion betting**: Implemented fractional Kelly criterion for optimal stake sizing
- **Edge detection**: Developed expected value calculations comparing model probabilities vs. market odds
- **Risk management**: Built Monte Carlo simulations to estimate drawdown and bust probabilities
- **Backtesting framework**: Created realistic sequential betting simulation with bankroll management

## 🏆 Results & Performance

### Model Metrics
- **Training Accuracy**: 99.99%
- **Test Accuracy**: 73.2%
- **AUC Score**: 81.1%
- **Cross-validation AUC**: 80.7% ± 0.26%

### Betting Strategy Performance (2025 Data)
<img width="1489" height="790" alt="PNL" src="https://github.com/user-attachments/assets/09088ad5-01d0-4650-b44e-32eaca11d257" />


### Key Strategy Metrics
- **Total Bets Placed**: 162
- **Win Rate**: 74% 
- **ROI**: 54.7%

## 🛠 Technical Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: XGBoost, Scikit-learn, TensorFlow/Keras
- **Optimization**: Optuna (Bayesian hyperparameter tuning)
- **Web Scraping**: Custom API integration, BeautifulSoup
- **Visualization**: Matplotlib, Plotly
- **Statistical Analysis**: Custom Elo implementation, Kelly Criterion

## 🎖 Data Science Highlights

1. **Domain Expertise**: Deep understanding of tennis statistics, tournament structures, and betting markets
2. **Feature Engineering**: Created sophisticated rolling statistics and player form indicators
3. **Model Interpretability**: Identified key predictive features (Elo ratings, recent form, head-to-head records)
5. **Quantitative Finance**: Applied Kelly Criterion and risk management principles
6. **Real-World Application**: Developed practical betting strategies with realistic constraints

## 💡 Business Impact

This project demonstrates the ability to:
- Transform raw sports data into actionable insights
- Build robust predictive models for high-noise environments  
- Implement quantitative trading strategies with risk management
- Handle real-time data integration and API development
- Create end-to-end machine learning pipelines

### References and Credits
- Jeff Sackmann Datasets - [Link Text](https://github.com/JeffSackmann/tennis_atp)
- Green Code - Project provided excellent reference point.
- Glad94 Infotennis Infosys API Match Scraper - [Link Text](https://github.com/glad94/infotennis/tree/main)

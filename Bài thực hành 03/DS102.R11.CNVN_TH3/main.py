import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from linear_regression import LinearRegression

def organize_data(data: pd.DataFrame):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)

    return (X, y)


def convert_month2int(month: str) -> int:
    month_dict = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12,
    }
    return month_dict[month]


def convert_day2int(day: str) -> int:
    day_dict = {
        'sun': 1,
        'mon': 2,
        'tue': 3,
        'wed': 4,
        'thu': 5,
        'fri': 6,
        'sat': 7,
    }
    return day_dict[day]


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    df['month'] = df['month'].map(convert_month2int)
    df['day'] = df['day'].map(convert_day2int)

    return df


def standardize_data(train_data, test_data):
    continuous_variables = [
        'FFMC', 'DMC', 'DC', 'ISI',
        'temp', 'wind', 'RH', 'rain'
    ]

    train_data = train_data.copy()
    test_data = test_data.copy()

    for variable in continuous_variables:
        mean = train_data[variable].mean()
        std = train_data[variable].std(ddof = 0)

        if std != 0:
            train_data[variable] = (train_data[variable] - mean) / std
            test_data[variable] = (test_data[variable] - mean) / std

    return train_data, test_data


if __name__ == "__main__":

    path = os.path.join(os.path.dirname(__file__), "data", "forestfires.csv")
    data = pd.read_csv(path)
    data = preprocess_data(data)

    data = data.sample(frac = 1, random_state = 42, ignore_index = True)

    train_size = round(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)

# Assignment 1
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test)
    rmse1 = lr_model.evaluate(y_test, y_predicted)

    print("Bài 1 - RMSE trên dữ liệu gốc:", rmse1)

# Assignment 2
    train_scaled, test_scaled = standardize_data(train_data, test_data)

    X_train_scaled, y_train_scaled = organize_data(train_scaled)
    X_test_scaled, y_test_scaled = organize_data(test_scaled)

    lr_scaled = LinearRegression()
    lr_scaled.fit(X_train_scaled, y_train_scaled)

    y_predicted_scaled = lr_scaled.predict(X_test_scaled)
    rmse2 = lr_scaled.evaluate(y_test_scaled, y_predicted_scaled)

    print("Bài 2 - RMSE trên dữ liệu chuẩn hóa:", rmse2)

    print("Trung bình sau chuẩn hóa:")
    print(train_scaled[
        ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'wind', 'RH', 'rain']
    ].mean())

    print("Phương sai sau chuẩn hóa:")
    print(train_scaled[
        ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'wind', 'RH', 'rain']
    ].var(ddof = 0))

# Assignment 3
    correlation = train_data.iloc[:, :-1].corr()

    sns.heatmap(correlation, annot = True, cmap = "coolwarm")
    plt.show()

    train_data_new = train_data.drop(columns = ['DMC'])
    test_data_new = test_data.drop(columns = ['DMC'])

    X_train_new, y_train_new = organize_data(train_data_new)
    X_test_new, y_test_new = organize_data(test_data_new)

    lr_new = LinearRegression()
    lr_new.fit(X_train_new, y_train_new)

    y_predicted_new = lr_new.predict(X_test_new)
    rmse3 = lr_new.evaluate(y_test_new, y_predicted_new)

    print("Bài 3 - RMSE sau khi loại bỏ DMC:", rmse3)

# Assignment 4
    try:
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression

        sklearn_model = SklearnLinearRegression()
        sklearn_model.fit(X_train, y_train)

        y_predicted_sklearn = sklearn_model.predict(X_test)
        rmse4 = np.sqrt(np.mean((y_test - y_predicted_sklearn) ** 2))

        print("Bài 4 - RMSE sử dụng Scikit-learn:", rmse4)
        print("RMSE của mô hình NumPy:", rmse1)
        print("RMSE của mô hình Scikit-learn:", rmse4)

    except (ImportError, OSError) as error:
        print("Không thể chạy Bài 4 do Scikit-learn bị lỗi khi tải thư viện:")
        print(error)
        print("Bài 1, 2 và 3 vẫn đã được thực hiện.")

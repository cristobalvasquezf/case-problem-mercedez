def check_stationarity(p_value: float):
    if p_value < 0.05:
        print("Data is likely stationary")
    else:
        print("Data is likely non-stationary")

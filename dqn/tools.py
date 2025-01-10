def percentage_increase(a, b):
    if a == 0:
        raise ValueError("The value of 'a' must not be zero.")
    increase = b - a
    percentage = (increase / a) * 100
    return round(percentage, 2)

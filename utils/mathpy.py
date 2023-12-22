import math

def normal_log_density(x, mean, log_std, std):
    """
    Computes the log density of a normal distribution with mean `mean` and standard deviation `std`
    at the given input `x`.
    """
    log_density = -(x - mean).pow(2) / (2 * std.pow(2)) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def normalize_angle(angle):
    angle = (angle % 360 + 360) % 360
    return angle

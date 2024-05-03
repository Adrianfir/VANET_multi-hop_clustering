import numpy as np


def ratio(num_bus, num_ch_sub_ch):
    # Ensure num_bus and num_ch_sub_ch are not zero for the logarithmic calculation
    num_bus = max(num_bus, 1)
    num_ch_sub_ch = max(num_ch_sub_ch, 1)

    beta_ch = 1
    beta_bus = 1
    beta_bus = 0.7 if num_bus == num_ch_sub_ch else beta_bus
    beta_bus = 0.8 if (2 <= (num_ch_sub_ch + 1) / (num_bus + 1) < 2.5) and (num_bus != 0) else beta_bus
    beta_bus = 0.95 if (2.5 <= (num_ch_sub_ch + 1) / (num_bus + 1) < 4) and (num_bus != 0) else beta_bus

    return beta_bus, beta_ch
print(ratio(2,12))
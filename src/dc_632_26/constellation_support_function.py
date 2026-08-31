
# efficiency = dmin_sqr / Eb
# Eb = Es / M
# M number of bits per symbol
import numpy as np

def minimum_distance_sqr(mod_map):
    constellation_points = list(mod_map.values())
    distance_list = []
    for i in range(len(constellation_points)):
        for j in range(len(constellation_points)):
            if i != j:
                x0 =  np.real(constellation_points[i])
                x1 = np.real(constellation_points[j])
                y0 = np.imag(constellation_points[i])
                y1 = np.imag(constellation_points[j])
                dist2 = (x1-x0)**2 + (y1-y0)**2
                distance_list.append(dist2)
    return min(distance_list)

def Energy_per_symbol(symbol_stream_vals, mod_map):
    num_keys = len(mod_map.keys())
    total_energy = np.linalg.norm(symbol_stream_vals)**2

    total_average_energy = total_energy / len(symbol_stream_vals)

    return total_average_energy

def bits_per_symbol(mod_map):
    return np.log2(len(mod_map))

def Energy_per_bit(symbol_stream_vals, mod_map):
    M = bits_per_symbol(mod_map) # bits per symbol

    Es = Energy_per_symbol(symbol_stream_vals, mod_map)

    Eb = Es / M

    return Eb

def calculate_efficiency(Eb, mod_map):
    d_min_sqr = minimum_distance_sqr(mod_map)

    return d_min_sqr / Eb

# Test Mod-Map

mod_map = {0b00: 1+1j, 0b01: -1+1j, 0b11: 1-1j, 0b10: -1-1j}
symbol_stream = [1+1j, -1+1j, 1-1j, -1-1j]

Eb = Energy_per_bit(symbol_stream, mod_map)
dmin = minimum_distance_sqr(mod_map)
Es = Energy_per_symbol(symbol_stream, mod_map)
bps = bits_per_symbol(mod_map)
eff = calculate_efficiency(Eb, mod_map)

## Results TODO: Need to fix Es, incorrect for our current given test 
print(f"Energy per bit: {Eb}")
print(f"Minimum distance squared: {dmin}")
print(f"Energy per symbol: {Es}")
print(f"Bits per symbol: {bps}")
print(f"Efficiency: {eff}")

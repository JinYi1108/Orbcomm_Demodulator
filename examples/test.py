import numpy as np
import sys
import os
import matplotlib.pylab as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbdemod import(
    enable_logging, ddc, cr, rrc, 
    symbol_timing_recovery, costas, decode,
    find_packet_start, bits_to_packets,validate_packet,
    plot_constellation, plot_eye_diagram
)

enable_logging(level="INFO")

fs_in = 480e6
freq_orbcomm = 137.46e6 # directly observed from the waterfall plot
fs_mid = 2.4e6
fs_out = 9600
baud_rate = 4800.0

seconds = 1
n = 0


#iq_data = np.fromfile("./iq_data_0.dat", dtype=np.complex64)
iq_data = np.fromfile("./raw_data/1552071892p6.dat", dtype=np.complex64)
plot_constellation(iq_data,  sample_skip=1,save=True,save_path='./examples/pic/p6/iq_data.png')


iq_cr = cr(iq_data, fs_out)
plot_constellation(iq_cr,  sample_skip=1, save=True,save_path='./examples/pic/p6/iq_cr.png')
plot_eye_diagram(iq_cr[::2], save=True,save_path='./examples/pic/p6/iq_cr_eye.png')

iq_rrc = rrc(iq_cr, fs_out,baud_rate)
plot_constellation(iq_rrc,  sample_skip=1,save=True,save_path='./examples/pic/p6/iq_rrc.png')
plot_eye_diagram(iq_rrc[::2], save=True,save_path='./examples/pic/p6/iq_rrc_eye.png')

iq_timed,tau,dtau,error = symbol_timing_recovery(iq_rrc)
plot_constellation(iq_timed,  sample_skip=1,save=True,save_path='./examples/pic/p6/iq_timed.png')
plot_eye_diagram(iq_timed, save=True,save_path='./examples/pic/p6/iq_timed_eye.png')

plt.figure()
plt.subplot(311)
plt.title("Timing offset (tau)")
plt.plot(tau)

plt.subplot(312)
plt.title("Derivative (Dtau)")
plt.plot(dtau)
plt.tight_layout()

plt.subplot(313)
plt.title("Timing error signal")
plt.plot(error)
plt.tight_layout()

plt.savefig('./examples/pic/p6/str.png')

iq_costas, phase,freq = costas(iq_timed)
plot_constellation(iq_costas, sample_skip=1, save=True,save_path='./examples/pic/p6/iq_costas.png')
plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.title('Phase output of PLL')
plt.plot(phase)
plt.grid()

plt.subplot(212)
plt.title('Frequency of PLL')
plt.plot(freq)
plt.grid()

plt.savefig('./examples/pic/p6/costas.png')

bit_decode, deg  = decode(iq_costas)
plt.figure()

plt.xlabel('Symbol Number')
plt.ylabel('Angle (degrees)')
plt.plot(deg, 'x')
plt.grid()
plt.savefig('./examples/pic/p6/angle.png')

offset, reverse_order = find_packet_start(bit_decode)

hex_packet = bits_to_packets(bit_decode, offset, reverse_order)

valid_hex_packet = validate_packet(hex_packet,output_file="./examples/orbdemod_packets_p6.txt")

print(hex_packet)
print('-'*50)
print(valid_hex_packet)







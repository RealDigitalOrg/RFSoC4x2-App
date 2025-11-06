import os, math, numpy as np

os.environ.setdefault("QT_API", "pyside6")



from PySide6 import QtWidgets, QtCore, QtGui

import pyqtgraph as pg

from pynq.overlays.base import BaseOverlay



# base system variables

FS_HZ_FALLBACK = 2.458e6

N_SAMPLES      = 4096

INTERVAL_MS    = 120

DECIM          = 2



ADC_BITS       = 14

RAW_FS_COUNTS  = float(2 ** (ADC_BITS - 1))  # 8192

ASSUMED_FS_VPP = 1.00  # Full scale voltage peak to peak



TX_TO_RX = {0: 2, 1: 3}

RX_LIST  = [2, 3]

NCO_OFFSET_HZ = {2: 10_000.0, 3: 10_000.0}



# DAC amplitude polynomial 

def cal_scale_poly(f_mhz: float) -> float:

    return (

        -1.07171717e-08 * (f_mhz**4)

        + 8.29121212e-07 * (f_mhz**3)

        + 1.96061919e-04 * (f_mhz**2)

        - 1.61006121e-02 * f_mhz

        + 1.11154798

    )



DAC_VFS_VPP_I_ONLY = 1.00



def volts_to_vpp(value: float, unit: str) -> float:

    u = unit.strip().lower()

    if u in ("vpp","pp","p-p"): return float(value)

    if u in ("vp","peak","pk"): return 2.0 * float(value)

    if u in ("vrms","rms"):     return 2.0 * math.sqrt(2.0) * float(value)

    raise ValueError("Unit must be Vpp, Vp, or Vrms")



# Attenuation models

# LUT for TX0→RX2

ATT_LUT_RX2 = [

    (5.0,  15.29),

    (10.0, 13.73),

    (20.0, 13.03),

    (50.0, 12.63),

    (70.0, 12.62),

    (80.0, 12.63),

    (90.0, 12.67),

]



# LUT for TX1→RX3

ATT_LUT_RX3 = [

    (5.0,  15.51),

    (10.0, 13.94),

    (20.0, 13.21),

    (50.0, 12.77),

    (100.0, 12.81),

]



def _interp_att(points, f_mhz: float) -> float:

    if not points: return 0.0

    pts = sorted(points, key=lambda t: t[0])

    if f_mhz <= pts[0][0]: return float(pts[0][1])

    if f_mhz >= pts[-1][0]: return float(pts[-1][1])

    for (x0,y0),(x1,y1) in zip(pts[:-1], pts[1:]):

        if x0 <= f_mhz <= x1:

            t = 0.0 if x1 == x0 else (f_mhz - x0)/(x1 - x0)

            return float(y0*(1.0 - t) + y1*t)

    return float(pts[-1][1])



def atten_db_lookup(rx_idx: int, f_mhz: float) -> float:

    """Path attenuation (dB) at RF frequency. Clamp and linearly interpolate."""

    if rx_idx == 2:   # TX0 → RX2

        return _interp_att(ATT_LUT_RX2, f_mhz)

    if rx_idx == 3:   # TX1 → RX3

        return _interp_att(ATT_LUT_RX3, f_mhz)

    return _interp_att(ATT_LUT_RX3, f_mhz)



# TX helpers

def set_mixer(tx, freq_mhz: float):

    try:

        tx.dac_block.MixerSettings["Freq"] = float(freq_mhz)

        for k, v in [("PhaseOffset", 0.0), ("EventSource", 0)]:

            try: tx.dac_block.MixerSettings[k] = v

            except Exception: pass

    except Exception: pass



def select_amplitude_source(tx):

    for attr, val in [("source","amplitude"), ("data_sel","amplitude"), ("data_source","amplitude")]:

        try:

            setattr(tx, attr, val); return

        except Exception: pass



def write_I_only_gain(ctrl, gain_0_to_1: float):

    scaled  = int(round(max(0.0, min(1.0, gain_0_to_1)) * 0x7FFF))

    int_reg = (scaled & 0xFFFF) | (0 << 16)

    ctrl.write(0x04, int_reg)



def apply_tx_config(tx, freq_mhz: float, request_val: float, request_unit: str):

    """Always applies polynomial on TX amplitude scaling."""

    req_vpp    = volts_to_vpp(request_val, request_unit)

    scale      = cal_scale_poly(freq_mhz) 

    target_vpp = req_vpp * scale

    gain = max(0.0, min(1.0, target_vpp / DAC_VFS_VPP_I_ONLY))

    tx.control.enable = False

    select_amplitude_source(tx)

    set_mixer(tx, freq_mhz)

    try: write_I_only_gain(tx.control, gain)

    except Exception: tx.control.gain = float(gain)

    tx.control.enable = True

    return {"req_vpp": req_vpp, "scale": scale, "target_vpp": target_vpp, "gain": gain}



# ADC helpers 

def normalize_adc_if_needed(y: np.ndarray):

    m = float(np.max(np.abs(y)))

    if m > 2.5:

        return (y.astype(np.float32) / RAW_FS_COUNTS), True

    return y.astype(np.float32), False



def remove_dc(y: np.ndarray) -> np.ndarray:

    return y - np.mean(y)



def measure_vpp_fs(y_fs_zero_mean: np.ndarray) -> float:

    return float(np.max(y_fs_zero_mean) - np.min(y_fs_zero_mean))



def vrms_from_vpp(vpp_volts: float) -> float:

    return vpp_volts / (2.0 * math.sqrt(2.0))



def get_ddc_scale_if_any(rx) -> float:

    candidates = []

    for path in [

        ("adc_block","FineMixerScale"),

        ("adc_block","MixerScale"),

        ("adc_block","DecimationGain"),

        ("block","DecimationGain"),

        ("control","gain"),

    ]:

        try:

            obj = rx

            for a in path: obj = getattr(obj, a)

            v = float(obj)

            if v > 0.0: candidates.append(v)

        except Exception: pass

    if not candidates: return 1.0

    try: return float(sorted(candidates)[-1])

    except Exception: return 1.0



def vpp_fs_to_volts(vpp_fs: float, fs_vpp_est: float, ddc_scale: float = 1.0) -> float:

    return vpp_fs * (fs_vpp_est / 2.0) * ddc_scale



# UI panes

class TxPane(QtWidgets.QGroupBox):

    def __init__(self, title: str, tx_idx: int, init_freq: float):

        super().__init__(title)

        self.tx_idx = tx_idx

        g = QtWidgets.QGridLayout(self)



        self.enable = QtWidgets.QCheckBox("Enable"); self.enable.setChecked(True)

        g.addWidget(self.enable, 0, 0)



        g.addWidget(QtWidgets.QLabel("Freq (MHz)"), 0, 1)

        self.freq = QtWidgets.QDoubleSpinBox()

        self.freq.setRange(0.0, 6000.0); self.freq.setDecimals(6); self.freq.setSingleStep(0.01)

        self.freq.setValue(init_freq)

        g.addWidget(self.freq, 0, 2)



        g.addWidget(QtWidgets.QLabel("Amplitude"), 1, 0)

        self.amp = QtWidgets.QDoubleSpinBox()

        self.amp.setDecimals(4); self.amp.setRange(0.0, 5.0); self.amp.setValue(0.50)

        g.addWidget(self.amp, 1, 1)



        self.unit = QtWidgets.QComboBox(); self.unit.addItems(["Vpp","Vp","Vrms"])

        g.addWidget(self.unit, 1, 2)



        self.apply = QtWidgets.QPushButton("Apply TX"); g.addWidget(self.apply, 0, 3, 2, 1)

        self.apply.clicked.connect(self.apply_now)


        self.status = QtWidgets.QLabel("")

        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        g.addWidget(self.status, 3, 0, 1, 4)



        self._last = None



    def apply_now(self):

        tx = base.radio.transmitter.channel[self.tx_idx]

        freq = self.freq.value()

        if not self.enable.isChecked():

            tx.control.enable = False

            self.status.setText(f"TX{self.tx_idx} disabled.")

            return

        info = apply_tx_config(

            tx, freq_mhz=freq,

            request_val=self.amp.value(),

            request_unit=self.unit.currentText(),

        )

        self._last = info

        self.status.setText(f"TX{self.tx_idx} @ {freq:.6f} MHz, {self.amp.value():.4g} {self.unit.currentText()} applied.")



# Main widget

class Main(QtWidgets.QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("RFSoC 4x2 — Clean UI: Loopback de-embed (+pre-poly), Ext-gen shows ADC-pin volts")

        self.resize(1520, 820)

        pg.setConfigOptions(antialias=False)



        self.fs_vpp_est = ASSUMED_FS_VPP

        self.fs_hz = self._detect_fs_hz(FS_HZ_FALLBACK)



        root = QtWidgets.QVBoxLayout(self)



        # Controls row aligned with plots (two-column grid)

        controls = QtWidgets.QGridLayout()

        controls.setColumnStretch(0, 1)

        controls.setColumnStretch(1, 1)

        root.addLayout(controls)



        self.tx0 = TxPane("TX0 → RX2", tx_idx=0, init_freq=20.00)

        self.tx1 = TxPane("TX1 → RX3", tx_idx=1, init_freq=20.00)

        controls.addWidget(self.tx0, 0, 0)   # left column

        controls.addWidget(self.tx1, 0, 1)   # right column



        # initial apply + NCOs

        self.tx0.apply_now(); self.tx1.apply_now()

        for tx_idx in TX_TO_RX: self._set_rx_nco_for(tx_idx)



        self.tx0.freq.valueChanged.connect(lambda _=None: self._on_freq_changed(0))

        self.tx1.freq.valueChanged.connect(lambda _=None: self._on_freq_changed(1))



        # Plots (two columns)

        grid = QtWidgets.QGridLayout(); root.addLayout(grid, 1)

        grid.setColumnStretch(0, 1)

        grid.setColumnStretch(1, 1)



        # Time base in microseconds

        self.t_us = (np.arange(N_SAMPLES) / self.fs_hz) * 1e6

        self.t_us_dec = self.t_us[::max(1, DECIM)]



        self.curves, self.readouts = {}, {}

        self.plots, self.pause_btns, self.paused = {}, {}, {}

        self.last_wave = {}  # store last t,y for Autoset



        for col, rx_idx in enumerate(RX_LIST):

            pw = pg.PlotWidget(title=f"ADC {rx_idx} — Loopback: de-embed (+pre-poly); Ext-gen: ADC-pin volts")

            pw.showGrid(x=True, y=True, alpha=0.25)

            pw.getAxis('bottom').enableAutoSIPrefix(False)

            pw.setLabel("bottom", "Time", units="ms")

            pw.setLabel("left", "Amplitude (V)")

            c = pw.plot(self.t_us_dec, np.zeros_like(self.t_us_dec), pen=pg.mkPen(width=1.3))

            grid.addWidget(pw, 0, col)

            self.curves[rx_idx] = c

            self.plots[rx_idx] = pw



            # Buttons row: Autoset + Pause/Resume

            btn_row = QtWidgets.QHBoxLayout()

            autoset = QtWidgets.QPushButton("Autoset")

            pause   = QtWidgets.QPushButton("Pause")

            btn_row.addWidget(autoset)

            btn_row.addWidget(pause)

            btn_row.addStretch(1)

            grid.addLayout(btn_row, 1, col)



            autoset.clicked.connect(lambda _=None, r=rx_idx: self._autoset(r))

            pause.clicked.connect(lambda _=None, r=rx_idx: self._toggle_pause(r))

            self.pause_btns[rx_idx] = pause

            self.paused[rx_idx] = False



            ro = QtWidgets.QLabel("Vpp≈0.000 V   Vrms≈0.000 V   Mode=—   f_RF≈— MHz   Atten≈— dB")

            grid.addWidget(ro, 2, col, alignment=QtCore.Qt.AlignLeft)

            self.readouts[rx_idx] = ro



        self.timer = QtCore.QTimer(self)

        self.timer.setInterval(INTERVAL_MS)

        self.timer.timeout.connect(self.update_once)

        self.timer.start()

   
 # Helpers/UI wiring   

    def _autoset(self, rx_idx: int):

        """

        Zoom to a clear waveform view:

          - Estimate frequency from the currently plotted (decimated) data

          - Show ~6 cycles (clamped to data length)

          - Y-scale to ±(robust peak*1.15), centered at 0

        Falls back to full autoRange if frequency is unclear.

        """

        try:

            t_us, y = self.last_wave.get(rx_idx, (None, None))

            if t_us is None or len(y) < 16:

                self.plots[rx_idx].autoRange()

                return



            # Estimate frequency from decimated data

            fs_dec_hz = self.fs_hz / max(1, DECIM)

            y0 = y - np.mean(y)

            f_est_hz = self._estimate_freq_hz(y0, fs_dec_hz)

            if not np.isfinite(f_est_hz) or f_est_hz <= 0:

                self.plots[rx_idx].autoRange()

                return



            # Choose a window of ~6 cycles (ensure a minimum window size)

            n_cycles = 6.0

            T_us = (1.0 / f_est_hz) * 1e6  # period in microseconds

            win_us = max(T_us * n_cycles, (t_us[-1] - t_us[0]) * 0.08)

            t_end = float(t_us[-1])

            t_start = max(float(t_us[0]), t_end - win_us)



            # Robust amplitude estimate & padding

            if len(y0) > 3:

                lo, hi = np.percentile(y0, [1.0, 99.0])

                amp = max(abs(lo), abs(hi))

            else:

                amp = float(np.max(np.abs(y0))) if len(y0) else 1e-3

            amp = max(amp, 1e-6)

            pad = amp * 0.15

            y_min, y_max = -amp - pad, amp + pad



            vb = self.plots[rx_idx].getPlotItem().getViewBox()

            vb.setXRange(t_start, t_end, padding=0.0)

            vb.setYRange(y_min, y_max, padding=0.0)

        except Exception:

            try: self.plots[rx_idx].autoRange()

            except Exception: pass



    def _toggle_pause(self, rx_idx: int):

        self.paused[rx_idx] = not self.paused[rx_idx]

        self.pause_btns[rx_idx].setText("Resume" if self.paused[rx_idx] else "Pause")

        txt = self.readouts[rx_idx].text().split("   ")

        if self.paused[rx_idx]:

            if not any(s.startswith("PAUSED") for s in txt):

                txt.insert(0, "PAUSED")

        else:

            txt = [s for s in txt if not s.startswith("PAUSED")]

        self.readouts[rx_idx].setText("   ".join(txt))



    def _on_freq_changed(self, tx_idx: int):

        self._set_rx_nco_for(tx_idx)



    def _set_rx_nco_for(self, tx_idx: int):

        rx_idx = TX_TO_RX[tx_idx]

        try:

            pane = self._txpane_for_tx(tx_idx)

            f_mhz = float(pane.freq.value())

            offset_mhz = (NCO_OFFSET_HZ.get(rx_idx, 10_000.0)) / 1e6

            RX[rx_idx].adc_block.MixerSettings["Freq"] = f_mhz - offset_mhz

        except Exception as e:

            print(f"RX{rx_idx} NCO set failed:", e)



    def _txpane_for_rx(self, rx_idx: int) -> 'TxPane':

        for tx_idx, rxi in TX_TO_RX.items():

            if rxi == rx_idx:

                return self._txpane_for_tx(tx_idx)

        return self.tx0



    def _txpane_for_tx(self, tx_idx: int) -> 'TxPane':

        return self.tx0 if tx_idx == 0 else self.tx1



    def _get_rx_nco_mhz(self, rx_idx: int) -> float:

        try: return float(RX[rx_idx].adc_block.MixerSettings["Freq"])

        except Exception: return 0.0



    def update_once(self):

        try:

            fs_vpp = ASSUMED_FS_VPP  # fixed full-scale

            for rx_idx in RX_LIST:

                if self.paused.get(rx_idx, False):

                    continue



                pane = self._txpane_for_rx(rx_idx)



                try:

                    x = RX[rx_idx].transfer(N_SAMPLES)

                except Exception as e:

                    print(f"RX{rx_idx} transfer error:", e); continue



                y = np.real(x)

                y_fs, _ = normalize_adc_if_needed(y)

                y0 = remove_dc(y_fs)



                ddc_scale = get_ddc_scale_if_any(RX[rx_idx])

                y_volts_adc = y0 * (fs_vpp / 2.0) * ddc_scale  # ADC-pin volts



                # Determine RF for info

                f_bb_hz = self._estimate_freq_hz(y0, self.fs_hz)

                if pane.enable.isChecked():

                    # LOOPBACK MODE (TX enabled) — de-embed & pre-poly

                    mode = "Loopback"

                    f_rf_mhz = float(pane.freq.value())

                    atten_db = atten_db_lookup(rx_idx, f_rf_mhz)

                    beta = 10.0 ** (atten_db / 20.0)

                    y_src = y_volts_adc * beta  # source (post-poly)

                    y_src /= max(1e-9, cal_scale_poly(f_rf_mhz))  # divide out TX poly -> show requested (pre-poly)

                else:

                    # EXTERNAL GENERATOR MODE (TX disabled) 

                    mode = "Ext-gen"

                    f_rf_mhz = abs(f_bb_hz)/1e6 + self._get_rx_nco_mhz(rx_idx)

                    atten_db = 0.0

                    y_src = y_volts_adc  # show ADC-pin volts directly



                # Plot (decimated) in µs domain

                y_plot = y_src[::DECIM] if DECIM > 1 else y_src

                self.curves[rx_idx].setData(self.t_us_dec, y_plot, _callSync="off")



                # Save for Autoset

                self.last_wave[rx_idx] = (self.t_us_dec, y_plot)



                # Readouts

                vpp_fs  = measure_vpp_fs(y0)

                vpp_adc = vpp_fs_to_volts(vpp_fs, fs_vpp, ddc_scale)

                if pane.enable.isChecked():

                    vpp_src = vpp_adc * (10.0 ** (atten_db / 20.0))

                    vpp_src /= max(1e-9, cal_scale_poly(float(pane.freq.value())))

                else:

                    vpp_src = vpp_adc  # ADC-pin in ext-gen mode

                vrms_src = vrms_from_vpp(vpp_src)



                self.readouts[rx_idx].setText(

                    f"Vpp≈{vpp_src:.3f} V   Vrms≈{vrms_src:.3f} V   "

                    f"Mode={mode}   f_RF≈{f_rf_mhz:.3f} MHz   Atten≈{atten_db:.2f} dB"

                )

        except Exception as e:

            print("RX update error:", e)



    # Frequency reading

    def _detect_fs_hz(self, fallback: float) -> float:

        candidates = []

        try:

            for rx_idx in RX_LIST:

                try: rx = RX[rx_idx]

                except Exception: continue

                for path in [

                    ("adc_block","SamplingRate"), ("adc_block","SampleRate"),

                    ("adc_block","Fs"), ("adc_block","Fs_Hz"),

                    ("block","SamplingRate"), ("block","SampleRate"),

                    ("control","SamplingRate"),

                ]:

                    try:

                        obj = rx

                        for a in path: obj = getattr(obj, a)

                        val = float(obj)

                        if val > 1e3: candidates.append(val)

                    except Exception: pass

            try:

                recv = base.radio.receiver

                for attr in ["sample_rate","samplerate","fs_hz","Fs","Fs_Hz","SamplingRate","SampleRate"]:

                    try:

                        val = float(getattr(recv, attr))

                        if val > 1e3: candidates.append(val)

                    except Exception: pass

            except Exception: pass

        except Exception: pass

        if candidates:

            try: return float(sorted(candidates)[-1])

            except Exception: pass

        return float(fallback)



    def _estimate_freq_hz(self, y_fs: np.ndarray, fs_hz: float) -> float:

        if len(y_fs) < 4: return 0.0

        y = y_fs - np.mean(y_fs)

        w = np.hanning(len(y))

        Y = np.fft.rfft(y * w)

        mag = np.abs(Y)

        if len(mag) <= 1: return 0.0

        k = int(np.argmax(mag[1:])) + 1

        return float(k * fs_hz / len(y))



    def closeEvent(self, ev):

        try:

            for ch in base.radio.transmitter.channel:

                ch.control.enable = False

        except Exception:

            pass

        super().closeEvent(ev)



# Main

base = BaseOverlay("base.bit")

base.init_rf_clks()

TX = base.radio.transmitter.channel

RX = base.radio.receiver.channel



if __name__ == "__main__":

    app = QtWidgets.QApplication([])

    w = Main(); w.show()

    app.exec()


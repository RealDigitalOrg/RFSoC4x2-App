#!/usr/bin/env python3

# RFSoC 4x2 — Dual scope (TX0→RX2, TX1→RX3)

# Auto RX NCO tracking for external signals (when TX disabled)

# UI: two TX panes above their plots, plus Autoset + Pause/Resume per ADC



import os, math, numpy as np

os.environ.setdefault("QT_API", "pyside6")



from PySide6 import QtWidgets, QtCore, QtGui

import pyqtgraph as pg

from pynq.overlays.base import BaseOverlay



#  System / plot

FS_HZ_FALLBACK = 2.458e9

N_SAMPLES      = 4096

INTERVAL_MS    = 120

DECIM          = 2

ADC_NCO_HZ = 1.2288e9

ADC_BITS       = 14

RAW_FS_COUNTS  = float(2 ** (ADC_BITS - 1))

ASSUMED_FS_VPP = 1.00



TX_TO_RX = {0: 2, 1: 3}

RX_LIST  = [2, 3]

NCO_OFFSET_HZ = {2: 10_000.0, 3: 10_000.0}



#  DAC amplitude poly

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



#  Attenuation models

ATT_LUT_RX3 = [

    (5.0, 15.51),

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

            t = (f_mhz - x0)/(x1 - x0)

            return float(y0*(1-t)+y1*t)

    return float(pts[-1][1])



_RX2_A,_RX2_B,_RX2_C,_RX2_D = (-2.31872862e-05, 4.10331001e-03, -2.19793859e-01, 1.59878476e+01)

def _atten_rx2_poly_db(f_mhz: float) -> float:

    f = max(5.0, min(100.0, float(f_mhz)))

    return ((_RX2_A*f + _RX2_B)*f + _RX2_C)*f + _RX2_D



def atten_db_lookup(rx_idx: int, f_mhz: float) -> float:

    if rx_idx == 2:

        return _atten_rx2_poly_db(f_mhz)

    return _interp_att(ATT_LUT_RX3, f_mhz)



#  TX helpers

def set_mixer(tx, freq_mhz: float):

    try:

        tx.dac_block.MixerSettings["Freq"] = float(freq_mhz)

        for k,v in [("PhaseOffset",0.0),("EventSource",0)]:

            try: tx.dac_block.MixerSettings[k]=v

            except: pass

    except: pass



def select_amplitude_source(tx):

    for attr,val in [("source","amplitude"),("data_sel","amplitude"),("data_source","amplitude")]:

        try:

            setattr(tx,attr,val)

            return

        except:

            pass



def write_I_only_gain(ctrl, gain_0_to_1: float):

    scaled=int(round(max(0.0,min(1.0,gain_0_to_1))*0x7FFF))

    ctrl.write(0x04, (scaled&0xFFFF)|(0<<16))



def apply_tx_config(tx, freq_mhz: float, request_val: float, request_unit: str, use_poly=True):

    req_vpp=volts_to_vpp(request_val,request_unit)

    scale=cal_scale_poly(freq_mhz)

    target_vpp=req_vpp*scale

    gain=max(0.0,min(1.0,target_vpp/DAC_VFS_VPP_I_ONLY))

    tx.control.enable=False

    select_amplitude_source(tx)

    set_mixer(tx,freq_mhz)

    try:

        write_I_only_gain(tx.control,gain)

    except:

        tx.control.gain=float(gain)

    tx.control.enable=True

    return {"req_vpp":req_vpp,"scale":scale,"target_vpp":target_vpp,"gain":gain}



#  ADC helpers

def normalize_adc_if_needed(y: np.ndarray):

    m=float(np.max(np.abs(y)))

    if m>2.5:

        return (y.astype(np.float32)/RAW_FS_COUNTS),True

    return y.astype(np.float32),False



def remove_dc(y): return y-np.mean(y)

def measure_vpp_fs(y): return float(np.max(y)-np.min(y))

def vrms_from_vpp(v): return v/(2.0*math.sqrt(2.0))



def get_ddc_scale_if_any(rx)->float:

    candidates=[]

    for path in [("adc_block","FineMixerScale"),("adc_block","MixerScale"),

                 ("adc_block","DecimationGain"),("block","DecimationGain"),("control","gain")]:

        try:

            obj=rx

            for a in path:

                obj=getattr(obj,a)

            v=float(obj)

            if v>0.0:

                candidates.append(v)

        except:

            pass

    if not candidates:

        return 1.0

    return float(sorted(candidates)[-1])



def vpp_fs_to_volts(vpp_fs,fs_vpp_est,ddc_scale=1.0):

    return vpp_fs*(fs_vpp_est/2.0)*ddc_scale



#  UI panes

class TxPane(QtWidgets.QGroupBox):

    def __init__(self,title,tx_idx,init_freq,cal_enabled_getter):

        super().__init__(title)

        self.tx_idx=tx_idx

        self._get_use_poly=cal_enabled_getter

        g=QtWidgets.QGridLayout(self)



        self.enable=QtWidgets.QCheckBox("Enable")

        self.enable.setChecked(True)

        g.addWidget(self.enable,0,0)



        g.addWidget(QtWidgets.QLabel("Freq (MHz)"),0,1)

        self.freq=QtWidgets.QDoubleSpinBox()

        self.freq.setRange(0.0,6000.0)

        self.freq.setDecimals(6)

        self.freq.setSingleStep(0.01)

        self.freq.setValue(init_freq)

        g.addWidget(self.freq,0,2)



        g.addWidget(QtWidgets.QLabel("Amplitude"),1,0)

        self.amp=QtWidgets.QDoubleSpinBox()

        self.amp.setDecimals(4)

        self.amp.setRange(0.0,5.0)

        self.amp.setValue(0.50)

        g.addWidget(self.amp,1,1)



        self.unit=QtWidgets.QComboBox()

        self.unit.addItems(["Vpp","Vp","Vrms"])

        g.addWidget(self.unit,1,2)



        self.apply=QtWidgets.QPushButton("Apply TX")

        g.addWidget(self.apply,0,3,2,1)

        self.apply.clicked.connect(self.apply_now)



        self.status=QtWidgets.QLabel("")

        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        g.addWidget(self.status,2,0,1,4)



        self._last=None



    def apply_now(self):

        tx=base.radio.transmitter.channel[self.tx_idx]

        freq=self.freq.value()

        if not self.enable.isChecked():

            tx.control.enable=False

            self.status.setText(f"TX{self.tx_idx} disabled.")

            return

        info=apply_tx_config(

            tx,

            freq_mhz=freq,

            request_val=self.amp.value(),

            request_unit=self.unit.currentText(),

            use_poly=self._get_use_poly(),

        )

        self._last=info

        self.status.setText(

            f"TX{self.tx_idx} @ {freq:.6f} MHz, {self.amp.value():.4g} {self.unit.currentText()} applied."

        )



#  Main widget

class Main(QtWidgets.QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("RFSoC 4x2 — Dual Scope + Auto NCO Tracking")

        self.resize(1520,820)

        pg.setConfigOptions(antialias=False)



        self.fs_hz=self._detect_fs_hz(FS_HZ_FALLBACK)

        self.nco_track={2:0.0,3:0.0}



        root=QtWidgets.QVBoxLayout(self)



        # One 2-column grid: row0=TX panes, row1=plots, row2=buttons, row3=readouts

        grid=QtWidgets.QGridLayout()

        grid.setColumnStretch(0,1)

        grid.setColumnStretch(1,1)

        root.addLayout(grid,1)



        # TX panes aligned above each plot

        self.tx0=TxPane("TX0→RX2",0,20.00,lambda: True)

        self.tx1=TxPane("TX1→RX3",1,20.00,lambda: True)

        grid.addWidget(self.tx0,0,0)

        grid.addWidget(self.tx1,0,1)



        self.tx0.apply_now()

        self.tx1.apply_now()

        for tx_idx,_ in TX_TO_RX.items():

            self._set_rx_nco_for(tx_idx)



        # Time base (µs)

        self.t_us=(np.arange(N_SAMPLES)/self.fs_hz)*1e6

        self.t_us_dec=self.t_us[::max(1,DECIM)]



        self.curves,self.readouts = {},{}

        self.plots = {}

        self.pause_btns, self.paused = {}, {}

        self.last_wave = {}

        
# Plots + buttons + readouts

        for col,rx_idx in enumerate(RX_LIST):

            pw=pg.PlotWidget(title=f"ADC {rx_idx} — Auto NCO Tracking")

            pw.showGrid(x=True,y=True,alpha=0.25)

            pw.getAxis("bottom").enableAutoSIPrefix(False)

            pw.setLabel("bottom","Time","μs")

            pw.setLabel("left","Amplitude (V)")

            c=pw.plot(self.t_us_dec,np.zeros_like(self.t_us_dec),pen=pg.mkPen(width=1.3))

            grid.addWidget(pw,1,col)

            self.curves[rx_idx]=c

            self.plots[rx_idx]=pw



            # Autoset + Pause/Resume buttons

            btn_row = QtWidgets.QHBoxLayout()

            autoset = QtWidgets.QPushButton("Autoset")

            pause   = QtWidgets.QPushButton("Pause")

            btn_row.addWidget(autoset)

            btn_row.addWidget(pause)

            btn_row.addStretch(1)

            grid.addLayout(btn_row,2,col)



            autoset.clicked.connect(lambda _=None, r=rx_idx: self._autoset(r))

            pause.clicked.connect(lambda _=None, r=rx_idx: self._toggle_pause(r))

            self.pause_btns[rx_idx] = pause

            self.paused[rx_idx] = False



            ro=QtWidgets.QLabel("Vpp≈0.000 V  Vrms≈0.000 V  Freq≈— MHz")

            grid.addWidget(ro,3,col,alignment=QtCore.Qt.AlignLeft)

            self.readouts[rx_idx]=ro



        self.timer=QtCore.QTimer(self)

        self.timer.setInterval(INTERVAL_MS)

        self.timer.timeout.connect(self.update_once)

        self.timer.start()



    #  Autoset / Pause

    def _autoset(self, rx_idx: int):

        """

        Zoom to a clear waveform view:

          - Use last decimated waveform

          - Estimate frequency

          - Show ~6 cycles (or a minimum window)

          - Scale Y around ±(amp * 1.15)

        Falls back to autoRange if anything looks weird.

        """

        try:

            t_us, y = self.last_wave.get(rx_idx, (None, None))

            if t_us is None or y is None or len(y) < 16:

                self.plots[rx_idx].autoRange()

                return



            # Estimate frequency from the decimated data

            fs_dec_hz = self.fs_hz / max(1, DECIM)

            y0 = y - np.mean(y)

            f_est_hz = self._estimate_freq_hz(y0, fs_dec_hz)

            if not np.isfinite(f_est_hz) or f_est_hz <= 0:

                self.plots[rx_idx].autoRange()

                return



            # Window ~6 cycles, with a minimum fraction of the record

            n_cycles = 6.0

            T_us = (1.0 / f_est_hz) * 1e6

            win_us = max(T_us * n_cycles, (t_us[-1] - t_us[0]) * 0.08)

            t_end = float(t_us[-1])

            t_start = max(float(t_us[0]), t_end - win_us)



            # Robust amplitude estimate + padding

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

            try:

                self.plots[rx_idx].autoRange()

            except Exception:

                pass



    def _toggle_pause(self, rx_idx: int):

        self.paused[rx_idx] = not self.paused[rx_idx]

        self.pause_btns[rx_idx].setText("Resume" if self.paused[rx_idx] else "Pause")



        txt_parts = self.readouts[rx_idx].text().split("   ")

        if self.paused[rx_idx]:

            if not any(s.startswith("PAUSED") for s in txt_parts):

                txt_parts.insert(0, "PAUSED")

        else:

            txt_parts = [s for s in txt_parts if not s.startswith("PAUSED")]

        self.readouts[rx_idx].setText("   ".join(txt_parts))



    #  Helpers

    def _auto_track_rx_nco(self, rx_idx:int, f_bb_hz:float):

        """Smoothly update RX NCO to follow external signal frequency."""

        try:

            current=float(RX[rx_idx].adc_block.MixerSettings["Freq"])

            target=current + (f_bb_hz/1e6)

            prev=self.nco_track[rx_idx]

            new=0.9*prev + 0.1*target

            step=max(-0.05,min(0.05,new-current))  # ±50 kHz per frame

            RX[rx_idx].adc_block.MixerSettings["Freq"]=current+step

            self.nco_track[rx_idx]=current+step

        except Exception as e:

            print(f"NCO track error RX{rx_idx}:",e)



    def _set_rx_nco_for(self, tx_idx:int):

        rx_idx = TX_TO_RX[tx_idx]

        try:

            pane = self._txpane_for_tx(tx_idx)

            rx = RX[rx_idx]

            if pane.enable.isChecked():

               f_mhz = float(pane.freq.value())

               offset_mhz = (NCO_OFFSET_HZ.get(rx_idx, 10_000.0)) / 1e6

               rx.adc_block.MixerSettings["Freq"] = f_mhz - offset_mhz

            else:

               rx.adc_block.MixerSettings["Freq"] = 0.0

               print(f"RX{rx_idx} NCO bypass for external input.")

        except Exception as e:

            print(f"RX{rx_idx} NCO set failed:", e)

    def _txpane_for_tx(self,tx_idx):

        return self.tx0 if tx_idx==0 else self.tx1



    def _txpane_for_rx(self,rx_idx):

        for tx_idx,rxi in TX_TO_RX.items():

            if rxi==rx_idx:

                return self._txpane_for_tx(tx_idx)

        return self.tx0



    def _detect_fs_hz(self,fallback):

        cand=[]

        try:

            for rx_idx in RX_LIST:

                rx=RX[rx_idx]

                for path in [("adc_block","SamplingRate"),("adc_block","SampleRate"),

                             ("adc_block","Fs"),("adc_block","Fs_Hz"),

                             ("block","SamplingRate"),("block","SampleRate"),

                             ("control","SamplingRate")]:

                    try:

                        obj=rx

                        for a in path:

                            obj=getattr(obj,a)

                        val=float(obj)

                        if val>1e3:

                            cand.append(val)

                    except:

                        pass

        except:

            pass

        if cand:

            try:

                return float(sorted(cand)[-1])

            except:

                pass

        return float(fallback)



    def _estimate_freq_hz(self,y_fs,fs_hz):

        if len(y_fs)<4:

            return 0.0

        y=y_fs-np.mean(y_fs)

        Y=np.fft.rfft(y*np.hanning(len(y)))

        mag=np.abs(Y)

        if len(mag)<=1:

            return 0.0

        k=int(np.argmax(mag[1:]))+1

        return float(k*fs_hz/len(y))



    def update_once(self):

        try:

            fs_vpp = ASSUMED_FS_VPP  # fixed

            for rx_idx in RX_LIST:

                # Skip updates if paused

                if self.paused.get(rx_idx, False):

                    continue



                pane=self._txpane_for_rx(rx_idx)



                x=RX[rx_idx].transfer(N_SAMPLES)

                y=np.real(x)

                y_fs,_=normalize_adc_if_needed(y)

                y0=remove_dc(y_fs)
#####
                ddc_scale = get_ddc_scale_if_any(RX[rx_idx])

                y_volts_adc = y0 * (fs_vpp / 2.0) * ddc_scale  # ADC-pin volts



                # Determine RF for info; choose mode

                f_est_hz = self._estimate_freq_hz(y0, self.fs_hz)

                f_rf_mhz = (ADC_NCO_HZ - f_est_hz)/1e6

                atten_db = atten_db_lookup(rx_idx, f_rf_mhz)

                beta = 10.0 ** (atten_db / 20.0)

                if not pane.enable.isChecked():

                	y_src = y_volts_adc * beta  # source (post-poly)

                else:

                	y_src = (y_volts_adc * beta) / max(1e-9, cal_scale_poly(f_rf_mhz))


###


                # Plot (decimated)

                y_plot=y_src[::DECIM] if DECIM > 1 else y_src

                self.curves[rx_idx].setData(self.t_us_dec,y_plot,_callSync="off")



                # Save for Autoset

                self.last_wave[rx_idx] = (self.t_us_dec, y_plot)



                # Readouts

                vpp_src=measure_vpp_fs(y_src)

                vrms_src=vrms_from_vpp(vpp_src)

                self.readouts[rx_idx].setText(

                    f"Vpp≈{vpp_src:.3f} V   Vrms≈{vrms_src:.3f} V   f_RF≈{f_rf_mhz:.3f} MHz"

                )

        except Exception as e:

            print("RX update error:",e)



    def _get_rx_nco(self,rx_idx):

        try:

            return float(RX[rx_idx].adc_block.MixerSettings["Freq"])

        except:

            return 0.0



    def closeEvent(self,ev):

        try:

            for ch in base.radio.transmitter.channel:

                ch.control.enable=False

        except:

            pass

        super().closeEvent(ev)



#  Bring-up

base=BaseOverlay("base.bit")

base.init_rf_clks()

TX=base.radio.transmitter.channel

RX=base.radio.receiver.channel



if __name__=="__main__":

    app=QtWidgets.QApplication([])

    w=Main();w.show()

    app.exec()


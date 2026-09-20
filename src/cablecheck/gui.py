"""
Small Tkinter interface for lab staff: pick files, choose the cable type,
fill in the measurement context, press Evaluate, read the verdict, open the
report, store the run.  Uses only the standard library plus matplotlib's Tk
backend, so it runs on any lab PC with Python installed.
"""
from __future__ import annotations

import threading
import traceback
import webbrowser
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as e:  # pragma: no cover
    raise SystemExit("tkinter is not available in this Python installation") from e

from .cli import _default_port_map
from .limits.library import list_cable_types
from .pipeline import FixtureSpec, MeasurementFile, SampleInfo, run_sample, write_json
from .report.plots import TITLES, plot_result

C_PASS, C_FAIL, C_INFO = "#008300", "#e34948", "#52514e"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cablecheck - cable measurement evaluation")
        self.geometry("1180x760")
        self.result = None
        self.files: list[str] = []
        self._build()

    # ------------------------------------------------------------ layout
    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        # files
        ff = ttk.LabelFrame(top, text="Measurement files (Touchstone)", padding=6)
        ff.grid(row=0, column=0, sticky="nsew", padx=4)
        self.lb = tk.Listbox(ff, height=5, width=48)
        self.lb.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(ff, text="Add files...", command=self.add_files).grid(row=1, column=0, sticky="w", pady=3)
        ttk.Button(ff, text="Remove", command=self.remove_file).grid(row=1, column=1, sticky="w")
        ttk.Label(ff, text="Port map:").grid(row=2, column=0, sticky="w")
        self.port_map = tk.StringVar(value="auto")
        ttk.Entry(ff, textvariable=self.port_map, width=34).grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Label(ff, text="Fixture:").grid(row=3, column=0, sticky="w")
        self.fixture_mode = tk.StringVar(value="none")
        ttk.Combobox(ff, textvariable=self.fixture_mode, values=["none", "files", "2xthru"], width=8, state="readonly").grid(row=3, column=1, sticky="w")
        self.fixture_path = tk.StringVar()
        ttk.Entry(ff, textvariable=self.fixture_path, width=26).grid(row=3, column=2, sticky="w")
        ttk.Button(ff, text="...", width=3, command=self.pick_fixture).grid(row=3, column=3, sticky="w")
        # sample
        sf = ttk.LabelFrame(top, text="Sample and context", padding=6)
        sf.grid(row=0, column=1, sticky="nsew", padx=4)
        self.vars = {}
        fields = [("sample_id", "Sample ID"), ("length_m", "Length / m"), ("lot", "Lot"), ("part_number", "Part number"),
                  ("operator", "Operator"), ("instrument", "Instrument"), ("calibration_date", "Calibration date"),
                  ("temperature_c", "Temperature / degC"), ("site", "Site"), ("notes", "Notes")]
        for i, (k, lab) in enumerate(fields):
            ttk.Label(sf, text=lab).grid(row=i % 5, column=2 * (i // 5), sticky="w", padx=(0, 4))
            v = tk.StringVar()
            self.vars[k] = v
            ttk.Entry(sf, textvariable=v, width=20).grid(row=i % 5, column=2 * (i // 5) + 1, sticky="w", padx=(0, 10))
        ttk.Label(sf, text="Cable type / limits").grid(row=5, column=0, sticky="w")
        self.types = {ct.id: ct for ct in list_cable_types()}
        self.cable_type = tk.StringVar(value=next(iter(self.types), ""))
        ttk.Combobox(sf, textvariable=self.cable_type, values=list(self.types), width=38, state="readonly").grid(row=5, column=1, columnspan=3, sticky="w")
        # actions
        af = ttk.Frame(top, padding=6)
        af.grid(row=0, column=2, sticky="ns")
        self.btn_eval = ttk.Button(af, text="Evaluate", command=self.evaluate)
        self.btn_eval.pack(fill="x", pady=2)
        ttk.Button(af, text="Save HTML report...", command=self.save_html).pack(fill="x", pady=2)
        ttk.Button(af, text="Save PDF report...", command=self.save_pdf).pack(fill="x", pady=2)
        ttk.Button(af, text="Save JSON...", command=self.save_json).pack(fill="x", pady=2)
        ttk.Button(af, text="Store in database...", command=self.store_db).pack(fill="x", pady=2)
        self.verdict = tk.Label(af, text="-", font=("Helvetica", 26, "bold"), fg=C_INFO)
        self.verdict.pack(pady=8)
        self.headline = tk.Label(af, text="", wraplength=220, justify="left")
        self.headline.pack()
        # results + plot
        mid = ttk.PanedWindow(self, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=8, pady=4)
        cols = ("quantity", "pair", "verdict", "margin", "at", "measured", "limit")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=18)
        for c, w in zip(cols, (200, 60, 60, 100, 90, 90, 80)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.tag_configure("pass", foreground=C_PASS)
        self.tree.tag_configure("fail", foreground=C_FAIL)
        self.tree.tag_configure("info", foreground=C_INFO)
        self.tree.bind("<<TreeviewSelect>>", self.show_plot)
        mid.add(self.tree, weight=1)
        self.plot_frame = ttk.Frame(mid)
        mid.add(self.plot_frame, weight=2)
        self.canvas = None
        self.status = tk.StringVar(value="ready")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=(0, 6))

    # ----------------------------------------------------------- actions
    def add_files(self):
        for p in filedialog.askopenfilenames(filetypes=[("Touchstone", "*.s?p *.s??p"), ("all", "*")]):
            self.files.append(p)
            self.lb.insert("end", Path(p).name)

    def remove_file(self):
        for i in reversed(self.lb.curselection()):
            self.lb.delete(i)
            del self.files[i]

    def pick_fixture(self):
        p = filedialog.askopenfilename(filetypes=[("Touchstone", "*.s?p"), ("all", "*")])
        if p:
            self.fixture_path.set(p)

    def _sample(self) -> SampleInfo:
        v = {k: s.get().strip() for k, s in self.vars.items()}
        return SampleInfo(sample_id=v["sample_id"] or "unnamed", cable_type=self.cable_type.get(),
                          length_m=float(v["length_m"]) if v["length_m"] else None, lot=v["lot"],
                          part_number=v["part_number"], operator=v["operator"], instrument=v["instrument"],
                          calibration_date=v["calibration_date"],
                          temperature_c=float(v["temperature_c"]) if v["temperature_c"] else None,
                          site=v["site"], notes=v["notes"])

    def _files(self) -> list[MeasurementFile]:
        from .io import read_touchstone
        mode = self.fixture_mode.get()
        fx = FixtureSpec()
        if mode == "files":
            fx = FixtureSpec("files", left=self.fixture_path.get(), right=self.fixture_path.get())
        elif mode == "2xthru":
            fx = FixtureSpec("2xthru", thru=self.fixture_path.get())
        out = []
        for p in self.files:
            pm = self.port_map.get().strip()
            if pm in ("", "auto"):
                pm = _default_port_map(read_touchstone(p).nports)
            out.append(MeasurementFile(p, pm, fx))
        return out

    def evaluate(self):
        if not self.files:
            messagebox.showwarning("cablecheck", "Add at least one measurement file.")
            return
        self.btn_eval.state(["disabled"])
        self.status.set("evaluating...")
        sample, files = self._sample(), self._files()

        def work():
            try:
                res = run_sample(sample, files)
                self.after(0, lambda: self._done(res))
            except Exception as e:
                tb = traceback.format_exc()
                # bind by default argument: `e` is unbound at the end of the except block, and the
                # callback runs later on the Tk thread
                self.after(0, lambda exc=e, tb=tb: self._fail(exc, tb))
        threading.Thread(target=work, daemon=True).start()

    def _fail(self, e, tb):
        self.btn_eval.state(["!disabled"])
        self.status.set(f"error: {e}")
        messagebox.showerror("cablecheck", f"{e}\n\n{tb[-1500:]}")

    def _done(self, res):
        self.result = res
        self.btn_eval.state(["!disabled"])
        ev = res.evaluation
        self.verdict.config(text=ev.verdict, fg={"PASS": C_PASS, "FAIL": C_FAIL}.get(ev.verdict, C_INFO))
        h = ev.headline
        if h:
            at = f"{h.x_worst / 1e6:.1f} MHz" if h.xunit == "Hz" and h.x_worst else ""
            self.headline.config(text=f"{TITLES.get(h.quantity, h.quantity)} [{h.pair}]\nworst margin {h.worst_margin:+.2f} {h.unit} {at}")
        for i in self.tree.get_children():
            self.tree.delete(i)
        for k, r in enumerate(ev.results):
            tag = "info" if r.passed is None else ("pass" if r.passed else "fail")
            word = "info" if r.passed is None else ("PASS" if r.passed else "FAIL")
            at = "" if r.x_worst is None else (f"{r.x_worst / 1e6:.2f} MHz" if r.xunit == "Hz" else f"{r.x_worst:.2f} m")
            meas = "" if r.value_at_worst is None else f"{r.value_at_worst:.2f} {r.unit}"
            if r.is_scalar and r.value_at_worst is None:
                meas = f"{float(r.value[0]):.3f} {r.unit}"
            self.tree.insert("", "end", iid=str(k), values=(TITLES.get(r.quantity, r.quantity), r.pair, word,
                                                             "" if r.worst_margin is None else f"{r.worst_margin:+.2f} {r.unit}",
                                                             at, meas, "" if r.limit_at_worst is None else f"{r.limit_at_worst:.2f}"),
                             tags=(tag,))
        warns = res.warnings + ev.warnings
        self.status.set(f"{ev.verdict}; {len(warns)} warning(s)" + (": " + warns[0] if warns else ""))

    def show_plot(self, _event=None):
        if self.result is None or not self.tree.selection():
            return
        r = self.result.evaluation.results[int(self.tree.selection()[0])]
        if r.is_scalar:
            return
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig = plot_result(r)
        fig.set_size_inches(6.4, 4.2)
        fig.tight_layout()
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _need_result(self) -> bool:
        if self.result is None:
            messagebox.showinfo("cablecheck", "Evaluate first.")
            return False
        return True

    def save_html(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(defaultextension=".html", initialfile=f"{self.result.sample.sample_id}.html")
        if p:
            from .report.html import write_html
            write_html(self.result, p)
            webbrowser.open(Path(p).as_uri())

    def save_pdf(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=f"{self.result.sample.sample_id}.pdf")
        if p:
            from .report.pdf import write_pdf
            write_pdf(self.result, p)
            self.status.set(f"wrote {p}")

    def save_json(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(defaultextension=".json", initialfile=f"{self.result.sample.sample_id}.json")
        if p:
            write_json(self.result, p)
            self.status.set(f"wrote {p}")

    def store_db(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(defaultextension=".sqlite", initialfile="results.sqlite", confirmoverwrite=False)
        if p:
            from .db import ResultsDB
            with ResultsDB(p) as db:
                rid = db.insert_run(self.result)
            self.status.set(f"stored run {rid} in {p}")


def main():
    App().mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()

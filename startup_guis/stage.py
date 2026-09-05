"""Stage and beam jog panel driven through the MCP bridge.

Talks only to the MCP server (no Tango client): instrument commands are the
``<Class>_<command>`` tools the bridge registers. The instrument class is
discovered from whichever class exposes ``get_stage``. Point it at another
bridge with ``ASYNCROSCOPY_MCP_URL`` (default: the local digital-twin bridge).
"""

import asyncio
import os
import tkinter as tk
from tkinter import ttk, messagebox

from fastmcp import Client

DEFAULT_MCP_URL = os.environ.get("ASYNCROSCOPY_MCP_URL", "http://127.0.0.1:8000/mcp")

INSTRUMENT_COMMANDS = (
    "get_stage", "move_stage", "get_defocus", "set_defocus", "get_image_shift", "set_image_shift",
    "get_beam_tilt", "set_beam_tilt", "auto_focus",
)


def resolve_instrument_tools(tool_names, instrument_class=None) -> dict[str, str]:
    """Map instrument commands to the bridge's ``<Class>_<command>`` tool names.

    With ``instrument_class`` unset, the class is the one exposing ``get_stage``
    (device-settings classes such as SCAN or STAGE do not).
    """
    names = set(tool_names)
    if instrument_class is None:
        candidates = sorted(name[: -len("_get_stage")] for name in names if name.endswith("_get_stage"))
        if not candidates:
            raise RuntimeError("No instrument tools found on the MCP server (no *_get_stage tool)")
        if len(candidates) > 1:
            raise RuntimeError(f"Several instrument classes expose get_stage: {candidates}; set instrument_class")
        instrument_class = candidates[0]
    return {
        command: f"{instrument_class}_{command}"
        for command in INSTRUMENT_COMMANDS
        if f"{instrument_class}_{command}" in names
    }


class MCPMicroscope:
    """Synchronous facade over the bridge's instrument tools for a Tk GUI."""

    def __init__(self, url: str = DEFAULT_MCP_URL, instrument_class: str | None = None):
        self.url = url
        self.instrument_class = instrument_class
        self.tool_names: dict[str, str] = {}

    def _run(self, coro):
        return asyncio.run(coro)

    def connect(self) -> None:
        async def go():
            async with Client(self.url) as client:
                return [tool.name for tool in await client.list_tools()]

        self.tool_names = resolve_instrument_tools(self._run(go()), self.instrument_class)

    def has(self, command: str) -> bool:
        return command in self.tool_names

    def call(self, command: str, arg=None):
        tool = self.tool_names.get(command)
        if tool is None:
            raise RuntimeError(f"The MCP server exposes no {command!r} tool for this instrument")
        arguments = {} if arg is None else {"arg": arg}

        async def go():
            async with Client(self.url) as client:
                result = await client.call_tool(tool, arguments)
                return result.data if result.data is not None else result.structured_content

        return self._run(go())


class AdvancedMicroscopeGUI:
    def __init__(self, root, microscope: MCPMicroscope | None = None):
        self.root = root
        self.root.title("Advanced Microscope Stage & Beam Controller")
        self.root.geometry("900x530")
        self.root.configure(bg="#222222")

        self.saved_stage_positions = {}

        self.microscope = microscope or MCPMicroscope()
        try:
            self.microscope.connect()
        except Exception as exc:
            messagebox.showerror("MCP", f"Could not reach the MCP server at {self.microscope.url}:\n{exc}")
            raise

        # --- Internal Coordinate States ---
        self.stage_x, self.stage_y, self.stage_z = 0.0, 0.0, 0.0
        self.stage_alpha, self.stage_beta = 0.0, 0.0
        self.beam_x, self.beam_y, self.defocus = 0.0, 0.0, 0.0
        self.beam_alpha, self.beam_beta = 0.0, 0.0

        # --- StringVars for UI Inputs ---
        self.x_var, self.y_var, self.z_var = tk.StringVar(value="0.00"), tk.StringVar(value="0.00"), tk.StringVar(value="0.00")
        self.alpha_var, self.beta_var = tk.StringVar(value="0.00"), tk.StringVar(value="0.00")
        self.bsx_var, self.bsy_var, self.defocus_var = tk.StringVar(value="0.00"), tk.StringVar(value="0.00"), tk.StringVar(value="0.00")
        self.btx_var, self.bty_var = tk.StringVar(value="0.00"), tk.StringVar(value="0.00")

        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure('TLabel', background='#222222', foreground='#ffffff', font=("Arial", 9, "bold"))

        # 1. TOP SECTION: EDITABLE DIGITAL READOUTS
        self.readout_frame = tk.LabelFrame(root, text=" Target Parameters (Editable) ", bg="#161616", fg="#ffffff", font=("Arial", 10, "bold"), padx=10, pady=10)
        self.readout_frame.pack(fill="x", padx=15, pady=10)

        self.create_grid_field(self.readout_frame, "Stage X:", self.x_var, "nm", 0, 0)
        self.create_grid_field(self.readout_frame, "Stage Y:", self.y_var, "nm", 0, 3)
        self.create_grid_field(self.readout_frame, "Stage Z:", self.z_var, "nm", 0, 6)
        self.create_grid_field(self.readout_frame, "Tilt α:", self.alpha_var, "deg", 0, 9)
        self.create_grid_field(self.readout_frame, "Tilt β:", self.beta_var, "deg", 0, 12)

        self.create_grid_field(self.readout_frame, "Beam S-X:", self.bsx_var, "nm", 1, 0)
        self.create_grid_field(self.readout_frame, "Beam S-Y:", self.bsy_var, "nm", 1, 3)
        self.create_grid_field(self.readout_frame, "Defocus Δf:", self.defocus_var, "nm", 1, 6)
        self.create_grid_field(self.readout_frame, "Beam T-X:", self.btx_var, "mrad", 1, 9)
        self.create_grid_field(self.readout_frame, "Beam T-Y:", self.bty_var, "mrad", 1, 12)

        btn_go = tk.Button(self.readout_frame, text="EXECUTE ALL ABSOLUTE DRIVES 🚀", font=("Arial", 10, "bold"), bg="#28a745", fg="white", command=self.go_to_absolute)
        btn_go.grid(row=2, column=0, columnspan=15, pady=8, padx=5, sticky="ew")

        # 2. MIDDLE SECTION: MANUAL CONTROL PANELS
        self.main_control_area = tk.Frame(root, bg="#222222")
        self.main_control_area.pack(fill="both", expand=True, padx=15, pady=5)

        self.stage_block = tk.LabelFrame(self.main_control_area, text=" Mechanical Stage System ", bg="#2b2b2b", fg="#00ffcc", font=("Arial", 11, "bold"), padx=5, pady=5)
        self.stage_block.pack(side="left", fill="both", expand=True, padx=5)

        self.panel_tilt = tk.LabelFrame(self.stage_block, text=" Stage Tilt (α/β) ", bg="#2b2b2b", fg="white", font=("Arial", 9))
        self.panel_tilt.grid(row=0, column=0, padx=5, pady=5)
        self.build_dpad(self.panel_tilt, "α+", "α-", "β-", "β+", lambda mx, my: self.jog_engine(mb=mx, ma=my), reset_cb=self.reset_stage_tilt)

        self.panel_xy = tk.LabelFrame(self.stage_block, text=" Stage XY ", bg="#2b2b2b", fg="white", font=("Arial", 9))
        self.panel_xy.grid(row=0, column=1, padx=5, pady=5)
        self.build_dpad(self.panel_xy, "Y+", "Y-", "X-", "X+", lambda mx, my: self.jog_engine(mx=mx, my=my), reset_cb=self.reset_stage_xy)

        self.panel_z = tk.LabelFrame(self.stage_block, text=" Stage Z ", bg="#2b2b2b", fg="white", font=("Arial", 9), padx=5)
        self.panel_z.grid(row=0, column=2, padx=5, pady=5, sticky="ns")
        tk.Button(self.panel_z, text="⏫ Z+", font=("Arial", 10), width=5, command=lambda: self.jog_engine(mz=1)).pack(pady=5)
        tk.Button(self.panel_z, text="🔄 H", font=("Arial", 9), width=5, bg="#444444", fg="white", command=self.reset_stage_z).pack(pady=5)
        tk.Button(self.panel_z, text="⏬ Z-", font=("Arial", 10), width=5, command=lambda: self.jog_engine(mz=-1)).pack(pady=5)

        self.center_divider = tk.Frame(self.main_control_area, bg="#222222")
        self.center_divider.pack(side="left", padx=10, fill="y")
        btn_master_home = tk.Button(self.center_divider, text="🌐\n\nH\nO\nM\nE\n\nA\nL\nL", font=("Arial", 10, "bold"), bg="#bd2130", fg="white", width=4, relief="raised", command=self.reset_all)
        btn_master_home.pack(fill="y", expand=True, pady=10)

        self.beam_block = tk.LabelFrame(self.main_control_area, text=" Electronic Beam Deflection ", bg="#232d37", fg="#ffcc00", font=("Arial", 11, "bold"), padx=5, pady=5)
        self.beam_block.pack(side="right", fill="both", expand=True, padx=5)

        self.panel_btilt = tk.LabelFrame(self.beam_block, text=" Beam Tilt (X/Y) ", bg="#232d37", fg="white", font=("Arial", 9))
        self.panel_btilt.grid(row=0, column=0, padx=5, pady=5)
        self.build_dpad(self.panel_btilt, "TY+", "TY-", "TX-", "TX+", lambda mx, my: self.jog_engine(mbtx=mx, mbty=my), reset_cb=self.reset_beam_tilt)

        self.panel_bshift = tk.LabelFrame(self.beam_block, text=" Beam Shift (X/Y) ", bg="#232d37", fg="white", font=("Arial", 9))
        self.panel_bshift.grid(row=0, column=1, padx=5, pady=5)
        self.build_dpad(self.panel_bshift, "BSY+", "BSY-", "BSX-", "BSX+", lambda mx, my: self.jog_engine(mbsx=mx, mbsy=my), reset_cb=self.reset_beam_shift)

        self.panel_df = tk.LabelFrame(self.beam_block, text=" Defocus (Δf) ", bg="#232d37", fg="white", font=("Arial", 9), padx=5)
        self.panel_df.grid(row=0, column=2, padx=5, pady=5, sticky="ns")
        tk.Button(self.panel_df, text="⏫ Δf+", font=("Arial", 10), width=5, command=lambda: self.jog_engine(mdf=1)).pack(pady=5)
        tk.Button(self.panel_df, text="🔄 H", font=("Arial", 9), width=5, bg="#444444", fg="white", command=self.reset_defocus).pack(pady=5)
        tk.Button(self.panel_df, text="⏬ Δf-", font=("Arial", 10), width=5, command=lambda: self.jog_engine(mdf=-1)).pack(pady=5)

        # 3. BOTTOM SECTION: STEP SIZE CONFIGS
        self.step_frame = tk.LabelFrame(root, text=" Incremental Step Sizes ", bg="#222222", fg="#ffffff", font=("Arial", 9, "bold"), padx=10, pady=5)
        self.step_frame.pack(fill="x", padx=15, pady=10)

        self.stg_lin_var = tk.StringVar(value="1.00")
        self.stg_ang_var = tk.StringVar(value="0.50")
        self.bm_sft_var = tk.StringVar(value="0.10")
        self.bm_tlt_var = tk.StringVar(value="0.50")

        self.add_step_selector(self.step_frame, "Stage Lin:", self.stg_lin_var, ["1.0", "10.0", "100.0", "1000.0"], "nm", 0)
        self.add_step_selector(self.step_frame, "Stage Tilt:", self.stg_ang_var, ["0.05", "0.10", "0.50", "2.00"], "deg", 3)
        self.add_step_selector(self.step_frame, "Beam Shift/Δf:", self.bm_sft_var, ["0.10", "1.00", "10.00", "100.00"], "nm", 6)
        self.add_step_selector(self.step_frame, "Beam Tilt:", self.bm_tlt_var, ["0.1", "1", "10", "100"], "mrad", 9)

        tk.Button(self.step_frame, text='Save_Stage', bg="#222222", fg="#ffffff", command=self.save_position).grid(row=1, column=0, padx=2)
        self.position_dropdown = ttk.Combobox(self.step_frame, state="readonly")
        self.position_dropdown.grid(row=1, column=1, padx=2)
        tk.Button(self.step_frame, text='Go To', bg="#222222", fg="#ffffff", command=self.go_to_position).grid(row=1, column=2, padx=2)
        tk.Button(self.step_frame, text='autoFocus', bg="#222222", fg="#ffffff", command=self.auto_focus).grid(row=1, column=3, padx=2)

        tk.Label(self.step_frame, text=f"MCP: {self.microscope.url}", bg="#222222", fg="#888888").grid(row=1, column=4, padx=8)

        self.current_conditions()
        self.refresh_entries()

    # --- Instrument access (all through MCP tools) ---
    def auto_focus(self):
        if self.microscope.has('auto_focus'):
            self.microscope.call('auto_focus')
        else:
            messagebox.showinfo("Auto Focus", "This instrument exposes no auto_focus tool.")

    def save_position(self):
        from tkinter import simpledialog
        positions = list(self.microscope.call('get_stage'))
        name = simpledialog.askstring("Save Position", "Enter a name for this position:")
        if not name:
            return
        self.saved_stage_positions.setdefault(name, positions)
        self.position_dropdown.config(values=list(self.saved_stage_positions.keys()))
        self.position_dropdown.set(name)

    def go_to_position(self):
        name = self.position_dropdown.get()
        if name in self.saved_stage_positions:
            positions = self.saved_stage_positions[name]
            self.stage_x, self.stage_y, self.stage_z = [float(v) * 1e9 for v in positions[:3]]
            self.stage_alpha = float(positions[3])
            if len(positions) > 4 and positions[4] is not None:
                self.stage_beta = float(positions[4])
            self.apply_targets()

    def current_conditions(self):
        position = list(self.microscope.call('get_stage'))
        self.stage_x, self.stage_y, self.stage_z = [float(v) * 1e9 for v in position[:3]]
        self.stage_alpha = float(position[3])
        self.stage_beta = float(position[4]) if len(position) > 4 and position[4] is not None else 0.0
        shift = list(self.microscope.call('get_image_shift'))
        self.beam_x, self.beam_y = float(shift[0]) * 1e9, float(shift[1]) * 1e9
        self.defocus = float(self.microscope.call('get_defocus')) * 1e9
        tilt = list(self.microscope.call('get_beam_tilt'))
        self.beam_alpha, self.beam_beta = float(tilt[0]) * 1e3, float(tilt[1]) * 1e3

    def set_stage(self):
        self.microscope.call('move_stage', [self.stage_x * 1e-9, self.stage_y * 1e-9, self.stage_z * 1e-9, self.stage_alpha, self.stage_beta])

    def set_defocus(self):
        self.microscope.call('set_defocus', self.defocus * 1e-9)

    def set_beam_shift(self):
        self.microscope.call('set_image_shift', [self.beam_x * 1e-9, self.beam_y * 1e-9])

    def set_beam_tilt(self):
        self.beam_alpha = min(self.beam_alpha, 1000.0)
        self.beam_beta = min(self.beam_beta, 1000.0)
        self.microscope.call('set_beam_tilt', [self.beam_alpha * 1e-3, self.beam_beta * 1e-3])

    # --- UI Helpers ---
    def create_grid_field(self, parent, text, var, unit, r, c):
        tk.Label(parent, text=text, bg="#161616", fg="white", font=("Arial", 9)).grid(row=r, column=c, padx=2, pady=4, sticky="e")
        ttk.Entry(parent, textvariable=var, width=7, font=("Consolas", 10)).grid(row=r, column=c+1, padx=2, pady=4)
        tk.Label(parent, text=unit, bg="#161616", fg="#777777", font=("Arial", 8)).grid(row=r, column=c+2, padx=2, pady=4, sticky="w")

    def add_step_selector(self, parent, text, var, values, unit, col):
        tk.Label(parent, text=text, bg="#222222", fg="#ffffff").grid(row=0, column=col, padx=4, sticky="e")
        ttk.Combobox(parent, textvariable=var, values=values, width=6, state="readonly").grid(row=0, column=col+1, padx=2)
        tk.Label(parent, text=unit + "  |", bg="#222222", fg="#888888").grid(row=0, column=col+2, padx=2, sticky="w")

    def build_dpad(self, parent, up_txt, dn_txt, lt_txt, rt_txt, command_cb, reset_cb):
        tk.Button(parent, text=f"🔼 {up_txt}", font=("Arial", 9), width=5, command=lambda: command_cb(0, 1)).grid(row=0, column=1, pady=3, padx=3)
        tk.Button(parent, text=f"◀️ {lt_txt}", font=("Arial", 9), width=5, command=lambda: command_cb(-1, 0)).grid(row=1, column=0, pady=3, padx=3)
        tk.Button(parent, text="🏠 H", font=("Arial", 9), width=5, bg="#444444", fg="white", command=reset_cb).grid(row=1, column=1, pady=3, padx=3)
        tk.Button(parent, text=f"▶️ {rt_txt}", font=("Arial", 9), width=5, command=lambda: command_cb(1, 0)).grid(row=1, column=2, pady=3, padx=3)
        tk.Button(parent, text=f"🔽 {dn_txt}", font=("Arial", 9), width=5, command=lambda: command_cb(0, -1)).grid(row=2, column=1, pady=3, padx=3)

    # --- Processing Logic Engine ---
    def apply_targets(self):
        """Push every target to the instrument, then read back and refresh the readouts."""
        try:
            self.set_stage()
            self.set_defocus()
            self.set_beam_shift()
            self.set_beam_tilt()
            self.current_conditions()
        except Exception as exc:
            messagebox.showerror("MCP", f"Instrument call failed:\n{exc}")
        self.refresh_entries()

    def refresh_entries(self):
        self.x_var.set(f"{self.stage_x:.2f}")
        self.y_var.set(f"{self.stage_y:.2f}")
        self.z_var.set(f"{self.stage_z:.2f}")
        self.alpha_var.set(f"{self.stage_alpha:.2f}")
        self.beta_var.set(f"{self.stage_beta:.2f}")
        self.bsx_var.set(f"{self.beam_x:.2f}")
        self.bsy_var.set(f"{self.beam_y:.2f}")
        self.defocus_var.set(f"{self.defocus:.2f}")
        self.btx_var.set(f"{self.beam_alpha:.2f}")
        self.bty_var.set(f"{self.beam_beta:.2f}")

    def fetch_vars_from_ui(self):
        self.stage_x, self.stage_y, self.stage_z = float(self.x_var.get()), float(self.y_var.get()), float(self.z_var.get())
        self.stage_alpha, self.stage_beta = float(self.alpha_var.get()), float(self.beta_var.get())
        self.beam_x, self.beam_y, self.defocus = float(self.bsx_var.get()), float(self.bsy_var.get()), float(self.defocus_var.get())
        self.beam_alpha, self.beam_beta = float(self.btx_var.get()), float(self.bty_var.get())

    def jog_engine(self, mx=0, my=0, mz=0, ma=0, mb=0, mbsx=0, mbsy=0, mdf=0, mbtx=0, mbty=0):
        try:
            step_stg_lin = float(self.stg_lin_var.get())
            step_stg_ang = float(self.stg_ang_var.get())
            step_bm_sft = float(self.bm_sft_var.get())
            step_bm_tlt = float(self.bm_tlt_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid inputs found. Check entry numeric parameters.")
            return
        self.stage_x += mx * step_stg_lin
        self.stage_y += my * step_stg_lin
        self.stage_z += mz * step_stg_lin
        self.stage_alpha += ma * step_stg_ang
        self.stage_beta += mb * step_stg_ang
        self.beam_x += mbsx * step_bm_sft
        self.beam_y += mbsy * step_bm_sft
        self.defocus += mdf * step_bm_sft
        self.beam_alpha += mbtx * step_bm_tlt
        self.beam_beta += mbty * step_bm_tlt
        self.apply_targets()

    def go_to_absolute(self):
        try:
            self.fetch_vars_from_ui()
        except ValueError:
            messagebox.showerror("Error", "Absolute move failed. Check syntax configuration numbers.")
            return
        self.apply_targets()

    # --- Granular Zero-Reset Implementations ---
    def reset_stage_xy(self):
        self.stage_x, self.stage_y = 0.0, 0.0
        self.apply_targets()

    def reset_stage_tilt(self):
        self.stage_alpha, self.stage_beta = 0.0, 0.0
        self.apply_targets()

    def reset_stage_z(self):
        self.stage_z = 0.0
        self.apply_targets()

    def reset_beam_shift(self):
        self.beam_x, self.beam_y = 0.0, 0.0
        self.apply_targets()

    def reset_beam_tilt(self):
        self.beam_alpha, self.beam_beta = 0.0, 0.0
        self.apply_targets()

    def reset_defocus(self):
        self.defocus = 0.0
        self.apply_targets()

    def reset_all(self):
        self.stage_x, self.stage_y, self.stage_z, self.stage_alpha, self.stage_beta = 0.0, 0.0, 0.0, 0.0, 0.0
        self.beam_x, self.beam_y, self.defocus, self.beam_alpha, self.beam_beta = 0.0, 0.0, 0.0, 0.0, 0.0
        self.apply_targets()


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedMicroscopeGUI(root)
    root.mainloop()

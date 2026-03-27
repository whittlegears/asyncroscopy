"""
Digital twin version of ThermoMicroscope for HAADF-EDX.

Useful for testing and development without requiring AutoScript hardware.
"""


import json
import time
import math
from typing import Optional

import numpy as np
import pyTEMlib.probe_tools as pt
import pyTEMlib.image_tools as it
from ase.io import read
from ase import Atoms
from ase.build import bulk

import numpy as np
import tango
from tango import AttrWriteType, DevEncoded, DevState
from tango.server import Device, attribute, command, device_property

from asyncroscopy.Microscope import Microscope

class ThermoDigitalTwin(Microscope):
    """
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    # not finishded
    manufacturer = attribute(
        label="ThermoDigitalTwin",
        dtype=str,
        doc="Simulation backend",
    )

    beam_pos = attribute(
        label="Beam Position",
        dtype=(float,),        # 1D array of floats
        max_dim_x=2,           # exactly 2 elements: [x, y]
        access=AttrWriteType.READ_WRITE,
        unit="fractional",
        min_value=0.0,
        max_value=1.0,
        doc="Beam position as [x, y] fractional coordinates, each in range [0.0, 1.0]",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)
        
        # Internal state
        self._stem_mode = True
        self._detector_proxies = {}
        self._manufacturer = "UTKTeam"
        self._beam_pos_x = 0.5
        self._beam_pos_y = 0.5
        self._particle_records = []
        self._imsize = 512
        self._fov = 200e-10  # meters, i.e. 200 angstroms
        self._stage_position = np.random.rand(3) * 1e-6  # random initial stage position in meters
        
        self._connect()
        
    def _connect(self):
        """Simulate connection by connecting to detector proxies."""
        self._connect_detector_proxies()
        self.set_state(DevState.ON)


    def _connect_detector_proxies(self) -> None:
        """Build DeviceProxy objects for each configured detector device."""
        # Extend this dict as more detectors are added
        # later, we want to do this automatically, not with a dictionary.
        addresses: dict[str, str] = {
            "eds":  self.eds_device_address,
            "stage": self.stage_device_address,
            "scan": self.scan_device_address,
        }
        print(addresses)
        for name, address in addresses.items():
            if not address:   # <-- minimal fix
                self.info_stream(f"Skipping {name}: no address configured")
                continue
            try:
                self._detector_proxies[name] = tango.DeviceProxy(address)
                self.info_stream(f"Connected to detector proxy: {name} @ {address}")
            except tango.DevFailed as e:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {e}")


    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_manufacturer(self) -> bool:
        # TODO: query self._microscope.optics.mode when AutoScript available
        return self._manufacturer


    def read_beam_pos(self):
        """Return beam position as [x, y] fractional coordinates."""
        return [self._beam_pos_x, self._beam_pos_y]

    # --- Write Method ---

    def write_beam_pos(self, value):
        """Set beam position from [x, y] fractional coordinates."""
        x, y = value[0], value[1]

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"beam_pos values must be in [0.0, 1.0], got x={x}, y={y}"
            )

        self._beam_pos_x = x
        self._beam_pos_y = y


    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------

    def _acquire_stem_image(self, imsize: int, dwell_time: float, detector_list: list) -> np.ndarray:
        """
        Simulate a stem image of nanopartcles
        For now, these params are hard-coded here.
        Eventually, we will have a sample module (for metadata, but most useful for DigitalTwins)
        """
        size = imsize
        self._imsize = imsize
        fov = self._fov * 1e10  # angstroms
        edge_crop = 20
        beam_current = 1000 # pA?  unsure
        blur_noise_level = float(0.1)
        pixel_size  = fov / size

        # ── Nanoparticle parameters
        particle_radius   = 16.0      # Angstroms, mean radius
        radius_std        = 2.0      # randomize size a bit
        aspect_ratio      = 0.4      # z_radius = aspect_ratio * xy_radius (flat pancake)
        min_separation    = 3.0      # minimum gap between particle surfaces (Angstroms)
        n_particles       = 40       # how many particles to try to place
        max_attempts      = 500      # attempts to place each particle without overlap
        bulk_types = {
            'Au':  bulk('Au', 'fcc', a=4.08),
            'Pt':  bulk('Pt', 'fcc', a=3.92),
            'Fe':  bulk('Fe', 'bcc', a=2.87),
        }
        bulk_names = list(bulk_types.keys())
        desired_angles = [(0, 0, 0), (60, 0, 0), (45, 45, 45)]

        # get probe
        ab = pt.get_target_aberrations("Spectra300", 200000)
        ab['acceleration_voltage'] = 200e3 # eV
        ab['FOV'] = fov /10 # nm
        ab['convergence_angle'] = 30 # mrad
        ab['wavelength'] = it.get_wavelength(ab['acceleration_voltage'])

        def sub_pix_gaussian(size=10, sigma=0.2, dx=0.0, dy=0.0):
            # returns sub-pix shifted gaussian
            coords = np.arange(size) - (size - 1) / 2.0
            x, y = np.meshgrid(coords, coords)
            g = np.exp(-(((x + dx) ** 2 + (y + dy) ** 2) / (2 * sigma**2)))
            g /= g.max()
            return g

        def create_pseudo_potential(xtal, pixel_size, sigma, bounds, atom_frame=11):
            # Create empty image
            x_min, x_max = bounds[0], bounds[1]
            y_min, y_max = bounds[2], bounds[3]
            pixels_x = int((x_max - x_min) / pixel_size)
            pixels_y = int((y_max - y_min) / pixel_size)
            potential_map = np.zeros((pixels_x, pixels_y))
            padding = atom_frame  # to avoid edge effects
            potential_map = np.pad(potential_map, padding, mode='constant', constant_values=0.0)

            # Map of atomic numbers - i.e. scattering intensity
            atomic_numbers = xtal.get_atomic_numbers()
            positions = xtal.get_positions()[:, :2]

            mask = ((positions[:, 0] >= x_min) & (positions[:, 0] < x_max) & (positions[:, 1] >= y_min) & (positions[:, 1] < y_max))
            positions = positions[mask]
            atomic_numbers = atomic_numbers[mask]

            for pos, atomic_number in zip(positions, atomic_numbers):
                x,y = np.round(pos/pixel_size)
                dx,dy = pos - np.round(pos)
        
                single_atom = sub_pix_gaussian(size=atom_frame, sigma=sigma, dx=dx, dy=dy) * atomic_number
                potential_map[int(x+padding+dx-padding//2-1):int(x+padding+dx+padding//2),int(y+padding+dy-padding//2-1):int(y+padding+dy+padding//2)] += single_atom
            potential_map = potential_map[padding:-padding, padding:-padding]
            normalized_map = potential_map / np.max(potential_map)

            return normalized_map

        def poisson_noise(image, counts = 1e9):
            # Normalize the image
            image = image - image.min()
            image = image / image.sum()
            noisy_image = np.random.poisson(image * counts)
            noisy_image = noisy_image - noisy_image.min()
            noisy_image = noisy_image / noisy_image.max()

            return noisy_image

        def lowfreq_noise(image, noise_level=0.1, freq_scale=0.1):
            size_x, size_y = image.shape

            noise = np.random.normal(0, noise_level, (size_x, size_y))
            noise_fft = np.fft.fft2(noise)

            # Create a frequency filter that emphasizes low frequencies
            x_freqs = np.fft.fftfreq(size_x)
            y_freqs = np.fft.fftfreq(size_y)
            freq_filter = np.outer(np.exp(-np.square(x_freqs) / (2 * freq_scale**2)),
                                np.exp(-np.square(y_freqs) / (2 * freq_scale**2)))

            # Apply the frequency filter to the noise in the frequency domain
            filtered_noise_fft = noise_fft * freq_filter
            noisy_image = np.fft.ifft2(filtered_noise_fft).real
            noisy_image = noisy_image - noisy_image.min()
            noisy_image = noisy_image / noisy_image.max()
            return noisy_image

        def rotation_matrix(alpha, beta, gamma):
            a, b, g = np.radians([alpha, beta, gamma])
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a),  np.cos(a), 0], [0, 0, 1]])
            Ry = np.array([[np.cos(b), 0,  np.sin(b)], [0, 1, 0],                  [-np.sin(b), 0, np.cos(b)]])
            Rx = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)],                  [0, np.sin(g), np.cos(g)]])
            return Rz @ Ry @ Rx

        # ── Place particle centers with exclusion zone ──────────────────────────────
        placed_centers  = []
        placed_particles = []
        particle_records = []   # <-- NEW: one entry per placed particle

        for _ in range(max_attempts * n_particles):
            if len(placed_particles) >= n_particles:
                break

            radius = np.random.normal(particle_radius, radius_std)
            radius = np.clip(radius, 3.0, None)

            margin = radius + 2.0

            sample_fov = (fov*1.5, fov*1.5, fov*0.5)  # angstroms
            cx = np.random.uniform(margin, sample_fov[0] - margin)
            cy = np.random.uniform(margin, sample_fov[1] - margin)
            cz = sample_fov[2] * 0.5

            too_close = False
            for (px, py, pr) in placed_centers:
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < (radius + pr + min_separation):
                    too_close = True
                    break
            if too_close:
                continue

            placed_centers.append((cx, cy, radius))
            btype  = np.random.choice(bulk_names)
            i      = len(placed_particles)
            angles = desired_angles[i] if i < len(desired_angles) else tuple(np.random.rand(3) * 360)
            placed_particles.append((btype, np.array([cx, cy, cz]), radius, angles))

            # ── record composition from the bulk unit cell ──────────────────────
            symbols_in_bulk = bulk_types[btype].get_chemical_symbols()
            counts_dict = {}
            for s in symbols_in_bulk:
                counts_dict[s] = counts_dict.get(s, 0) + 1
            total = sum(counts_dict.values())
            composition = {s: c / total for s, c in counts_dict.items()}  # fractions

            particle_records.append({
                'center':      np.array([cx / pixel_size - edge_crop, cy / pixel_size - edge_crop]),  # pixels, image coords
                'radius':      radius / pixel_size,    # pixels
                'btype':       btype,
                'composition': composition,
            })

        # add particle records to self for later retrieval in spectrum acquisition
        self._particle_records = particle_records
        print(f"Placed {len(placed_particles)} particles")



        # ── Carve each nanoparticle from its bulk ───────────────────────────────────
        all_positions = []
        all_symbols   = []

        for (btype, center, radius, angles) in placed_particles:
            this_bulk = bulk_types[btype]
            a_lat     = this_bulk.cell.lengths()[0]
            z_radius  = radius * aspect_ratio

            # supercell just big enough to carve from
            rep       = int(radius * 2 / a_lat) + 3
            supercell = this_bulk.repeat((rep, rep, rep))

            R         = rotation_matrix(*angles)
            positions = supercell.get_positions().copy()
            positions -= positions.mean(axis=0)   # center at origin before rotation
            positions  = positions @ R.T

            # ellipsoidal mask (flat in z)
            r_scaled = np.sqrt(
                (positions[:, 0] / radius)**2 +
                (positions[:, 1] / radius)**2 +
                (positions[:, 2] / z_radius)**2
            )
            mask = r_scaled <= 1.0

            positions  = positions[mask] + center
            symbols    = [s for s, m in zip(supercell.get_chemical_symbols(), mask) if m]

            all_positions.append(positions)
            all_symbols.extend(symbols)

        all_positions = np.vstack(all_positions)
        xtal = Atoms(symbols=all_symbols, positions=all_positions)

        # ── Rest is unchanged ────────────────────────────────────────────────────────
        edge        = 2 * edge_crop * pixel_size
        frame       = (0, fov+edge, 0, fov+edge)
        potential   = create_pseudo_potential(xtal, pixel_size, sigma=1, bounds=frame, atom_frame=11)
        probe, A_k, chi = pt.get_probe(ab, size+2*edge_crop, size+2*edge_crop, verbose=True)

        psf_shifted = np.fft.ifftshift(probe)
        image = np.fft.ifft2(np.fft.fft2(potential) * np.fft.fft2(psf_shifted))
        image = np.absolute(image)
        image = image[edge_crop:-edge_crop, edge_crop:-edge_crop]

        scan_time  = dwell_time * size * size
        counts     = scan_time * (beam_current * 1e-12) / (1.602e-19)
        noisy_image = poisson_noise(image, counts=counts)
        blur_noise  = lowfreq_noise(noisy_image, noise_level=0.1, freq_scale=.1)
        noisy_image += blur_noise * blur_noise_level
        sim_im = np.array(noisy_image, dtype=np.float32)

        return sim_im


    def _acquire_spectrum(self, detector_name: str, exposure_time: float):
        px, py = self.read_beam_pos()   # fractional [0, 1]
        px_pix = px * self._imsize
        py_pix = py * self._imsize

        for rec in self._particle_records:
            cx, cy = rec['center']   # pixels
            r      = rec['radius']   # pixels
            if (px_pix - cx)**2 + (py_pix - cy)**2 <= r**2:
                raw   = {el: frac for el, frac in rec['composition'].items()}
                total = sum(raw.values())
                return {el: v / total + np.random.normal(0.01, 0.1) for el, v in raw.items()}

        all_elements = {el for rec in self._particle_records for el in rec['composition']}
        return {el: np.abs(np.random.normal(0, 0.05)) for el in all_elements}


    def _place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        x, y = position
        self.write_beam_pos([x, y])


    def _set_fov(self, fov) -> None:
        """set field of view in meters"""
        # For the digital twin, we can just store this as a property and use it in acquisition simulations.
        self._fov = fov


    def _get_stage(self):
        """Return current stage position as (x, y, z, a, b) in meters."""
        return self._stage_position
    
    def _move_stage(self, position):
        """Move stage to specified position (x, y, z, a, b) in meters."""
        self.old_pos = self._stage_position
        relative_move = np.array(position) - self._stage_position

        # shift the particle records/ atoms object positions by this much, negative

        random_shift = np.random.normal(0, 5e-8, size=5) 
        self._stage_position = position + random_shift

# ----------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ThermoDigitalTwin.run_server()

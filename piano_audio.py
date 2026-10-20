import ctypes
import math
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Voice:
    key_index: int
    frequencies: np.ndarray
    amplitudes: np.ndarray
    decay_rates: np.ndarray
    phases: np.ndarray
    hammer: np.ndarray
    velocity: float
    pan: float
    age: float = 0.0
    release_at: Optional[float] = None


class _AudioComponentDescription(ctypes.Structure):
    _fields_ = [
        ("componentType", ctypes.c_uint32),
        ("componentSubType", ctypes.c_uint32),
        ("componentManufacturer", ctypes.c_uint32),
        ("componentFlags", ctypes.c_uint32),
        ("componentFlagsMask", ctypes.c_uint32),
    ]


class PianoAudio:
    """Low-latency sampled piano on macOS with a portable synth fallback."""

    _MELODIC_MIDI_CHANNELS = tuple(channel for channel in range(16) if channel != 9)

    def __init__(
        self,
        sample_rate: int = 44100,
        blocksize: int = 256,
        gain: float = 0.16,
        max_voices: int = 96,
        backend: str = "auto",
        reverb: float = 0.3,
        start: bool = True,
    ):
        if backend not in ("auto", "dls", "synth"):
            raise ValueError("backend must be 'auto', 'dls', or 'synth'")

        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.gain = gain
        self.max_voices = max_voices
        self.reverb = float(np.clip(reverb, 0.0, 0.5))
        self.requested_backend = backend
        self.backend = None
        self.available = False
        self.error = None

        self._lock = threading.Lock()
        self._voices = []
        self._stream = None
        self._rng = np.random.default_rng()

        self._audiotoolbox = None
        self._audio_graph = None
        self._music_device = None

        self._reverb_left = np.zeros(max(1, int(sample_rate * 0.067)), dtype=np.float32)
        self._reverb_right = np.zeros(max(1, int(sample_rate * 0.083)), dtype=np.float32)
        self._reverb_left_index = 0
        self._reverb_right_index = 0
        self._reverb_lowpass_left = 0.0
        self._reverb_lowpass_right = 0.0

        if start:
            self.start()

    @staticmethod
    def key_to_midi(key_index: int) -> int:
        return 21 + key_index

    @staticmethod
    def midi_to_frequency(midi_note: int) -> float:
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    @classmethod
    def key_to_frequency(cls, key_index: int) -> float:
        return cls.midi_to_frequency(cls.key_to_midi(key_index))

    def start(self):
        if self.available:
            return

        errors = []
        if self.requested_backend in ("auto", "dls") and sys.platform == "darwin":
            try:
                self._start_macos_dls()
                self.error = None
                self.available = True
                self.backend = "macos-dls"
                return
            except Exception as exc:
                errors.append("macOS DLS: {}".format(exc))
                self._close_macos_dls()

        if self.requested_backend == "dls":
            self.error = RuntimeError("; ".join(errors) or "macOS DLS is unavailable")
            return

        try:
            self._start_synth()
            self.error = None
            self.available = True
            self.backend = "synth"
        except Exception as exc:
            errors.append("synth: {}".format(exc))
            self.error = RuntimeError("; ".join(errors))
            self.available = False
            self.backend = None
            self._close_stream()

    def close(self):
        if not self.available and self._stream is None and self._audio_graph is None:
            return
        self.all_notes_off(immediate=True)
        self.available = False
        self._close_stream()
        self._close_macos_dls()
        self.backend = None

    def play_key(self, key_index: int, velocity: float = 1.0):
        if not self.available or not 0 <= key_index < 88:
            return

        velocity = float(np.clip(velocity, 0.05, 1.0))
        if self.backend == "macos-dls":
            midi_note = self.key_to_midi(key_index)
            midi_velocity = int(round(18 + 109 * velocity**0.72))
            self._midi_event(0x90 | self._midi_channel(key_index), midi_note, midi_velocity)
            return

        voice = self._make_voice(key_index, velocity)
        with self._lock:
            self._release_key_locked(key_index)
            self._voices.append(voice)
            if len(self._voices) > self.max_voices:
                self._voices = self._voices[-self.max_voices :]

    def release_key(self, key_index: int):
        if not self.available or not 0 <= key_index < 88:
            return
        if self.backend == "macos-dls":
            self._midi_event(
                0x80 | self._midi_channel(key_index),
                self.key_to_midi(key_index),
                0,
            )
            return
        with self._lock:
            self._release_key_locked(key_index)

    def all_notes_off(self, immediate: bool = False):
        if self.backend == "macos-dls" and self._music_device is not None:
            controller = 120 if immediate else 123
            for channel in range(16):
                self._midi_event(0xB0 | channel, controller, 0)
            return

        with self._lock:
            if immediate:
                self._voices.clear()
            else:
                for voice in self._voices:
                    if voice.release_at is None:
                        voice.release_at = voice.age

    def _start_synth(self):
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=2,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def _close_stream(self):
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def _start_macos_dls(self):
        toolbox = ctypes.CDLL(
            "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"
        )
        graph_type = ctypes.c_void_p
        node_type = ctypes.c_int32
        status_type = ctypes.c_int32

        toolbox.NewAUGraph.argtypes = [ctypes.POINTER(graph_type)]
        toolbox.NewAUGraph.restype = status_type
        toolbox.AUGraphAddNode.argtypes = [
            graph_type,
            ctypes.POINTER(_AudioComponentDescription),
            ctypes.POINTER(node_type),
        ]
        toolbox.AUGraphAddNode.restype = status_type
        toolbox.AUGraphOpen.argtypes = [graph_type]
        toolbox.AUGraphOpen.restype = status_type
        toolbox.AUGraphNodeInfo.argtypes = [
            graph_type,
            node_type,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        toolbox.AUGraphNodeInfo.restype = status_type
        toolbox.AUGraphConnectNodeInput.argtypes = [
            graph_type,
            node_type,
            ctypes.c_uint32,
            node_type,
            ctypes.c_uint32,
        ]
        toolbox.AUGraphConnectNodeInput.restype = status_type
        toolbox.AUGraphInitialize.argtypes = [graph_type]
        toolbox.AUGraphInitialize.restype = status_type
        toolbox.AUGraphStart.argtypes = [graph_type]
        toolbox.AUGraphStart.restype = status_type
        toolbox.AUGraphStop.argtypes = [graph_type]
        toolbox.AUGraphStop.restype = status_type
        toolbox.DisposeAUGraph.argtypes = [graph_type]
        toolbox.DisposeAUGraph.restype = status_type
        toolbox.MusicDeviceMIDIEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        toolbox.MusicDeviceMIDIEvent.restype = status_type

        graph = graph_type()
        self._check_status(toolbox.NewAUGraph(ctypes.byref(graph)), "NewAUGraph")
        self._audiotoolbox = toolbox
        self._audio_graph = graph

        synth_node = node_type()
        output_node = node_type()
        synth_desc = _AudioComponentDescription(
            self._fourcc("aumu"),
            self._fourcc("dls "),
            self._fourcc("appl"),
            0,
            0,
        )
        output_desc = _AudioComponentDescription(
            self._fourcc("auou"),
            self._fourcc("def "),
            self._fourcc("appl"),
            0,
            0,
        )
        self._check_status(
            toolbox.AUGraphAddNode(graph, ctypes.byref(synth_desc), ctypes.byref(synth_node)),
            "AUGraphAddNode(DLSSynth)",
        )
        self._check_status(
            toolbox.AUGraphAddNode(graph, ctypes.byref(output_desc), ctypes.byref(output_node)),
            "AUGraphAddNode(DefaultOutput)",
        )
        self._check_status(toolbox.AUGraphOpen(graph), "AUGraphOpen")

        music_device = ctypes.c_void_p()
        self._check_status(
            toolbox.AUGraphNodeInfo(
                graph, synth_node, None, ctypes.byref(music_device)
            ),
            "AUGraphNodeInfo",
        )
        self._music_device = music_device
        self._check_status(
            toolbox.AUGraphConnectNodeInput(
                graph, synth_node, 0, output_node, 0
            ),
            "AUGraphConnectNodeInput",
        )
        self._check_status(toolbox.AUGraphInitialize(graph), "AUGraphInitialize")
        self._check_status(toolbox.AUGraphStart(graph), "AUGraphStart")

        channel_volume = int(np.clip(round(104 * self.gain / 0.16), 0, 127))
        reverb_send = int(round(self.reverb * 240))
        for pan_index, channel in enumerate(self._MELODIC_MIDI_CHANNELS):
            self._midi_event(0xC0 | channel, 0, 0)
            self._midi_event(0xB0 | channel, 7, channel_volume)
            self._midi_event(
                0xB0 | channel,
                10,
                int(
                    round(
                        20
                        + 87
                        * pan_index
                        / (len(self._MELODIC_MIDI_CHANNELS) - 1)
                    )
                ),
            )
            self._midi_event(0xB0 | channel, 91, reverb_send)

    def _close_macos_dls(self):
        toolbox = self._audiotoolbox
        graph = self._audio_graph
        self._music_device = None
        self._audio_graph = None
        self._audiotoolbox = None
        if toolbox is not None and graph is not None:
            toolbox.AUGraphStop(graph)
            toolbox.DisposeAUGraph(graph)

    def _midi_event(self, status: int, data1: int, data2: int):
        if self._audiotoolbox is None or self._music_device is None:
            return
        result = self._audiotoolbox.MusicDeviceMIDIEvent(
            self._music_device,
            status,
            data1,
            data2,
            0,
        )
        self._check_status(result, "MusicDeviceMIDIEvent")

    @staticmethod
    def _midi_channel(key_index: int) -> int:
        channels = PianoAudio._MELODIC_MIDI_CHANNELS
        return channels[min(len(channels) - 1, key_index * len(channels) // 88)]

    @staticmethod
    def _fourcc(value: str) -> int:
        return int.from_bytes(value.encode("ascii"), "big")

    @staticmethod
    def _check_status(status: int, operation: str):
        if status:
            raise OSError("{} failed with OSStatus {}".format(operation, status))

    def _make_voice(self, key_index: int, velocity: float) -> _Voice:
        midi_note = self.key_to_midi(key_index)
        base_frequency = self.key_to_frequency(key_index)
        harmonic_numbers = np.arange(1, 13, dtype=np.float32)

        distance_from_middle = abs(midi_note - 60) / 48.0
        inharmonicity = 0.00006 + 0.00038 * distance_from_middle**2
        partial_frequencies = (
            base_frequency
            * harmonic_numbers
            * np.sqrt(1.0 + inharmonicity * harmonic_numbers**2)
        )
        valid = partial_frequencies < self.sample_rate * 0.46
        partial_frequencies = partial_frequencies[valid]
        harmonic_numbers = harmonic_numbers[valid]

        if midi_note < 33:
            detune_cents = np.array([0.0], dtype=np.float32)
        elif midi_note < 42:
            detune_cents = np.array([-0.35, 0.35], dtype=np.float32)
        else:
            detune_cents = np.array([-0.55, 0.0, 0.62], dtype=np.float32)

        frequencies = (
            partial_frequencies[:, None]
            * (2.0 ** (detune_cents[None, :] / 1200.0))
        ).reshape(-1)
        brightness = 1.9 - 0.72 * velocity
        partial_amplitudes = 1.0 / harmonic_numbers**brightness
        string_amplitudes = np.ones(len(detune_cents), dtype=np.float32)
        amplitudes = (partial_amplitudes[:, None] * string_amplitudes).reshape(-1)
        amplitudes /= max(float(amplitudes.sum()), 1e-6)

        pitch_position = (midi_note - 21) / 87.0
        base_decay = 0.16 + 1.05 * pitch_position**1.55
        partial_decay = base_decay * (1.0 + 0.16 * (harmonic_numbers - 1))
        decay_rates = np.repeat(partial_decay, len(detune_cents)).astype(np.float32)
        phases = self._rng.uniform(0.0, 2.0 * math.pi, len(frequencies)).astype(
            np.float32
        )

        hammer_length = max(1, int(self.sample_rate * 0.028))
        hammer_t = np.arange(hammer_length, dtype=np.float32) / self.sample_rate
        hammer = self._rng.standard_normal(hammer_length).astype(np.float32)
        hammer = np.convolve(hammer, np.array([0.2, 0.6, 0.2]), mode="same")
        hammer *= np.exp(-hammer_t * (105.0 - 35.0 * velocity))
        hammer *= 0.12 + 0.22 * velocity

        return _Voice(
            key_index=key_index,
            frequencies=frequencies.astype(np.float32),
            amplitudes=amplitudes.astype(np.float32),
            decay_rates=decay_rates,
            phases=phases,
            hammer=hammer.astype(np.float32),
            velocity=velocity,
            pan=0.12 + 0.76 * key_index / 87.0,
        )

    def _release_key_locked(self, key_index: int):
        for voice in self._voices:
            if voice.key_index == key_index and voice.release_at is None:
                voice.release_at = voice.age

    def _callback(self, outdata, frames, _time, _status):
        t = np.arange(frames, dtype=np.float32) / self.sample_rate
        left = np.zeros(frames, dtype=np.float32)
        right = np.zeros(frames, dtype=np.float32)

        with self._lock:
            active = []
            for voice in self._voices:
                age = voice.age + t
                attack_time = 0.0015 + 0.0025 * (1.0 - voice.velocity)
                attack = np.minimum(age / attack_time, 1.0)

                phase = (
                    voice.phases[:, None]
                    + (2.0 * math.pi * voice.frequencies[:, None]) * t
                )
                partial_envelope = np.exp(-voice.decay_rates[:, None] * age)
                tone = np.sum(
                    voice.amplitudes[:, None] * partial_envelope * np.sin(phase),
                    axis=0,
                )

                hammer_start = int(round(voice.age * self.sample_rate))
                hammer_end = min(hammer_start + frames, len(voice.hammer))
                if hammer_end > hammer_start:
                    count = hammer_end - hammer_start
                    tone[:count] += voice.hammer[hammer_start:hammer_end]

                if voice.release_at is not None:
                    release_age = np.maximum(age - voice.release_at, 0.0)
                    damper_rate = 11.0 + 8.0 * voice.key_index / 87.0
                    tone *= np.exp(-damper_rate * release_age)

                tone *= attack * voice.velocity**0.78
                left_gain = math.cos(voice.pan * math.pi * 0.5)
                right_gain = math.sin(voice.pan * math.pi * 0.5)
                left += (left_gain * tone).astype(np.float32)
                right += (right_gain * tone).astype(np.float32)

                voice.phases = np.mod(
                    voice.phases
                    + 2.0 * math.pi * voice.frequencies * frames / self.sample_rate,
                    2.0 * math.pi,
                ).astype(np.float32)
                voice.age += frames / self.sample_rate

                released_for = (
                    0.0
                    if voice.release_at is None
                    else voice.age - voice.release_at
                )
                if voice.age < 14.0 and released_for < 0.75:
                    active.append(voice)
            self._voices = active

        left, right = self._apply_reverb(left, right)
        output = np.column_stack((left, right))
        outdata[:] = np.tanh(self.gain * output).astype(np.float32)

    def _apply_reverb(self, left: np.ndarray, right: np.ndarray):
        if self.reverb <= 0:
            return left, right

        wet_left = np.empty_like(left)
        wet_right = np.empty_like(right)
        left_buffer = self._reverb_left
        right_buffer = self._reverb_right
        left_index = self._reverb_left_index
        right_index = self._reverb_right_index
        lowpass_left = self._reverb_lowpass_left
        lowpass_right = self._reverb_lowpass_right

        for i in range(len(left)):
            delayed_left = float(left_buffer[left_index])
            delayed_right = float(right_buffer[right_index])
            lowpass_left = 0.72 * lowpass_left + 0.28 * delayed_left
            lowpass_right = 0.72 * lowpass_right + 0.28 * delayed_right
            wet_left[i] = lowpass_left
            wet_right[i] = lowpass_right
            left_buffer[left_index] = left[i] + 0.31 * lowpass_right
            right_buffer[right_index] = right[i] + 0.31 * lowpass_left
            left_index = (left_index + 1) % len(left_buffer)
            right_index = (right_index + 1) % len(right_buffer)

        self._reverb_left_index = left_index
        self._reverb_right_index = right_index
        self._reverb_lowpass_left = lowpass_left
        self._reverb_lowpass_right = lowpass_right
        return left + self.reverb * wet_left, right + self.reverb * wet_right

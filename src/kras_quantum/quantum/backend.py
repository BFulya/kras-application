"""
Kuantum arka ucu (backend) soyutlamasi.

Makale, uc ayri uretim kolunu karsilastirir (Fig. 1d):

  * Classical      -> prior yok, sadece LSTM
  * Quantum (SIM)  -> QCBM bir kuantum simulatorunde
  * Quantum (HW)   -> QCBM gercek kuantum donaniminda

Makale, QCBM'i IBM'in "Guadalupe" sisteminde calistirmistir
(16 qubit, Falcon r4P islemci tipi; bkz. "Benchmark setup").
Guadalupe 2023'te emekliye ayrildi; asagidaki `IBMBackend` gunumuzun
IBM Quantum cihazlarini (Eagle/Heron) ayni arayuzle kullanir.

Bu modul iki arka uc sunar:
  AerBackend  -> yerel simulasyon (varsayilan, kimlik bilgisi gerektirmez)
  IBMBackend  -> gercek donanim (qiskit-ibm-runtime ve IBM hesabi gerektirir)

Ortak arayuz
------------
  .n_qubits
  .name
  .is_hardware
  .probabilities(circuit, params) -> np.ndarray, uzunluk 2**n  (Born kurali)
  .sample(circuit, params, shots) -> dict[int, int]            (olcum sayimlari)

`probabilities`, Born kuralinin  p(x) = |<x|psi(theta)>|^2  tam degerini verir;
simulatorde dogrudan statevector'dan, donanimda ise shot frekanslarindan
tahmin edilir (donanimda tam dagilim erisilebilir degildir).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

__all__ = ["QuantumBackend", "AerBackend", "IBMBackend", "get_backend"]


def _bind(circuit: QuantumCircuit, theta: ParameterVector, params: np.ndarray) -> QuantumCircuit:
    """Parametre vektorunu devreye baglar."""
    params = np.asarray(params, dtype=float)
    if params.shape != (len(theta),):
        raise ValueError(
            f"Parametre boyutu uyusmuyor: devre {len(theta)} bekliyor, "
            f"verilen {params.shape}"
        )
    return circuit.assign_parameters({theta: params})


class QuantumBackend(ABC):
    """QCBM'in uzerinde calistigi kuantum arka ucu."""

    name: str = "abstract"
    is_hardware: bool = False

    def __init__(self, n_qubits: int, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.seed = seed

    @abstractmethod
    def probabilities(
        self, circuit: QuantumCircuit, theta: ParameterVector, params: np.ndarray
    ) -> np.ndarray:
        """Born kurali olasilik dagilimi, uzunluk 2**n_qubits."""

    @abstractmethod
    def sample(
        self, circuit: QuantumCircuit, theta: ParameterVector, params: np.ndarray, shots: int
    ) -> dict[int, int]:
        """Olcum sonuclari: {bitstring_tamsayi: sayim}."""

    def __repr__(self) -> str:  # pragma: no cover - sadece tani amacli
        kind = "donanim" if self.is_hardware else "simulator"
        return f"<{type(self).__name__} name={self.name!r} qubits={self.n_qubits} tur={kind}>"


class AerBackend(QuantumBackend):
    """Yerel simulasyon (qiskit-aer).

    Makaledeki "Quantum (SIM)" koluna karsilik gelir. Ayrica donanim
    kosularini hazirlarken hizli yineleme icin kullanilir.

    Parameters
    ----------
    noise_model
        Opsiyonel qiskit-aer gurultu modeli. Verilirse `probabilities`
        de shot-bazli tahmine duser (gurultulu devrenin saf statevector'u
        anlamli degildir).
    """

    is_hardware = False

    def __init__(
        self, n_qubits: int, seed: int | None = None, noise_model: object | None = None
    ) -> None:
        super().__init__(n_qubits, seed)
        self.noise_model = noise_model
        self.name = "aer_simulator" + ("+noise" if noise_model is not None else "")
        self._sim = None  # tembel yukleme: AerSimulator ancak gerektiginde olusturulur

    @property
    def simulator(self):
        if self._sim is None:
            from qiskit_aer import AerSimulator

            kwargs: dict = {}
            if self.noise_model is not None:
                kwargs["noise_model"] = self.noise_model
            if self.seed is not None:
                kwargs["seed_simulator"] = self.seed
            self._sim = AerSimulator(**kwargs)
        return self._sim

    def probabilities(
        self, circuit: QuantumCircuit, theta: ParameterVector, params: np.ndarray
    ) -> np.ndarray:
        bound = _bind(circuit, theta, params)

        if self.noise_model is None:
            # Gurultusuz: tam statevector -> kesin Born olasiliklari.
            probs = Statevector(bound).probabilities()
            return np.asarray(probs, dtype=float)

        # Gurultulu: shot frekanslarindan tahmin.
        shots = 8192
        counts = self.sample(circuit, theta, params, shots)
        probs = np.zeros(2**self.n_qubits, dtype=float)
        for idx, c in counts.items():
            probs[idx] = c / shots
        return probs

    def sample(
        self, circuit: QuantumCircuit, theta: ParameterVector, params: np.ndarray, shots: int
    ) -> dict[int, int]:
        bound = _bind(circuit, theta, params)
        bound = bound.copy()
        bound.measure_all()

        compiled = transpile(bound, self.simulator)
        result = self.simulator.run(compiled, shots=shots, memory=False).result()
        raw = result.get_counts()

        # Qiskit bitstring'leri little-endian ve bosluklu gelebilir; tamsayiya ceviriyoruz.
        return {int(bits.replace(" ", ""), 2): count for bits, count in raw.items()}


class IBMBackend(QuantumBackend):
    """Gercek IBM Quantum donanimi (makaledeki "Quantum (HW)" kolu).

    qiskit-ibm-runtime ve kayitli bir IBM Quantum hesabi gerektirir:

        pip install qiskit-ibm-runtime
        python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \\
                   QiskitRuntimeService.save_account(channel='ibm_quantum', token='...')"

    Makale "Guadalupe" (16 qubit, Falcon r4P) kullanmistir; bu cihaz artik
    hizmet disidir. `backend_name=None` birakilirsa, istenen qubit sayisini
    karsilayan en az mesgul cihaz secilir.

    NOT: Donanimda tam dagilim olculemez. `probabilities`, shot
    frekanslarindan tahmin doner; 2**16 = 65536 olasi sonuc icin bu tahmin
    kacinilmaz olarak seyrektir. Bu, makaledeki yaklasimin dogal bir
    kisitidir: donanim kolu, dagilimi degil ORNEKLERI uretir.
    """

    is_hardware = True

    def __init__(
        self,
        n_qubits: int,
        backend_name: str | None = None,
        seed: int | None = None,
        optimization_level: int = 3,
    ) -> None:
        super().__init__(n_qubits, seed)
        self.backend_name = backend_name
        self.optimization_level = optimization_level
        self.name = backend_name or "ibm_least_busy"
        self._backend = None
        self._sampler = None

    @property
    def backend(self):
        if self._backend is None:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Gercek donanim icin qiskit-ibm-runtime gerekli: "
                    "pip install qiskit-ibm-runtime"
                ) from exc

            service = QiskitRuntimeService()
            if self.backend_name:
                self._backend = service.backend(self.backend_name)
            else:
                self._backend = service.least_busy(
                    operational=True, simulator=False, min_num_qubits=self.n_qubits
                )
            self.name = self._backend.name
        return self._backend

    @property
    def sampler(self):
        if self._sampler is None:
            from qiskit_ibm_runtime import SamplerV2

            self._sampler = SamplerV2(mode=self.backend)
        return self._sampler

    def sample(
        self, circuit: QuantumCircuit, theta: ParameterVector, params: np.ndarray, shots: int
    ) -> dict[int, int]:
        bound = _bind(circuit, theta, params).copy()
        bound.measure_all()

        isa_circuit = transpile(
            bound, backend=self.backend, optimization_level=self.optimization_level
        )
        job = self.sampler.run([isa_circuit], shots=shots)
        result = job.result()[0]

        # SamplerV2, olcum kaydini klasik register adiyla dondurur.
        register = next(iter(result.data.keys()))
        raw = getattr(result.data, register).get_counts()
        return {int(bits.replace(" ", ""), 2): count for bits, count in raw.items()}

    def probabilities(
        self, circuit: QuantumCircuit, theta: ParameterVector, params: np.ndarray, shots: int = 4096
    ) -> np.ndarray:
        counts = self.sample(circuit, theta, params, shots)
        total = sum(counts.values()) or 1
        probs = np.zeros(2**self.n_qubits, dtype=float)
        for idx, c in counts.items():
            probs[idx] = c / total
        return probs


def get_backend(kind: str = "sim", n_qubits: int = 16, **kwargs) -> QuantumBackend:
    """Fabrika fonksiyonu.

    kind
      "sim" | "simulator" | "aer" -> AerBackend
      "hw"  | "hardware" | "ibm"  -> IBMBackend
    """
    kind = kind.lower()
    if kind in ("sim", "simulator", "aer"):
        return AerBackend(n_qubits=n_qubits, **kwargs)
    if kind in ("hw", "hardware", "ibm"):
        return IBMBackend(n_qubits=n_qubits, **kwargs)
    raise ValueError(f"Bilinmeyen backend turu: {kind!r} ('sim' veya 'hw' bekleniyor)")

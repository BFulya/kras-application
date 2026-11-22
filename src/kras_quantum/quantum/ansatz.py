"""
QCBM varyasyonel devresi (ansatz).

Makale referansi
----------------
Extended Data Fig. 5 | "Quantum Circuit Born Machine (QCBM) Model":

    "Schematic representation of the Quantum Circuit Born Machine (QCBM)
     implemented in our numerical experiments, illustrating a variational
     quantum circuit with a configuration of three layers and four qubits.
     In practice, our numerical experiments utilized a system with 16 qubits.
     The depicted quantum gates, including parameterized rotations (Rx, Rz)
     and entangling CNOT gates, are orchestrated to evolve the initial state
     |0> into a complex quantum state |psi(theta)>."

Fig. 5'teki kapi dizilimi (soldan saga, 4 qubit icin):

    q0: Rx(t0) - Rz(t4) -+- Rxx(t8) -+------------------- Rz(t11) - Rx(t15)
    q1: Rx(t1) - Rz(t5) -+-----------+- Rxx(t9) -+------- Rz(t12) - Rx(t16)
    q2: Rx(t2) - Rz(t6) -------------+-----------+ Rxx(t10) - Rz(t13) - Rx(t17)
    q3: Rx(t3) - Rz(t7) -------------------------+------- Rz(t14) - Rx(t18)

Yani: [Rx x n] [Rz x n] [Rxx linear x (n-1)] [Rz x n] [Rx x n]
      = 4n + (n-1) parametre.  n=4 icin 16 + 3 = 19 -> semadaki theta_0..theta_18
      ile birebir ortusur.

Makale/sema tutarsizligi
------------------------
Makale metni "16 qubits and four layers and 96 parameters" diyor. Fig. 5
topolojisini 16 qubit / 4 katmana olceklemek 96 degil 220 parametre verir
(4 x (16+16+15) + 32). 96 sayisi ancak parametresiz entangler kullanan bir
varyantla elde edilir: 2 x 16 x 3 = 96.

Bu yuzden iki ansatz modu sunuyoruz; hangisinin kullanildigi her kosuda
raporlanir:

  * "fig5"              -> Extended Data Fig. 5 topolojisi (parametreli Rxx).
                           Sema ile birebir uyumlu. VARSAYILAN.
  * "hardware_efficient" -> parametresiz CNOT merdiveni; n=16, L=3 icin
                           tam 96 parametre (makale metnindeki sayi).

Ayrintili tartisma: README "Makaleden sapmalar ve yorumlar" bolumu.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["AnsatzSpec", "build_ansatz", "count_parameters"]


_VALID_MODES = ("fig5", "hardware_efficient")
_VALID_ENTANGLEMENT = ("linear", "circular")


@dataclass(frozen=True)
class AnsatzSpec:
    """Ansatz'i tanimlayan degismez tanim."""

    n_qubits: int
    n_layers: int
    mode: str = "fig5"
    entanglement: str = "linear"  # Makale: "We used linear topology for our project."

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError(f"n_qubits >= 2 olmali, verilen: {self.n_qubits}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers >= 1 olmali, verilen: {self.n_layers}")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode {_VALID_MODES} icinden biri olmali, verilen: {self.mode!r}")
        if self.entanglement not in _VALID_ENTANGLEMENT:
            raise ValueError(
                f"entanglement {_VALID_ENTANGLEMENT} icinden biri olmali, "
                f"verilen: {self.entanglement!r}"
            )

    @property
    def n_parameters(self) -> int:
        return count_parameters(self)


def _entangling_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    """Dolanikilik topolojisine gore qubit ciftleri.

    linear   : (0,1), (1,2), ..., (n-2, n-1)        -> n-1 cift  [makalenin kullandigi]
    circular : linear + (n-1, 0)                    -> n cift
    """
    pairs = [(q, q + 1) for q in range(n_qubits - 1)]
    if entanglement == "circular":
        pairs.append((n_qubits - 1, 0))
    return pairs


def count_parameters(spec: AnsatzSpec) -> int:
    """Devre kurulmadan parametre sayisini hesaplar."""
    n, L = spec.n_qubits, spec.n_layers
    n_pairs = len(_entangling_pairs(n, spec.entanglement))

    if spec.mode == "fig5":
        # Her katman: Rx(n) + Rz(n) + Rxx(n_pairs); ardindan kapanis Rz(n) + Rx(n).
        return L * (2 * n + n_pairs) + 2 * n
    # hardware_efficient: her katman Rx(n) + Rz(n), entangler parametresiz CNOT.
    return 2 * n * L


def build_ansatz(spec: AnsatzSpec) -> tuple[QuantumCircuit, ParameterVector]:
    """QCBM varyasyonel devresini kurar.

    Devre olcum ICERMEZ: ayni devre hem tam statevector olasiliklari
    (exact NLL) hem de shot-bazli ornekleme icin kullanilir. Olcum,
    ornekleyici tarafindan gerektiginde eklenir.

    Returns
    -------
    (circuit, theta)
        theta[k], Fig. 5'teki theta_k ile ayni sirayi izler.
    """
    n = spec.n_qubits
    theta = ParameterVector("theta", count_parameters(spec))
    qc = QuantumCircuit(n, name=f"QCBM[{spec.mode}]")
    pairs = _entangling_pairs(n, spec.entanglement)
    k = 0  # parametre sayaci

    if spec.mode == "fig5":
        for layer in range(spec.n_layers):
            # --- tek-qubit rotasyon katmani: Rx sonra Rz ---
            for q in range(n):
                qc.rx(theta[k], q)
                k += 1
            for q in range(n):
                qc.rz(theta[k], q)
                k += 1
            # --- parametreli Rxx dolanikilik katmani (lineer topoloji) ---
            for q1, q2 in pairs:
                qc.rxx(theta[k], q1, q2)
                k += 1
            qc.barrier(label=f"L{layer}")

        # --- Fig. 5'teki kapanis rotasyonlari: Rz sonra Rx ---
        for q in range(n):
            qc.rz(theta[k], q)
            k += 1
        for q in range(n):
            qc.rx(theta[k], q)
            k += 1

    else:  # hardware_efficient
        for layer in range(spec.n_layers):
            for q in range(n):
                qc.rx(theta[k], q)
                k += 1
            for q in range(n):
                qc.rz(theta[k], q)
                k += 1
            if layer < spec.n_layers - 1:  # son katmandan sonra dolanikilik gereksiz
                for q1, q2 in pairs:
                    qc.cx(q1, q2)
                qc.barrier(label=f"L{layer}")

    assert k == len(theta), f"Parametre sayisi tutmadi: {k} != {len(theta)}"
    return qc, theta

import numpy as np

class Qubit:
    def __init__(self, alpha=1.0, beta=0.0):
        """Инициализация кубита с комплексными амплитудами alpha и beta.
        |ψ⟩ = α|0⟩ + β|1⟩, где |α|^2 + |β|^2 = 1"""
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        if norm == 0:
            raise ValueError("Амплитуды кубита не могут быть одновременно нулевыми")
        self.state = np.array([alpha, beta], dtype=complex) / norm

    def apply_gate(self, gate_matrix):
        """Применение квантового гейта к кубиту"""
        if gate_matrix.shape != (2, 2):
            raise ValueError("Гейт должен быть матрицей 2x2")
        self.state = np.dot(gate_matrix, self.state)
        # Нормализация для устранения численных ошибок
        norm = np.sqrt(np.abs(self.state[0])**2 + np.abs(self.state[1])**2)
        self.state = self.state / norm
        return self

    def get_state(self):
        """Возвращает текущее состояние кубита"""
        return self.state

    def measure(self):
        """Измерение кубита в вычислительном базисе"""
        probs = np.abs(self.state)**2
        result = np.random.choice([0, 1], p=probs)
        # Коллапс состояния после измерения
        if result == 0:
            self.state = np.array([1.0, 0.0], dtype=complex)
        else:
            self.state = np.array([0.0, 1.0], dtype=complex)
        return result

class QuantumGates:
    @staticmethod
    def pauli_x():
        """Гейт Паули-X (NOT)"""
        return np.array([[0, 1],
                        [1, 0]], dtype=complex)

    @staticmethod
    def pauli_y():
        """Гейт Паули-Y"""
        return np.array([[0, -1j],
                        [1j, 0]], dtype=complex)

    @staticmethod
    def pauli_z():
        """Гейт Паули-Z"""
        return np.array([[1, 0],
                        [0, -1]], dtype=complex)

    @staticmethod
    def hadamard():
        """Гейт Адамара"""
        return np.array([[1, 1],
                        [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def cnot():
        """Двухкубитный гейт CNOT"""
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=complex)

    @staticmethod
    def apply_cnot(control_qubit, target_qubit):
        """Применение CNOT к двум кубитам"""
        if not isinstance(control_qubit, Qubit) or not isinstance(target_qubit, Qubit):
            raise ValueError("Оба аргумента должны быть объектами класса Qubit")

        # Формируем двухкубитное состояние через тензорное произведение
        state = np.kron(control_qubit.get_state(), target_qubit.get_state())

        # Применяем CNOT
        new_state = np.dot(QuantumGates.cnot(), state)

        # Обновляем состояния отдельных кубитов (упрощённо)
        control_qubit.state = np.array([new_state[0] + new_state[2], new_state[1] + new_state[3]], dtype=complex)
        target_qubit.state = np.array([new_state[0] + new_state[1], new_state[2] + new_state[3]], dtype=complex)

        # Нормализация состояний
        norm_control = np.sqrt(np.abs(control_qubit.state[0])**2 + np.abs(control_qubit.state[1])**2)
        norm_target = np.sqrt(np.abs(target_qubit.state[0])**2 + np.abs(target_qubit.state[1])**2)

        if norm_control != 0:
            control_qubit.state = control_qubit.state / norm_control
        if norm_target != 0:
            target_qubit.state = target_qubit.state / norm_target

        return control_qubit, target_qubit

def print_state(qubit, name="Кубит"):
    a0, a1 = qubit.get_state()
    p0 = np.abs(a0)**2
    p1 = np.abs(a1)**2
    print(f"{name}:")
    print(f" Амплитуда |0⟩: {a0:.3f}, амплитуда |1⟩: {a1:.3f}")
    print(f" Вероятность |0⟩: {p0:.3f}, вероятность |1⟩: {p1:.3f}")

def print_explanation(gate_name, qubit):
    print(f"\nПрименили гейт {gate_name}.")
    if gate_name == "Pauli-X":
        print("- Pauli-X — это как квантовый NOT: |0⟩ превращается в |1⟩, а |1⟩ в |0⟩.")
    elif gate_name == "Pauli-Y":
        print("- Pauli-Y меняет состояние и добавляет фазовый сдвиг — как поворот в комплексной плоскости.")
    elif gate_name == "Pauli-Z":
        print("- Pauli-Z меняет знак амплитуды |1⟩, не трогая |0⟩ — переключатель фазы.")
    elif gate_name == "Hadamard":
        print("- Hadamard создаёт суперпозицию — кубит одновременно и в состоянии |0⟩ и |1⟩ с равными шансами.")
    print(f"Теперь вероятности измерения:\n - Получить 0: {np.abs(qubit.state[0])**2:.3f}\n - Получить 1: {np.abs(qubit.state[1])**2:.3f}")


if __name__ == "__main__":
    # Создаем кубит в состоянии |0⟩
    q = Qubit(1, 0)
    print("=== Начальное состояние кубита ===")
    print_state(q)

    # Тест Pauli-X
    print("\n=== Тест Pauli-X ===")
    q.apply_gate(QuantumGates.pauli_x())
    print_state(q)
    print_explanation("Pauli-X", q)

    # Тест Pauli-Y
    print("=== Тест Pauli-Y ===")
    q = Qubit(1, 0)  # сбросим в |0⟩
    q.apply_gate(QuantumGates.pauli_y())
    print_state(q)
    print_explanation("Pauli-Y", q)

    # Тест Pauli-Z
    print("=== Тест Pauli-Z ===")
    q = Qubit(1, 0)
    q.apply_gate(QuantumGates.pauli_z())
    print_state(q)
    print_explanation("Pauli-Z", q)

    # Тест Hadamard
    print("=== Тест Hadamard ===")
    q = Qubit(1, 0)
    q.apply_gate(QuantumGates.hadamard())
    print_state(q)
    print_explanation("Hadamard", q)

    # Тест CNOT
    print("=== Тест CNOT ===")
    q1 = Qubit(0, 1)  # |1⟩
    q2 = Qubit(1, 0)  # |0⟩
    print("Начальное состояние двухкубитной системы:")
    print_state(q1, "Кубит 1")
    print_state(q2, "Кубит 2")

    q1, q2 = QuantumGates.apply_cnot(q1, q2)

    print("После применения CNOT:")
    print_state(q1, "Кубит 1")
    print_state(q2, "Кубит 2")


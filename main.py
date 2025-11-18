import numpy as np
import pandas as pd
import os
import sys

# --- Import algoritmů ---
# Předpokládá se, že tyto soubory existují ve složce algorithms/
from algorithms.differential_evolution import differential_evolution
from algorithms.particle_swarm_optimization import particle_swarm_optimization
from algorithms.soma import soma_all_to_one
from algorithms.firefly_algorithm import firefly_algorithm
from algorithms.tlbo import tlbo

# --- Import testovacích funkcí ---
# Předpokládá se, že tyto soubory existují ve složce functions/
from functions.sphere import Sphere
from functions.ackley import Ackley
from functions.schwefel import Schwefel
from functions.rosenbrock import Rosenbrock
from functions.rastrigin import Rastrigin
from functions.griewank import Griewank
from functions.levy import Levy
from functions.michalewicz import Michalewicz
from functions.zakharov import Zakharov


def run_benchmark():
    # ==========================================
    # NASTAVENÍ EXPERIMENTU (podle Exercise 10)
    # ==========================================
    # Zadání: D=30, NP=30, Max_OFE=3000 [cite: 140, 141, 142]
    DIMENSION = 30
    POP_SIZE = 30
    MAX_OFE = 3000
    NUM_EXPERIMENTS = 30
    OUTPUT_FILE = "results_exercise10.xlsx"

    # Seznam funkcí k testování
    functions_classes = [
        Sphere, Ackley, Schwefel, Rosenbrock,
        Rastrigin, Griewank, Levy, Michalewicz, Zakharov
    ]

    # --- VÝPOČET GENERACÍ PRO DODRŽENÍ LIMITU Max_OFE ---
    # Cílem je dát každému algoritmu maximum prostoru v rámci limitu 3000 evaluací.

    # 1. DE, PSO, FA:
    # - Spotřeba: 1 evaluace na jedince za generaci.
    # - Výpočet: 3000 / 30 = 100 generací.

    # 2. TLBO:
    # - Spotřeba: 2 evaluace na jedince za generaci (Teacher phase + Learner phase).
    # - Výpočet: 3000 / (30 * 2) = 50 generací.

    # 3. SOMA (All-to-One):
    # - Spotřeba: (NP-1) jedinců skáče k lídrovi.
    # - Kroky: path_length=3.0, step=0.11 => cca 27 kroků na jedince.
    # - Evaluace na 1 migraci: 29 jedinců * 27 kroků = 783 evaluací.
    # - Výpočet: 3000 / 783 = 3.83.
    # - Nastavíme 3 migrace, abychom nepřekročili limit (4 migrace by byly cca 3132 evaluací).

    algos = {
        "DE": {"func": differential_evolution, "gens": 100, "args": {"NP": POP_SIZE, "F": 0.5, "CR": 0.9}},
        "PSO": {"func": particle_swarm_optimization, "gens": 100,
                "args": {"pop_size": POP_SIZE, "c1": 2.0, "c2": 2.0, "w": 0.7}},
        "SOMA": {"func": soma_all_to_one, "gens": 3, "args": {"pop_size": POP_SIZE, "path_length": 3.0, "step": 0.11}},
        "FA": {"func": firefly_algorithm, "gens": 100, "args": {"pop_size": POP_SIZE, "alpha": 0.2, "beta_0": 1.0}},
        "TLBO": {"func": tlbo, "gens": 50, "args": {"population_size": POP_SIZE}}
    }

    print(f"=== STARTING BENCHMARK (Exercise 10) ===")
    print(f"Dimension: {DIMENSION}")
    print(f"Population: {POP_SIZE}")
    print(f"Max OFE limit: {MAX_OFE}")
    print(f"Experiments per function: {NUM_EXPERIMENTS}")
    print("-" * 60)

    # Použijeme Pandas ExcelWriter pro zápis do více listů (sheetů)
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:

            for FuncClass in functions_classes:
                func_instance = FuncClass(dimension=DIMENSION)
                print(f"Processing function: {func_instance.name: <15}", end="")

                # Příprava indexů tabulky (Experiment 1..30, Mean, Std)
                rows = [f"Experiment {i + 1}" for i in range(NUM_EXPERIMENTS)]
                rows.append("Mean")
                rows.append("Std. Dev.")

                # Vytvoření prázdného DataFrame pro aktuální funkci
                results_table = pd.DataFrame(index=rows, columns=algos.keys())

                for algo_name, config in algos.items():
                    best_values = []

                    # Spuštění 30 experimentů
                    for i in range(NUM_EXPERIMENTS):
                        # Vytvoření nové instance funkce (pro čistý start)
                        f = FuncClass(dimension=DIMENSION)

                        # Příprava argumentů
                        kwargs = config["args"].copy()

                        # Správné pojmenování parametru pro počet generací/migrací
                        if algo_name == "SOMA":
                            kwargs["M_max"] = config["gens"]
                        elif algo_name == "TLBO":
                            kwargs["max_generations"] = config["gens"]
                        elif algo_name == "FA":
                            kwargs["max_gen"] = config["gens"]
                        elif algo_name == "DE":
                            kwargs["G"] = config["gens"]
                        elif algo_name == "PSO":
                            kwargs["M_max"] = config["gens"]

                        # Spuštění algoritmu
                        # Očekává se návratová hodnota: (best_pos, best_val, history)
                        # history ignorujeme (_) pro úsporu paměti při 30 opakováních
                        _, best_val, _ = config["func"](f, **kwargs)
                        best_values.append(best_val)

                    # Zápis výsledků do tabulky
                    # 1. Hodnoty jednotlivých experimentů
                    results_table.loc[rows[:NUM_EXPERIMENTS], algo_name] = best_values

                    # 2. Statistiky (Průměr a Směrodatná odchylka) [cite: 144]
                    mean_val = np.mean(best_values)
                    std_val = np.std(best_values)

                    results_table.loc["Mean", algo_name] = mean_val
                    results_table.loc["Std. Dev.", algo_name] = std_val

                    # Malý progress v konzoli (tečka za každý hotový algoritmus)
                    print(".", end="", flush=True)

                # Uložení listu do Excelu (název listu = název funkce)
                results_table.to_excel(writer, sheet_name=func_instance.name)
                print(" Done.")

        print("=" * 60)
        print(f"Benchmark finished successfully!")
        print(f"Results saved to: {os.path.abspath(OUTPUT_FILE)}")

    except ImportError:
        print("\n\nERROR: Chybí knihovna 'openpyxl'. Nainstalujte ji příkazem: pip install openpyxl")
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    run_benchmark()
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from itertools import permutations

# Makespan calculation
def calculate_makespan(jobs_order, processing_times):
    num_jobs = len(jobs_order)
    num_machines = len(processing_times[0])
    completion = np.zeros((num_jobs, num_machines))

    for i, job in enumerate(jobs_order):
        for m in range(num_machines):
            if i == 0 and m == 0:
                completion[i][m] = processing_times[job][m]
            elif i == 0:
                completion[i][m] = completion[i][m-1] + processing_times[job][m]
            elif m == 0:
                completion[i][m] = completion[i-1][m] + processing_times[job][m]
            else:
                completion[i][m] = max(completion[i-1][m], completion[i][m-1]) + processing_times[job][m]
    return completion[-1][-1]

# NEH algorithm
def neh_algorithm(processing_times):
    num_jobs = len(processing_times)
    total_times = [sum(job) for job in processing_times]
    sorted_jobs = sorted(range(num_jobs), key=lambda x: -total_times[x])
    partial_sequence = [sorted_jobs[0]]

    for job in sorted_jobs[1:]:
        best_seq = []
        best_makespan = float('inf')
        for i in range(len(partial_sequence) + 1):
            temp_seq = partial_sequence[:i] + [job] + partial_sequence[i:]
            temp_makespan = calculate_makespan(temp_seq, processing_times)
            if temp_makespan < best_makespan:
                best_makespan = temp_makespan
                best_seq = temp_seq
        partial_sequence = best_seq
    return partial_sequence, calculate_makespan(partial_sequence, processing_times)

# Random heuristic
def random_schedule(processing_times, trials=100):
    best_makespan = float('inf')
    for _ in range(trials):
        perm = list(range(len(processing_times)))
        random.shuffle(perm)
        ms = calculate_makespan(perm, processing_times)
        if ms < best_makespan:
            best_makespan = ms
    return best_makespan

# SPT heuristic
def spt_heuristic(processing_times):
    total_times = [sum(job) for job in processing_times]
    sorted_jobs = sorted(range(len(processing_times)), key=lambda x: total_times[x])
    return calculate_makespan(sorted_jobs, processing_times)

# Brute-force
def brute_force(processing_times):
    jobs = list(range(len(processing_times)))
    best_order = None
    best_makespan = float('inf')
    for perm in permutations(jobs):
        ms = calculate_makespan(perm, processing_times)
        if ms < best_makespan:
            best_makespan = ms
            best_order = perm
    return list(best_order), best_makespan

# Generate data
def generate_random_processing_times(num_jobs, num_machines, seed=None):
    if seed is not None:
        random.seed(seed)
    return [[random.randint(1, 5) for _ in range(num_machines)] for _ in range(num_jobs)]

# Create results folder
def ensure_results_folder():
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Main test loop
def test_all_methods(job_sizes, machine_counts, brute_force_limit=7):
    results = {}

    for machines in machine_counts:
        print(f"\n--- TEST FOR {machines} MACHINES ---")
        neh_times, rand_times, spt_times, brute_times = [], [], [], []
        neh_makespans, random_makespans, spt_makespans, brute_makespans = [], [], [], []

        for jobs in job_sizes:
            processing_times = generate_random_processing_times(jobs, machines, seed=1)

            # NEH
            start = time.time()
            _, neh_ms = neh_algorithm(processing_times)
            end = time.time()
            neh_time = end - start
            neh_times.append(neh_time)
            neh_makespans.append(neh_ms)

            # Random
            start = time.time()
            rand_ms = random_schedule(processing_times)
            end = time.time()
            rand_time = end - start
            rand_times.append(rand_time)
            random_makespans.append(rand_ms)

            # SPT
            start = time.time()
            spt_ms = spt_heuristic(processing_times)
            end = time.time()
            spt_time = end - start
            spt_times.append(spt_time)
            spt_makespans.append(spt_ms)

            # Brute-force
            if jobs <= brute_force_limit:
                start = time.time()
                _, brute_ms = brute_force(processing_times)
                end = time.time()
                brute_time = end - start
                brute_times.append(brute_time)
            else:
                brute_ms = None
                brute_time = None
                brute_times.append(None)
            brute_makespans.append(brute_ms)

            # Print detailed timings
            if brute_time is not None:
                brute_str = f"{brute_ms} ({brute_time:.4f}s)"
            else:
                brute_str = "-"

            print(f"{jobs} jobs | NEH: {neh_ms} ({neh_time:.4f}s) | "
                  f"Rand: {rand_ms} ({rand_time:.4f}s) | "
                  f"SPT: {spt_ms} ({spt_time:.4f}s) | "
                  f"Brute: {brute_str}")

        results[machines] = {
            'jobs': job_sizes,
            'neh_times': neh_times,
            'rand_times': rand_times,
            'spt_times': spt_times,
            'brute_times': brute_times,
            'neh': neh_makespans,
            'rand': random_makespans,
            'spt': spt_makespans,
            'brute': brute_makespans
        }

    return results

# Plots
def plot_and_save_results(results):
    results_dir = ensure_results_folder()

    for machines, data in results.items():
        jobs = data['jobs']
        neh_ms = data['neh']
        rand_ms = data['rand']
        spt_ms = data['spt']
        brute_ms = data['brute']

        neh_times = data['neh_times']
        rand_times = data['rand_times']
        spt_times = data['spt_times']
        brute_times = data['brute_times']

        # Makespan comparison
        plt.figure(figsize=(10, 5))
        plt.plot(jobs, neh_ms, marker='o', label='NEH')
        plt.plot(jobs, rand_ms, marker='s', label='Random')
        plt.plot(jobs, spt_ms, marker='^', label='SPT')
        brute_jobs = [j for j, b in zip(jobs, brute_ms) if b is not None]
        brute_vals = [b for b in brute_ms if b is not None]
        if brute_vals:
            plt.plot(brute_jobs, brute_vals, marker='x', label='OPT (Brute-force)')
        plt.title(f"Makespan porównanie metod (maszyn: {machines})")
        plt.xlabel("Liczba zadań")
        plt.ylabel("Makespan")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"makespan_porownanie_maszyny_{machines}.png"))
        plt.close()

        # Czas działania wszystkich metod
        plt.figure(figsize=(10, 5))
        plt.plot(jobs, neh_times, marker='o', label='NEH')
        plt.plot(jobs, rand_times, marker='s', label='Random')
        plt.plot(jobs, spt_times, marker='^', label='SPT')

        plt.title(f"Czas działania metod (maszyn: {machines})")
        plt.xlabel("Liczba zadań")
        plt.ylabel("Czas [s]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"czas_porownanie_maszyny_{machines}.png"))
        plt.close()

# Run
if __name__ == "__main__":
    job_sizes = [5, 6, 7, 10, 15, 20, 30]
    machine_counts = [3, 5, 10]
    results = test_all_methods(job_sizes, machine_counts)
    plot_and_save_results(results)

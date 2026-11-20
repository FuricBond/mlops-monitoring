"""
MLOps Assignment 8: Controlled Prediction Traffic Generator
Simulates realistic prediction traffic against the deployed Iris model service.
Supports 'normal' traffic and 'errors' injection mode with rate and duration controls.
"""

import argparse
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import requests

# Base Iris feature distributions for representative species
# Used to generate realistic measurements adhering to botanical ranges
IRIS_SPECIES_PROFILES = {
    "setosa": {
        "sepal_length": (4.8, 5.2),
        "sepal_width": (3.2, 3.8),
        "petal_length": (1.3, 1.7),
        "petal_width": (0.1, 0.4),
    },
    "versicolor": {
        "sepal_length": (5.5, 6.3),
        "sepal_width": (2.5, 3.0),
        "petal_length": (3.9, 4.7),
        "petal_width": (1.1, 1.5),
    },
    "virginica": {
        "sepal_length": (6.2, 7.2),
        "sepal_width": (2.8, 3.4),
        "petal_length": (5.1, 6.2),
        "petal_width": (1.8, 2.4),
    },
}


def generate_valid_iris_payload() -> Tuple[Dict[str, float], str]:
    """
    Generates a botanically realistic and valid Iris measurement payload.
    Adds small random variations while strictly adhering to Pydantic constraints (0 < val <= 20).
    """
    species = random.choice(list(IRIS_SPECIES_PROFILES.keys()))
    profile = IRIS_SPECIES_PROFILES[species]

    payload = {
        "sepal_length": round(random.uniform(*profile["sepal_length"]) + random.uniform(-0.1, 0.1), 2),
        "sepal_width": round(random.uniform(*profile["sepal_width"]) + random.uniform(-0.1, 0.1), 2),
        "petal_length": round(random.uniform(*profile["petal_length"]) + random.uniform(-0.1, 0.1), 2),
        "petal_width": round(max(0.05, random.uniform(*profile["petal_width"]) + random.uniform(-0.05, 0.05)), 2),
    }

    # Ensure all values remain strictly within valid Pydantic bounds (0 < x <= 20)
    for k in payload:
        payload[k] = min(max(payload[k], 0.05), 20.0)

    return payload, species


def generate_invalid_iris_payload() -> Tuple[Dict[str, Any], str]:
    """
    Generates an invalid payload designed to trigger HTTP 422 Unprocessable Entity
    from the FastAPI Pydantic schema validation.
    """
    payload, base_species = generate_valid_iris_payload()
    invalid_cases = [
        ("negative_sepal_length", lambda p: p.update({"sepal_length": -round(random.uniform(0.5, 5.0), 2)})),
        ("missing_petal_width", lambda p: p.pop("petal_width", None)),
        ("zero_petal_width", lambda p: p.update({"petal_width": 0.0})),
        ("excessive_sepal_length", lambda p: p.update({"sepal_length": round(random.uniform(20.5, 30.0), 2)})),
        ("missing_sepal_length", lambda p: p.pop("sepal_length", None)),
        ("zero_sepal_width", lambda p: p.update({"sepal_width": 0.0})),
    ]

    fault_name, fault_func = random.choice(invalid_cases)
    fault_func(payload)
    return payload, f"invalid ({fault_name})"


def run_traffic_simulation(
    mode: str,
    duration: float,
    rate: float,
    target_url: str = "http://127.0.0.1:8000/predict",
    timeout: float = 5.0,
) -> None:
    """
    Executes controlled traffic generation against the prediction service.
    """
    if rate <= 0:
        raise ValueError("Rate must be greater than 0 requests per second.")
    if duration <= 0:
        raise ValueError("Duration must be greater than 0 seconds.")

    request_interval = 1.0 / rate
    start_time = time.time()
    end_time = start_time + duration

    # Metrics counters
    total_requests = 0
    status_200_count = 0
    status_422_count = 0
    other_errors_count = 0
    client_latencies = []

    print("=" * 70)
    print("IRIS PREDICTION SERVICE - TRAFFIC SIMULATOR")
    print("=" * 70)
    print(f"Target URL:        {target_url}")
    print(f"Simulation Mode:   {mode.upper()}")
    print(f"Target Rate:       {rate:.1f} req/s (interval: {request_interval:.3f}s)")
    print(f"Planned Duration:  {duration:.1f} seconds")
    print(f"Timeout:           {timeout:.1f} seconds")
    if mode == "errors":
        print("Error Injection:   ~30% intentionally invalid requests (HTTP 422)")
    else:
        print("Traffic Profile:   100% valid Iris measurements (HTTP 200)")
    print("=" * 70)
    print(f"{'Time':<9} | {'Req #':<6} | {'Status':<10} | {'Client Latency':<16} | {'Profile/Detail'}")
    print("-" * 70)

    session = requests.Session()

    try:
        while time.time() < end_time:
            iter_start = time.perf_counter()

            # Determine payload based on mode
            if mode == "errors" and random.random() < 0.30:
                payload, detail = generate_invalid_iris_payload()
            else:
                payload, detail = generate_valid_iris_payload()

            total_requests += 1
            status_desc = "ERROR"
            call_start = time.perf_counter()

            try:
                response = session.post(
                    target_url,
                    json=payload,
                    timeout=timeout,
                )
                call_duration = time.perf_counter() - call_start
                client_latencies.append(call_duration)
                status_code = response.status_code

                if status_code == 200:
                    status_200_count += 1
                    status_desc = "200 OK"
                    # Extract prediction for progress log
                    pred_data = response.json()
                    detail_str = f"{detail} -> {pred_data.get('predicted_species', 'N/A')} ({pred_data.get('confidence', 0):.2f})"
                elif status_code == 422:
                    status_422_count += 1
                    status_desc = "422 INVALID"
                    detail_str = detail
                else:
                    other_errors_count += 1
                    status_desc = f"{status_code} ERROR"
                    detail_str = f"Unexpected response: {response.text[:40]}"

                elapsed = time.time() - start_time
                print(
                    f"{elapsed:>6.1f}s   | {total_requests:<6} | {status_desc:<10} | "
                    f"{call_duration * 1000:>7.2f} ms      | {detail_str}"
                )

            except requests.exceptions.Timeout:
                call_duration = time.perf_counter() - call_start
                other_errors_count += 1
                elapsed = time.time() - start_time
                print(
                    f"{elapsed:>6.1f}s   | {total_requests:<6} | {'TIMEOUT':<10} | "
                    f"{call_duration * 1000:>7.2f} ms      | Request timed out after {timeout}s"
                )
            except requests.exceptions.ConnectionError as conn_err:
                other_errors_count += 1
                elapsed = time.time() - start_time
                print(
                    f"{elapsed:>6.1f}s   | {total_requests:<6} | {'CONN_ERR':<10} | "
                    f"{'N/A':>10}      | Connection failed: {conn_err}"
                )
            except Exception as exc:
                other_errors_count += 1
                elapsed = time.time() - start_time
                print(
                    f"{elapsed:>6.1f}s   | {total_requests:<6} | {'EXCEPTION':<10} | "
                    f"{'N/A':>10}      | Unexpected error: {exc}"
                )

            # Enforce pacing to match target rate
            spent_time = time.perf_counter() - iter_start
            sleep_time = max(0.0, request_interval - spent_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] Simulation interrupted by user (Ctrl+C). Generating summary...")
    finally:
        session.close()

    actual_duration = time.time() - start_time
    avg_latency_ms = (
        (sum(client_latencies) / len(client_latencies) * 1000)
        if client_latencies
        else 0.0
    )
    actual_rate = total_requests / actual_duration if actual_duration > 0 else 0.0

    print("=" * 70)
    print("TRAFFIC SIMULATION EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Execution Mode:                 {mode.upper()}")
    print(f"Target Rate:                    {rate:.2f} req/s")
    print(f"Achieved Rate:                  {actual_rate:.2f} req/s")
    print(f"Elapsed Time:                   {actual_duration:.2f} seconds")
    print(f"Total Requests Sent:            {total_requests}")
    print(f"HTTP 200 (Success):             {status_200_count} ({status_200_count/total_requests*100:.1f}%)" if total_requests else "HTTP 200 (Success): 0")
    print(f"HTTP 422 (Validation Error):    {status_422_count} ({status_422_count/total_requests*100:.1f}%)" if total_requests else "HTTP 422 (Validation Error): 0")
    print(f"Other Errors:                   {other_errors_count}")
    print(f"Client-Observed Avg Latency:    {avg_latency_ms:.2f} ms ({avg_latency_ms/1000:.4f} s)")
    print("-" * 70)
    print("LATENCY MEASUREMENT DISTINCTION:")
    print("  * Client-Observed Response Time:")
    print("    Includes client-side socket creation, network transmission, HTTP payload")
    print("    serialization/deserialization, server execution, and network response return.")
    print("  * Server-Side Prometheus Latency (ml_prediction_latency_seconds):")
    print("    Measures strictly the FastAPI internal processing time (middleware to endpoint)")
    print("    excluding external client and network overheads.")
    print("=" * 70)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled prediction traffic generator for Iris ML model service."
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "errors"],
        default="normal",
        help="Traffic mode: 'normal' sends 100%% valid requests, 'errors' injects ~30%% invalid requests.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Traffic generation duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=3.0,
        help="Target request rate in requests per second (default: 3.0, typical range: 2-5).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="Target prediction endpoint URL (default: http://127.0.0.1:8000/predict).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP request timeout in seconds (default: 5.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_traffic_simulation(
        mode=args.mode,
        duration=args.duration,
        rate=args.rate,
        target_url=args.url,
        timeout=args.timeout,
    )

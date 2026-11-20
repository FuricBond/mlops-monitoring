"""
MLOps Assignment 8: Controlled Prediction Traffic and Synthetic Feedback Generator
Simulates realistic prediction traffic against the deployed Iris model service.
Supports:
  - Modes: 'normal', 'errors' injection, 'drift' simulation, and feedback demonstration modes.
  - Synthetic Ground-Truth Feedback: 'normal' (high accuracy ~95%) and 'degraded' (low accuracy ~40%).
"""

import argparse
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import requests

# Supported Iris flower species names
ALL_SPECIES = ["setosa", "versicolor", "virginica"]

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


def generate_drift_iris_payload() -> Tuple[Dict[str, float], str]:
    """
    Generates valid Iris measurements with a deliberate distribution shift (data drift).
    Significantly shifts sepal length and petal length upwards towards high virginica/oversized ranges
    while strictly remaining within the API validation bounds (0 < val <= 20).
    """
    payload = {
        "sepal_length": round(random.uniform(7.4, 8.4), 2),
        "sepal_width": round(random.uniform(2.8, 3.4), 2),
        "petal_length": round(random.uniform(5.9, 6.9), 2),
        "petal_width": round(random.uniform(2.0, 2.4), 2),
    }

    for k in payload:
        payload[k] = min(max(payload[k], 0.05), 20.0)

    return payload, f"drift (SL={payload['sepal_length']}, PL={payload['petal_length']})"


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
    feedback: str = "none",
    target_url: str = "http://127.0.0.1:8000/predict",
    feedback_url: str = "http://127.0.0.1:8000/feedback",
    timeout: float = 5.0,
) -> None:
    """
    Executes controlled traffic and optional synthetic feedback generation against the prediction service.
    """
    if rate <= 0:
        raise ValueError("Rate must be greater than 0 requests per second.")
    if duration <= 0:
        raise ValueError("Duration must be greater than 0 seconds.")

    # Resolve combined mode aliases
    effective_mode = mode
    effective_feedback = feedback

    if mode == "feedback-normal":
        effective_mode = "normal"
        effective_feedback = "normal"
    elif mode == "feedback-degraded":
        effective_mode = "normal"
        effective_feedback = "degraded"

    request_interval = 1.0 / rate
    start_time = time.time()
    end_time = start_time + duration

    # Metrics counters
    total_requests = 0
    status_200_count = 0
    status_422_count = 0
    other_errors_count = 0
    client_latencies = []

    # Feedback counters
    feedback_sent = 0
    feedback_correct = 0
    feedback_incorrect = 0
    feedback_errors = 0

    print("=" * 75)
    print("IRIS PREDICTION SERVICE - TRAFFIC & SYNTHETIC FEEDBACK GENERATOR")
    print("=" * 75)
    print(f"Target Predict URL:  {target_url}")
    print(f"Feedback URL:        {feedback_url if effective_feedback != 'none' else 'DISABLED'}")
    print(f"Simulation Mode:     {effective_mode.upper()}")
    print(f"Feedback Mode:       {effective_feedback.upper()} (SYNTHETIC)")
    print(f"Target Rate:         {rate:.1f} req/s (interval: {request_interval:.3f}s)")
    print(f"Planned Duration:    {duration:.1f} seconds")
    print(f"Timeout:             {timeout:.1f} seconds")

    if effective_feedback == "normal":
        print("Feedback Profile:    SYNTHETIC High Accuracy (~95% correct feedback)")
    elif effective_feedback == "degraded":
        print("Feedback Profile:    SYNTHETIC Degraded Accuracy (~40% correct feedback, <85% alert trigger)")

    if effective_mode == "errors":
        print("Traffic Profile:     Error Injection (~30% intentionally invalid requests -> HTTP 422)")
    elif effective_mode == "drift":
        print("Traffic Profile:     Drift injection (shifted sepal & petal lengths -> HTTP 200)")
    else:
        print("Traffic Profile:     Valid Iris measurements (HTTP 200)")

    if effective_feedback != "none":
        print("-" * 75)
        print("[!] NOTICE: All feedback sent is SYNTHETIC for monitoring demonstration.")
        print("    It does NOT represent genuine production accuracy or ground truth.")
    print("=" * 75)
    print(f"{'Time':<8} | {'Req #':<6} | {'Status':<10} | {'Latency':<10} | {'Details / Synthetic Feedback'}")
    print("-" * 75)

    session = requests.Session()

    try:
        while time.time() < end_time:
            iter_start = time.perf_counter()

            # Determine payload based on mode
            if effective_mode == "errors" and random.random() < 0.30:
                payload, detail = generate_invalid_iris_payload()
            elif effective_mode == "drift":
                payload, detail = generate_drift_iris_payload()
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
                    pred_data = response.json()
                    pred_species = pred_data.get("predicted_species", "unknown")
                    conf = pred_data.get("confidence", 0.0)
                    detail_str = f"{detail} -> {pred_species} ({conf:.2f})"

                    # Handle synthetic ground-truth feedback if enabled
                    if effective_feedback in ("normal", "degraded") and pred_species in ALL_SPECIES:
                        if effective_feedback == "normal":
                            # ~95% correct feedback
                            is_correct = random.random() < 0.95
                        else:
                            # ~40% correct feedback (degraded performance)
                            is_correct = random.random() < 0.40

                        if is_correct:
                            actual_species = pred_species
                            expected_outcome = "correct"
                        else:
                            other_species = [s for s in ALL_SPECIES if s != pred_species]
                            actual_species = random.choice(other_species)
                            expected_outcome = "incorrect"

                        try:
                            fb_resp = session.post(
                                feedback_url,
                                json={"predicted": pred_species, "actual": actual_species},
                                timeout=timeout,
                            )
                            if fb_resp.status_code == 200:
                                feedback_sent += 1
                                if expected_outcome == "correct":
                                    feedback_correct += 1
                                else:
                                    feedback_incorrect += 1
                                detail_str += f" | [SYNTHETIC FB] pred={pred_species}, act={actual_species} ({expected_outcome})"
                            else:
                                feedback_errors += 1
                                detail_str += f" | [FB_ERR: {fb_resp.status_code}]"
                        except Exception as fb_exc:
                            feedback_errors += 1
                            detail_str += f" | [FB_EXC: {fb_exc}]"

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
                    f"{elapsed:>5.1f}s   | {total_requests:<6} | {status_desc:<10} | "
                    f"{call_duration * 1000:>6.1f} ms | {detail_str}"
                )

            except requests.exceptions.Timeout:
                call_duration = time.perf_counter() - call_start
                other_errors_count += 1
                elapsed = time.time() - start_time
                print(
                    f"{elapsed:>5.1f}s   | {total_requests:<6} | {'TIMEOUT':<10} | "
                    f"{call_duration * 1000:>6.1f} ms | Request timed out after {timeout}s"
                )
            except requests.exceptions.ConnectionError as conn_err:
                other_errors_count += 1
                elapsed = time.time() - start_time
                print(
                    f"{elapsed:>5.1f}s   | {total_requests:<6} | {'CONN_ERR':<10} | "
                    f"{'N/A':>9} | Connection failed: {conn_err}"
                )
            except Exception as exc:
                other_errors_count += 1
                elapsed = time.time() - start_time
                print(
                    f"{elapsed:>5.1f}s   | {total_requests:<6} | {'EXCEPTION':<10} | "
                    f"{'N/A':>9} | Unexpected error: {exc}"
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

    print("=" * 75)
    print("TRAFFIC & FEEDBACK SIMULATION SUMMARY")
    print("=" * 75)
    print(f"Execution Mode:                 {effective_mode.upper()}")
    print(f"Feedback Mode:                  {effective_feedback.upper()} (SYNTHETIC)")
    print(f"Target Rate:                    {rate:.2f} req/s")
    print(f"Achieved Rate:                  {actual_rate:.2f} req/s")
    print(f"Elapsed Time:                   {actual_duration:.2f} seconds")
    print(f"Total Prediction Requests:      {total_requests}")
    print(f"HTTP 200 (Success):             {status_200_count} ({status_200_count/total_requests*100:.1f}%)" if total_requests else "HTTP 200 (Success): 0")
    print(f"HTTP 422 (Validation Error):    {status_422_count} ({status_422_count/total_requests*100:.1f}%)" if total_requests else "HTTP 422 (Validation Error): 0")
    print(f"Other Prediction Errors:        {other_errors_count}")
    print(f"Client-Observed Avg Latency:    {avg_latency_ms:.2f} ms ({avg_latency_ms/1000:.4f} s)")

    if effective_feedback != "none":
        sim_acc = (feedback_correct / feedback_sent * 100.0) if feedback_sent else 0.0
        print("-" * 75)
        print("SYNTHETIC GROUND-TRUTH FEEDBACK SUMMARY:")
        print(f"Total Feedback Sent:            {feedback_sent}")
        print(f"Feedback 'correct':             {feedback_correct} ({feedback_correct/feedback_sent*100:.1f}%)" if feedback_sent else "Feedback 'correct': 0")
        print(f"Feedback 'incorrect':           {feedback_incorrect} ({feedback_incorrect/feedback_sent*100:.1f}%)" if feedback_sent else "Feedback 'incorrect': 0")
        print(f"Feedback Network Errors:        {feedback_errors}")
        print(f"Synthetic Accuracy:             {sim_acc:.1f}%")
        print("-" * 75)
        print("[DISCLAIMER] All ground-truth feedback generated in this simulation is")
        print("synthetic and intended solely for monitoring demonstration. It does NOT")
        print("represent genuine production accuracy.")

    print("=" * 75)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled prediction traffic and synthetic feedback generator for Iris ML model service."
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "errors", "drift", "feedback-normal", "feedback-degraded"],
        default="normal",
        help="Traffic mode: 'normal', 'errors', 'drift', 'feedback-normal', or 'feedback-degraded'.",
    )
    parser.add_argument(
        "--feedback",
        choices=["none", "normal", "degraded"],
        default="none",
        help="Synthetic ground-truth feedback mode: 'none', 'normal' (~95%% correct), 'degraded' (~40%% correct).",
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
        help="Target request rate in requests per second (default: 3.0).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="Target prediction endpoint URL (default: http://127.0.0.1:8000/predict).",
    )
    parser.add_argument(
        "--feedback-url",
        type=str,
        default="http://127.0.0.1:8000/feedback",
        help="Target feedback endpoint URL (default: http://127.0.0.1:8000/feedback).",
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
        feedback=args.feedback,
        target_url=args.url,
        feedback_url=args.feedback_url,
        timeout=args.timeout,
    )

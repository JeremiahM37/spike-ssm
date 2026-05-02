"""Tests for RTOS task scheduling simulation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.rtos_sim.task_scheduler import (
    RTOSSimulator, Task, Priority,
    create_spiking_lm_tasks,
)


def test_create_tasks():
    tasks = create_spiking_lm_tasks()
    assert len(tasks) == 3
    names = {t.name for t in tasks}
    assert "inference" in names
    assert "mf_learning" in names
    assert "telemetry" in names


def test_utilization():
    sim = RTOSSimulator()
    for task in create_spiking_lm_tasks():
        sim.add_task(task)
    u = sim.utilization()
    assert 0 < u < 1.0  # Should be utilizable


def test_schedulability():
    sim = RTOSSimulator()
    # Use light task set that passes RM bound
    tasks = create_spiking_lm_tasks(
        inference_period_ms=100,
        inference_wcet_ms=30,
        mf_period_ms=2000,
        mf_wcet_ms=200,
        telemetry_period_ms=10000,
        telemetry_wcet_ms=10,
    )
    for task in tasks:
        sim.add_task(task)
    assert sim.schedulable()


def test_simulation_runs():
    sim = RTOSSimulator()
    for task in create_spiking_lm_tasks():
        sim.add_task(task)
    metrics = sim.simulate(duration_ms=5000, time_step_ms=1.0)
    assert metrics.total_time_ms == 5000
    assert metrics.completions["inference"] > 0
    assert metrics.completions["mf_learning"] > 0


def test_inference_not_starved():
    """Inference (high priority) should always complete on time."""
    sim = RTOSSimulator()
    for task in create_spiking_lm_tasks():
        sim.add_task(task)
    metrics = sim.simulate(duration_ms=10000)
    assert metrics.deadline_misses["inference"] == 0


def test_high_load():
    """Under high load, low-priority MF should be preempted but still run."""
    sim = RTOSSimulator()
    tasks = create_spiking_lm_tasks(
        inference_period_ms=20,   # Very frequent inference
        inference_wcet_ms=15,
        mf_period_ms=500,
        mf_wcet_ms=100,
    )
    for task in tasks:
        sim.add_task(task)
    metrics = sim.simulate(duration_ms=5000)
    # MF should still complete some updates
    assert metrics.completions["mf_learning"] > 0
    # Inference should not miss deadlines
    assert metrics.deadline_misses["inference"] == 0

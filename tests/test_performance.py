"""Performance benchmarks for JuniperData client operations (CAN-DEF-007)."""

import os
import time
from typing import Any, Dict, List

import pytest

from juniper_data_client.testing import FakeDataClient

# ---------------------------------------------------------------------------
# Environment gate for live-service benchmarks
# ---------------------------------------------------------------------------

BENCHMARK_LIVE = os.environ.get("JUNIPER_DATA_BENCHMARK", "0") == "1"
JUNIPER_DATA_URL = os.environ.get("JUNIPER_DATA_URL", "http://localhost:8100")

pytestmark = [pytest.mark.performance]


# ======================================================================
# Helpers
# ======================================================================


def _measure(func, *args: Any, **kwargs: Any) -> tuple:
    """Execute *func* and return (elapsed_seconds, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


# ======================================================================
# FakeDataClient performance tests — always run
# ======================================================================


class TestFakeClientPerformance:
    """Performance benchmarks against the in-memory FakeDataClient.

    These tests always run (no live service required). They assert sub-100 ms
    for single operations and sub-1 s for batch operations to guard against
    unexpected regressions in the synthetic generators and in-memory store.
    """

    # ------------------------------------------------------------------
    # Single dataset creation latency
    # ------------------------------------------------------------------

    def test_create_spiral_latency(self, fake_client: FakeDataClient) -> None:
        """Single spiral dataset creation completes in < 100 ms."""
        elapsed, result = _measure(fake_client.create_spiral_dataset, n_spirals=2, seed=42)

        assert "dataset_id" in result
        assert elapsed < 0.1, f"Spiral creation took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake spiral create: {elapsed * 1000:.2f} ms")

    def test_create_xor_latency(self, fake_client: FakeDataClient) -> None:
        """Single XOR dataset creation completes in < 100 ms."""
        elapsed, result = _measure(fake_client.create_dataset, "xor", {"n_points": 200, "seed": 7})

        assert "dataset_id" in result
        assert elapsed < 0.1, f"XOR creation took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake xor create: {elapsed * 1000:.2f} ms")

    def test_create_circle_latency(self, fake_client: FakeDataClient) -> None:
        """Single circle dataset creation completes in < 100 ms."""
        elapsed, result = _measure(fake_client.create_dataset, "circle", {"n_points": 200, "seed": 3})

        assert "dataset_id" in result
        assert elapsed < 0.1, f"Circle creation took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake circle create: {elapsed * 1000:.2f} ms")

    def test_create_moon_latency(self, fake_client: FakeDataClient) -> None:
        """Single moon dataset creation completes in < 100 ms."""
        elapsed, result = _measure(fake_client.create_dataset, "moon", {"n_points": 200, "seed": 5})

        assert "dataset_id" in result
        assert elapsed < 0.1, f"Moon creation took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake moon create: {elapsed * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Batch creation throughput
    # ------------------------------------------------------------------

    def test_batch_create_10_datasets(self, fake_client: FakeDataClient) -> None:
        """Batch-creating 10 datasets completes in < 1 s."""
        specs: List[Dict[str, Any]] = [{"generator": "spiral", "params": {"n_spirals": 2, "seed": i}} for i in range(10)]

        elapsed, result = _measure(fake_client.batch_create, specs)

        assert result["total_created"] == 10
        assert result["total_failed"] == 0
        assert elapsed < 1.0, f"Batch create (10) took {elapsed:.4f}s (limit 1.0s)"
        print(f"  fake batch create 10: {elapsed * 1000:.2f} ms ({elapsed / 10 * 1000:.2f} ms/dataset)")

    def test_batch_create_30_datasets(self, fake_client: FakeDataClient) -> None:
        """Batch-creating 30 datasets completes in < 1 s."""
        generators = ["spiral", "xor", "circle", "moon"]
        specs: List[Dict[str, Any]] = [{"generator": generators[i % len(generators)], "params": {"seed": i}} for i in range(30)]

        elapsed, result = _measure(fake_client.batch_create, specs)

        assert result["total_created"] == 30
        assert result["total_failed"] == 0
        assert elapsed < 1.0, f"Batch create (30) took {elapsed:.4f}s (limit 1.0s)"
        print(f"  fake batch create 30: {elapsed * 1000:.2f} ms ({elapsed / 30 * 1000:.2f} ms/dataset)")

    # ------------------------------------------------------------------
    # Artifact download latency
    # ------------------------------------------------------------------

    def test_download_artifact_npz_latency(self, fake_client: FakeDataClient) -> None:
        """NPZ artifact download (in-memory) completes in < 100 ms."""
        ds = fake_client.create_spiral_dataset(n_spirals=3, n_points_per_spiral=200, seed=99)
        dataset_id = ds["dataset_id"]

        elapsed, arrays = _measure(fake_client.download_artifact_npz, dataset_id)

        assert "X_train" in arrays
        assert elapsed < 0.1, f"NPZ download took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake npz download: {elapsed * 1000:.2f} ms")

    def test_download_artifact_bytes_latency(self, fake_client: FakeDataClient) -> None:
        """Raw NPZ bytes download completes in < 100 ms."""
        ds = fake_client.create_spiral_dataset(n_spirals=2, seed=10)
        dataset_id = ds["dataset_id"]

        elapsed, raw = _measure(fake_client.download_artifact_bytes, dataset_id)

        assert isinstance(raw, bytes)
        assert len(raw) > 0
        assert elapsed < 0.1, f"Bytes download took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake bytes download: {elapsed * 1000:.2f} ms ({len(raw)} bytes)")

    # ------------------------------------------------------------------
    # Dataset listing / pagination performance
    # ------------------------------------------------------------------

    def test_list_datasets_pagination(self, fake_client: FakeDataClient) -> None:
        """Listing datasets with pagination completes in < 100 ms per page."""
        # Populate 50 datasets
        for i in range(50):
            fake_client.create_dataset("xor", {"seed": i})

        # Page through in chunks of 10
        total_elapsed = 0.0
        total_ids: List[str] = []
        for offset in range(0, 50, 10):
            elapsed, page = _measure(fake_client.list_datasets, limit=10, offset=offset)
            total_elapsed += elapsed
            total_ids.extend(page)
            assert elapsed < 0.1, f"Listing page at offset={offset} took {elapsed:.4f}s (limit 0.1s)"

        assert len(total_ids) == 50
        print(f"  fake list 50 datasets (5 pages): {total_elapsed * 1000:.2f} ms total")

    # ------------------------------------------------------------------
    # Metadata retrieval latency
    # ------------------------------------------------------------------

    def test_get_metadata_latency(self, fake_client: FakeDataClient) -> None:
        """Metadata retrieval for a single dataset completes in < 100 ms."""
        ds = fake_client.create_dataset("spiral", {"n_spirals": 2, "seed": 1})
        dataset_id = ds["dataset_id"]

        elapsed, meta = _measure(fake_client.get_dataset_metadata, dataset_id)

        assert meta["dataset_id"] == dataset_id
        assert meta["generator"] == "spiral"
        assert elapsed < 0.1, f"Metadata retrieval took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake metadata get: {elapsed * 1000:.2f} ms")

    def test_get_metadata_bulk_latency(self, fake_client: FakeDataClient) -> None:
        """Retrieving metadata for 20 datasets sequentially completes in < 1 s."""
        ids = []
        for i in range(20):
            ds = fake_client.create_dataset("xor", {"seed": i})
            ids.append(ds["dataset_id"])

        start = time.perf_counter()
        for dataset_id in ids:
            fake_client.get_dataset_metadata(dataset_id)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Bulk metadata (20) took {elapsed:.4f}s (limit 1.0s)"
        print(f"  fake metadata bulk (20): {elapsed * 1000:.2f} ms ({elapsed / 20 * 1000:.2f} ms/dataset)")

    # ------------------------------------------------------------------
    # Health check latency
    # ------------------------------------------------------------------

    def test_health_check_latency(self, fake_client: FakeDataClient) -> None:
        """Health check completes in < 100 ms."""
        elapsed, result = _measure(fake_client.health_check)

        assert result["status"] == "healthy"
        assert elapsed < 0.1, f"Health check took {elapsed:.4f}s (limit 0.1s)"
        print(f"  fake health check: {elapsed * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Batch export (ZIP) latency
    # ------------------------------------------------------------------

    def test_batch_export_latency(self, fake_client: FakeDataClient) -> None:
        """Exporting 5 datasets as a ZIP archive completes in < 1 s."""
        ids = []
        for i in range(5):
            ds = fake_client.create_dataset("spiral", {"n_spirals": 2, "seed": i + 100})
            ids.append(ds["dataset_id"])

        elapsed, zip_bytes = _measure(fake_client.batch_export, ids)

        assert isinstance(zip_bytes, bytes)
        assert len(zip_bytes) > 0
        assert elapsed < 1.0, f"Batch export (5) took {elapsed:.4f}s (limit 1.0s)"
        print(f"  fake batch export 5: {elapsed * 1000:.2f} ms ({len(zip_bytes)} bytes)")

    def test_batch_export_10_latency(self, fake_client: FakeDataClient) -> None:
        """Exporting 10 datasets as a ZIP archive completes in < 1 s."""
        ids = []
        generators = ["spiral", "xor", "circle", "moon"]
        for i in range(10):
            ds = fake_client.create_dataset(generators[i % len(generators)], {"seed": i + 200})
            ids.append(ds["dataset_id"])

        elapsed, zip_bytes = _measure(fake_client.batch_export, ids)

        assert isinstance(zip_bytes, bytes)
        assert len(zip_bytes) > 0
        assert elapsed < 1.0, f"Batch export (10) took {elapsed:.4f}s (limit 1.0s)"
        print(f"  fake batch export 10: {elapsed * 1000:.2f} ms ({len(zip_bytes)} bytes)")


# ======================================================================
# Live JuniperDataClient performance tests — gated behind env var
# ======================================================================


@pytest.mark.skipif(not BENCHMARK_LIVE, reason="Set JUNIPER_DATA_BENCHMARK=1 and JUNIPER_DATA_URL to run live benchmarks")
class TestLiveClientPerformance:
    """Performance benchmarks against a running JuniperData service.

    These tests only run when ``JUNIPER_DATA_BENCHMARK=1`` is set and a live
    service is reachable at ``JUNIPER_DATA_URL``. Timing results are printed
    but no hard assertions are made on latency — real service performance
    depends on hardware, network, and load.
    """

    @pytest.fixture(autouse=True)
    def _live_client(self):
        """Create a real JuniperDataClient for the test, close after."""
        from juniper_data_client import JuniperDataClient

        self.client = JuniperDataClient(base_url=JUNIPER_DATA_URL)
        yield
        self.client.close()

    # Helper to track created dataset IDs for cleanup
    @pytest.fixture(autouse=True)
    def _cleanup_datasets(self):
        """Track and clean up datasets created during the test."""
        self._created_ids: List[str] = []
        yield
        # Best-effort cleanup
        if self._created_ids:
            try:
                self.client.batch_delete(self._created_ids)
            except Exception:
                for dataset_id in self._created_ids:
                    try:
                        self.client.delete_dataset(dataset_id)
                    except Exception:
                        pass

    def _track(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Register a created dataset for post-test cleanup."""
        if "dataset_id" in result:
            self._created_ids.append(result["dataset_id"])
        return result

    # ------------------------------------------------------------------
    # Health check latency
    # ------------------------------------------------------------------

    def test_health_check_latency(self) -> None:
        """Measure health check round-trip time against the live service."""
        elapsed, result = _measure(self.client.health_check)

        assert result["status"] == "healthy"
        print(f"  live health check: {elapsed * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Single dataset creation latency
    # ------------------------------------------------------------------

    def test_create_spiral_latency(self) -> None:
        """Measure spiral dataset creation latency against the live service."""
        elapsed, result = _measure(self.client.create_spiral_dataset, n_spirals=2, seed=42)
        self._track(result)

        assert "dataset_id" in result
        print(f"  live spiral create: {elapsed * 1000:.2f} ms")

    def test_create_xor_latency(self) -> None:
        """Measure XOR dataset creation latency against the live service."""
        elapsed, result = _measure(self.client.create_dataset, "xor", {"n_points": 200, "seed": 7})
        self._track(result)

        assert "dataset_id" in result
        print(f"  live xor create: {elapsed * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Batch creation throughput
    # ------------------------------------------------------------------

    def test_batch_create_10_datasets(self) -> None:
        """Measure batch creation of 10 datasets against the live service."""
        specs: List[Dict[str, Any]] = [{"generator": "spiral", "params": {"n_spirals": 2, "seed": i}} for i in range(10)]

        elapsed, result = _measure(self.client.batch_create, specs)

        assert result["total_created"] == 10
        for item in result["results"]:
            if item.get("dataset_id"):
                self._created_ids.append(item["dataset_id"])
        print(f"  live batch create 10: {elapsed * 1000:.2f} ms ({elapsed / 10 * 1000:.2f} ms/dataset)")

    def test_batch_create_30_datasets(self) -> None:
        """Measure batch creation of 30 datasets against the live service."""
        generators = ["spiral", "xor", "circle", "moon"]
        specs: List[Dict[str, Any]] = [{"generator": generators[i % len(generators)], "params": {"seed": i}} for i in range(30)]

        elapsed, result = _measure(self.client.batch_create, specs)

        assert result["total_created"] == 30
        for item in result["results"]:
            if item.get("dataset_id"):
                self._created_ids.append(item["dataset_id"])
        print(f"  live batch create 30: {elapsed * 1000:.2f} ms ({elapsed / 30 * 1000:.2f} ms/dataset)")

    # ------------------------------------------------------------------
    # Artifact download latency
    # ------------------------------------------------------------------

    def test_download_artifact_npz_latency(self) -> None:
        """Measure NPZ artifact download latency against the live service."""
        ds = self._track(self.client.create_spiral_dataset(n_spirals=3, n_points_per_spiral=200, seed=99))

        elapsed, arrays = _measure(self.client.download_artifact_npz, ds["dataset_id"])

        assert "X_train" in arrays
        print(f"  live npz download: {elapsed * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Dataset listing / pagination performance
    # ------------------------------------------------------------------

    def test_list_datasets_pagination(self) -> None:
        """Measure dataset listing with pagination against the live service."""
        # Create 20 datasets to ensure we have data to paginate
        for i in range(20):
            ds = self.client.create_dataset("xor", {"seed": i + 5000})
            self._track(ds)

        total_elapsed = 0.0
        for offset in range(0, 20, 10):
            elapsed, page = _measure(self.client.list_datasets, limit=10, offset=offset)
            total_elapsed += elapsed

        print(f"  live list pagination (2 pages): {total_elapsed * 1000:.2f} ms total")

    # ------------------------------------------------------------------
    # Metadata retrieval latency
    # ------------------------------------------------------------------

    def test_get_metadata_latency(self) -> None:
        """Measure metadata retrieval latency against the live service."""
        ds = self._track(self.client.create_dataset("spiral", {"n_spirals": 2, "seed": 1}))

        elapsed, meta = _measure(self.client.get_dataset_metadata, ds["dataset_id"])

        assert meta["dataset_id"] == ds["dataset_id"]
        print(f"  live metadata get: {elapsed * 1000:.2f} ms")

    # ------------------------------------------------------------------
    # Batch export (ZIP) latency
    # ------------------------------------------------------------------

    def test_batch_export_latency(self) -> None:
        """Measure batch ZIP export latency against the live service."""
        ids = []
        for i in range(5):
            ds = self._track(self.client.create_dataset("spiral", {"n_spirals": 2, "seed": i + 100}))
            ids.append(ds["dataset_id"])

        elapsed, zip_bytes = _measure(self.client.batch_export, ids)

        assert isinstance(zip_bytes, bytes)
        assert len(zip_bytes) > 0
        print(f"  live batch export 5: {elapsed * 1000:.2f} ms ({len(zip_bytes)} bytes)")

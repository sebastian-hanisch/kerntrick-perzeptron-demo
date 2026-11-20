"""Kopier-Test: `ktp_rng.py` ist eine Bytekopie der Vorlage `tools/templates/rng_splitmix64_exact.py` des Website-Repos (SplitMix64, einheitliche Zufallsquelle der Linie).

Zwei Wege: (1) liegt das Website-Repo neben diesem Ordner, Bytegleichheit mit der Vorlage; (2) immer: Prüfsumme der Datei (SHA-256 über den Text mit LF) und feste Ausgaben,
damit auch ein Klon ohne das Website-Repo (CI) eine veränderte Kopie bemerkt. Plattform: die Ausgaben sind bitgleich auf Windows und Linux (normal() rechnet mit `decimal`)."""

import hashlib
from pathlib import Path

import ktp_rng

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
SHA256_LF = "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f"


def test_copy_is_byte_equal_to_the_template_if_the_website_repo_is_next_to_it():
    if not TEMPLATE.exists():
        import pytest

        pytest.skip("Vorlage nicht vorhanden (Klon ohne Website-Repo): Prüfsumme unten gilt")
    assert (ROOT / "ktp_rng.py").read_bytes().replace(b"\r\n", b"\n") == TEMPLATE.read_bytes().replace(b"\r\n", b"\n")


def test_copy_has_the_recorded_checksum():
    assert hashlib.sha256((ROOT / "ktp_rng.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest() == SHA256_LF


def test_splitmix64_known_sequences():
    r = ktp_rng.SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    r = ktp_rng.SplitMix64(42)
    assert [r.random().hex() for _ in range(2)] == ["0x1.7bae644c5fd6dp-1", "0x1.477f199d93378p-3"]


def test_below_and_shuffle_are_available_and_permute():
    r = ktp_rng.SplitMix64(1)
    assert all(0 <= r.below(7) < 7 for _ in range(100))
    items = list(range(20))
    r.shuffle(items)
    assert sorted(items) == list(range(20)) and items != list(range(20))

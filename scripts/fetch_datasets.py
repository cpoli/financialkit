"""Download the bundled datasets and write the CSVs in ``financialkit/datasets/data/``.

This is the only code in financialkit that touches the network (see
CLAUDE.md, "Bundled datasets"). The author runs it by hand, never from CI,
the tests or the gallery, and commits the CSVs it writes::

    python scripts/fetch_datasets.py            # download, then write
    python scripts/fetch_datasets.py --raw DIR  # reuse files already in DIR

Downloads go to ``--raw`` (a temporary directory by default) under the
names in ``SOURCES``. Only the standard library is used: Shiller's
workbook is old binary Excel, so a minimal OLE2/BIFF8 reader is included.
After writing, update the retrieval date in ``financialkit/datasets/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import math
import re
import struct
import tempfile
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "financialkit" / "datasets" / "data"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRED_TENORS = ("DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10")

SOURCES = {
    # shillerdata.com links to the workbook on its file host; if this
    # address moves, copy the new "ie_data.xls" link from the page.
    "ie_data.xls": "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/1449043e-a1ed-4e2e-8e9c-7b9a2a064493/ie_data.xls",
    "fred_treasury.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(FRED_TENORS),
    "ff_daily.zip": FRENCH + "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff3.zip": FRENCH + "F-F_Research_Data_Factors_CSV.zip",
    "ff5.zip": FRENCH + "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "momentum.zip": FRENCH + "F-F_Momentum_Factor_CSV.zip",
    "ff25.zip": FRENCH + "25_Portfolios_5x5_CSV.zip",
}


def download(raw: Path) -> None:
    for name, url in SOURCES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "financialkit-fetch-datasets"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (fixed https URLs)
            (raw / name).write_bytes(resp.read())
        print(f"downloaded {name}")


# --- Binary Excel (OLE2 compound file + BIFF8), numbers and shared strings only ---


def _ole_stream(data: bytes, name: str) -> bytes:
    """Return the named stream of an OLE2 compound file stored in regular sectors."""
    ssz = 1 << struct.unpack_from("<H", data, 0x1E)[0]
    n_fat, dir_start = struct.unpack_from("<II", data, 0x2C)
    difat_start, n_difat = struct.unpack_from("<II", data, 0x44)

    def sector(i: int) -> bytes:
        return data[512 + i * ssz : 512 + (i + 1) * ssz]

    fat_sectors = list(struct.unpack_from("<109I", data, 0x4C))
    s = difat_start
    for _ in range(n_difat):
        block = struct.unpack(f"<{ssz // 4}I", sector(s))
        fat_sectors += block[:-1]
        s = block[-1]
    fat: list[int] = []
    for i in fat_sectors[:n_fat]:
        fat += struct.unpack(f"<{ssz // 4}I", sector(i))

    def chain(start: int) -> bytes:
        out, s = [], start
        while s < 0xFFFFFFFA:
            out.append(sector(s))
            s = fat[s]
        return b"".join(out)

    directory = chain(dir_start)
    for off in range(0, len(directory), 128):
        n = struct.unpack_from("<H", directory, off + 64)[0]
        if directory[off : off + max(n - 2, 0)].decode("utf-16-le") == name:
            start, size = struct.unpack_from("<II", directory, off + 116)
            return chain(start)[:size]
    raise KeyError(name)


def _records(wb: bytes, pos: int = 0):
    while pos + 4 <= len(wb):
        rid, n = struct.unpack_from("<HH", wb, pos)
        yield rid, wb[pos + 4 : pos + 4 + n]
        pos += 4 + n


def _rk(v: int) -> float:
    """Decode an RK number: a 30-bit integer or the top 30 bits of a double, optionally divided by 100."""
    if v & 2:
        x = float((v >> 2) - (1 << 30) if v >= 1 << 31 else v >> 2)
    else:
        x = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return x / 100 if v & 1 else x


def _sst(pieces: list[bytes]) -> list[str]:
    """Decode a shared-string table split across SST and CONTINUE records."""
    strings: list[str] = []
    k, buf = 0, pieces[0]
    pos = 8
    total = struct.unpack_from("<I", buf, 4)[0]

    def more() -> None:
        nonlocal k, buf, pos
        k += 1
        buf, pos = pieces[k], 0

    for _ in range(total):
        if pos >= len(buf):
            more()
        nchar = struct.unpack_from("<H", buf, pos)[0]
        flags = buf[pos + 2]
        pos += 3
        n_runs = ext = 0
        if flags & 8:
            n_runs = struct.unpack_from("<H", buf, pos)[0]
            pos += 2
        if flags & 4:
            ext = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
        chars: list[str] = []
        wide = flags & 1
        while nchar:
            if pos >= len(buf):
                more()
                wide = buf[0] & 1
                pos = 1
            width = 2 if wide else 1
            take = min(nchar, (len(buf) - pos) // width)
            raw = buf[pos : pos + take * width]
            chars.append(raw.decode("utf-16-le" if wide else "latin-1"))
            pos += take * width
            nchar -= take
        skip = 4 * n_runs + ext
        while skip:
            if pos >= len(buf):
                more()
            step = min(skip, len(buf) - pos)
            pos += step
            skip -= step
        strings.append("".join(chars))
    return strings


def read_xls_sheet(path: Path, sheet: str) -> dict[tuple[int, int], float | str]:
    """Cells of one worksheet as ``{(row, col): value}`` (numbers and strings)."""
    wb = _ole_stream(path.read_bytes(), "Workbook")
    sheets: dict[str, int] = {}
    sst_pieces: list[bytes] = []
    last = 0
    for rid, body in _records(wb):
        if rid == 0x0085:  # BOUNDSHEET
            n, flags = body[6], body[7]
            sheets[body[8 : 8 + n * (2 if flags & 1 else 1)].decode("utf-16-le" if flags & 1 else "latin-1")] = struct.unpack_from("<I", body)[0]
        elif rid == 0x00FC:  # SST
            sst_pieces = [body]
        elif rid == 0x003C and last == 0x00FC:  # CONTINUE of the SST
            sst_pieces.append(body)
            continue
        elif rid == 0x000A:  # EOF of the workbook globals
            break
        last = rid
    strings = _sst(sst_pieces) if sst_pieces else []
    cells: dict[tuple[int, int], float | str] = {}
    for rid, body in _records(wb, sheets[sheet]):
        if rid == 0x0203:  # NUMBER
            r, c = struct.unpack_from("<HH", body)
            cells[r, c] = struct.unpack_from("<d", body, 6)[0]
        elif rid == 0x027E:  # RK
            r, c = struct.unpack_from("<HH", body)
            cells[r, c] = _rk(struct.unpack_from("<I", body, 6)[0])
        elif rid == 0x00BD:  # MULRK
            r, c = struct.unpack_from("<HH", body)
            for j in range((len(body) - 6) // 6):
                cells[r, c + j] = _rk(struct.unpack_from("<I", body, 6 + 6 * j)[0])
        elif rid == 0x0006:  # FORMULA: keep a cached numeric result
            r, c = struct.unpack_from("<HH", body)
            if body[12:14] != b"\xff\xff":
                cells[r, c] = struct.unpack_from("<d", body, 6)[0]
        elif rid == 0x00FD:  # LABELSST
            r, c, _, i = struct.unpack_from("<HHHI", body)
            cells[r, c] = strings[i]
        elif rid == 0x000A:
            break
    return cells


# --- Writers ---


def _fmt(x: float, digits: int) -> str:
    return "" if math.isnan(x) else f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _write(name: str, header: list[str], rows: list[list[str]], source: str) -> None:
    path = OUT / name
    with path.open("w", newline="") as fh:
        fh.write(f"# {source}\n")
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path.name}: {len(rows)} rows, {path.stat().st_size / 1024:.0f} KB")


def write_shiller(raw: Path) -> None:
    cells = read_xls_sheet(raw / "ie_data.xls", "Data")
    hdr = next(r for (r, c), v in cells.items() if c == 0 and v == "Date")
    want = {"Date": "date", "P": "price", "D": "dividend", "E": "earnings", "CPI": "cpi"}
    cols = {c: want[str(v).strip()] for (r, c), v in cells.items() if r == hdr and str(v).strip() in want}
    # The long rate's header spans two rows ("Rate GS10" on the row below "Long Interest").
    gs10 = next(c for (r, c), v in cells.items() if r in (hdr, hdr + 1) and isinstance(v, str) and "GS10" in v)
    cols[gs10] = "long_rate"
    order = ["date", "price", "dividend", "earnings", "cpi", "long_rate"]
    by_name = {v: c for c, v in cols.items()}
    rows = []
    r = hdr + 1
    while True:
        d = cells.get((r, by_name["date"]))
        if isinstance(d, float):
            year, month = int(d), round((d - int(d)) * 100)
            vals = [cells.get((r, by_name[k])) for k in order[1:]]
            num = [float(v) if isinstance(v, float) else math.nan for v in vals]
            if not math.isnan(num[0]):
                rows.append([f"{year}{month:02d}", _fmt(num[0], 2), _fmt(num[1], 4), _fmt(num[2], 4), _fmt(num[3], 3), _fmt(num[4], 2)])
        elif r > hdr + 10:
            break
        r += 1
    assert rows[0][0] == "187101" and abs(float(rows[0][1]) - 4.44) < 0.01, rows[0]
    source = "Robert J. Shiller, U.S. Stock Markets 1871-Present and CAPE Ratio, shillerdata.com (ie_data.xls)"
    _write("shiller_monthly.csv", ["yyyymm", *order[1:]], rows, source)


def _french_sections(raw: Path, name: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Split a French CSV into (title, header, rows) sections."""
    with zipfile.ZipFile(raw / name) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    sections: list[tuple[str, list[str], list[list[str]]]] = []
    title = ""
    for line in io.StringIO(text):
        cells = [c.strip() for c in line.rstrip("\r\n").split(",")]
        if len(cells) > 1 and cells[0] == "":
            sections.append((title, cells[1:], []))
        elif len(cells) > 1 and re.fullmatch(r"\d{4,8}", cells[0]) and sections:
            sections[-1][2].append(cells)
        elif cells[0]:
            title = cells[0]
    return sections


def _monthly(raw: Path, name: str, title: str = "") -> tuple[list[str], list[list[str]]]:
    for t, header, rows in _french_sections(raw, name):
        if rows and len(rows[0][0]) == 6 and title in t:
            return header, rows
    raise ValueError(f"no monthly section {title!r} in {name}")


def write_french(raw: Path) -> None:
    credit = "Kenneth R. French, Data Library (mba.tuck.dartmouth.edu/pages/faculty/ken.french), built from CRSP and Compustat; returns in percent"
    header, rows = _french_sections(raw, "ff_daily.zip")[0][1:]
    i_mkt, i_rf = header.index("Mkt-RF") + 1, header.index("RF") + 1
    _write("ff_market_daily.csv", ["yyyymmdd", "mkt_rf", "rf"], [[r[0], r[i_mkt], r[i_rf]] for r in rows], credit + "; daily")

    h3, r3 = _monthly(raw, "ff3.zip")
    h5, r5 = _monthly(raw, "ff5.zip")
    hm, rm = _monthly(raw, "momentum.zip")
    five = {r[0]: r[1:] for r in r5}
    mom = {r[0]: r[1] for r in rm}
    names5 = [h.lower().replace("-", "_") for h in h5]
    rows = []
    for r in r3:
        ym = r[0]
        f5 = dict(zip(names5, five[ym], strict=True)) if ym in five else {}
        rows.append([ym, r[1], r[2], r[3], f5.get("rmw", ""), f5.get("cma", ""), mom.get(ym, ""), r[4]])
    assert h3 == ["Mkt-RF", "SMB", "HML", "RF"] and hm[0].strip() == "Mom"
    _write("ff_factors_monthly.csv", ["yyyymm", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"], rows, credit + "; monthly; RMW and CMA from 1963-07")

    h25, r25 = _monthly(raw, "ff25.zip", "Value Weighted")
    assert len(h25) == 25 and h25[0] == "SMALL LoBM" and h25[-1] == "BIG HiBM", h25
    start = next(i for i, r in enumerate(r25) if r[0] >= "196307")
    names25 = [f"me{i}_bm{j}" for i in range(1, 6) for j in range(1, 6)]
    _write("ff25_size_value_monthly.csv", ["yyyymm", *names25], r25[start:], credit + "; monthly value-weighted 5x5 size-by-book-to-market portfolios")


def write_fred(raw: Path) -> None:
    with (raw / "fred_treasury.csv").open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header[1:] == list(FRED_TENORS), header
        last: dict[str, list[str]] = {}
        for row in reader:
            if all(row[1:]) and "." not in row[1:]:
                last[row[0][:7]] = row  # keep the last complete day of each month
    this_month = datetime.date.today().strftime("%Y-%m")  # still incomplete
    rows = [[ym.replace("-", ""), *row[1:]] for ym, row in sorted(last.items()) if ym != this_month]
    cols = ["yyyymm", *[t.lower().replace("dgs", "y") for t in FRED_TENORS]]
    source = "FRED (Federal Reserve Bank of St. Louis), H.15 constant-maturity Treasury yields, percent, last complete business day of each month"
    _write("treasury_yields_monthly.csv", cols, rows, source)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", type=Path, help="directory holding (or receiving) the downloaded files")
    p.add_argument("--offline", action="store_true", help="use the files already in --raw")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = args.raw or Path(tmp)
        raw.mkdir(parents=True, exist_ok=True)
        if not args.offline:
            download(raw)
        write_shiller(raw)
        write_french(raw)
        write_fred(raw)


if __name__ == "__main__":
    main()
